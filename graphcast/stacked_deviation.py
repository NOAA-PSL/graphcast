"""Wrappers for Predictors which allow them to work with normalized data.

The Predictor which is wrapped sees normalized inputs and targets, and makes
normalized predictions. The wrapper handles translating the predictions back
to the original domain.

Note:
    * the loss does not pass through the half precision casting, if it exists, although the predictions do get cast to half precision
"""

import logging
import chex
import jax.numpy as jnp
from typing import Optional, Tuple

from graphcast.losses import stacked_mse
from graphcast.stacked_predictor_base import StackedPredictor, StackedLossAndChannelLoss
from graphcast.stacked_normalization import normalize, unnormalize, StackedInputsAndResiduals
from graphcast import xarray_tree
import xarray

class StackedInputsResidualsDeviations(StackedInputsAndResiduals):
    """In addition to the normalization and residual framework, compute deviation loss term

    In terms of precision, this is similar to the diagnostics loss, because we have to do some
    custom stuff in the loss_and_predictions function.
    """

    def __init__(
        self,
        predictor: StackedPredictor, # either StackedBfloat16Cast or StackedGraphCast
        stddev_by_level: dict,
        mean_by_level: dict,
        diffs_stddev_by_level: dict,
        deviation_stddev_by_level: dict,
        last_input_channel_mapping: dict,
    ):
        super().__init__(
            predictor=predictor,
            stddev_by_level=stddev_by_level,
            mean_by_level=mean_by_level,
            diffs_stddev_by_level=diffs_stddev_by_level,
            last_input_channel_mapping=last_input_channel_mapping,
        )

        self._deviation_locations = None
        self._deviation_scales = deviation_stddev_by_level
        self._checkit(self._deviation_scales)

    def __call__(
        self,
        inputs: chex.Array,
    ) -> chex.Array:
        """Note that __call__ works on both ensemble members in the pair,
        but self.normalized_predict works on a single ensemble member...
        Reason being:
            * this one is just so we can use all the same machinery for "construct_wrapped_graphcast" and "init_model"
            * the other is more of a utility function for this and the loss function, should have a preceding underscore

        Returns shape
        [n_samples, n_members=2, n_latitude, n_longitude, n_channels]
        """
        predictions = []
        for this_input in [inputs[:, 0, ...], inputs[:, 1, ...]]:
            this_norm_prediction = self.normalized_predict(this_input)
            this_prediction = self._unnormalize_prediction_and_add_input(this_input, this_norm_prediction)
            predictions.append(
                this_prediction[None]
            )

        predictions = jnp.concatenate(predictions).swapaxes(0, 1)
        return predictions

    def normalize_deviations(self, deviations):
        return normalize(deviations, self._deviation_scales["targets"], self._deviation_locations["targets"])

    def loss_and_predictions(  # pytype: disable=signature-mismatch  # jax-ndarray
        self,
        inputs: tuple[chex.Array],
        targets: tuple[chex.Array],
        loss_weights: dict[chex.Array],
    ) -> Tuple[StackedLossAndChannelLoss, chex.Array]:
        """
        I'm hackily assuming that inputs and targets are tuples of length 2 with the ICs and targets
        for the ensemble member pairs

        Note that inputs and targets have shape:
            [n_samples_per_batch, n_members, n_latitude, n_longitude, n_channels]
        """

        # compute normalized and unnormalized prediction for each initial condition
        predictions = []
        forecast_mse_per_member = []
        forecast_mse_per_member_per_channel = []
        for this_input, this_target in zip(
            [inputs[:,0,...], inputs[:,1,...]],
            [targets[:,0,...], targets[:,1,...]],
        ):

            # first, compute forecast and MSE in normalized space
            this_norm_prediction = self.normalized_predict(this_input)
            this_norm_target_residual = self._subtract_input_and_normalize_target(this_input, this_target)

            loss1, loss2 = stacked_mse(
                this_norm_prediction,
                this_norm_target_residual,
                loss_weights["forecast_mse"],
            )
            forecast_mse_per_member.append(loss1)
            forecast_mse_per_member_per_channel.append(loss1)

            # now get unnormalized predictions for deviation
            predictions.append(
                self._unnormalize_prediction_and_add_input(this_input, this_norm_predictions)
            )

        # compute deviations in un-normalized space
        prediction_deviations = predictions[1] - predictions[0]
        norm_prediction_deviations = self.normalize_deviations(prediction_deviations)

        target_deviations = targets[1] - targets[0]
        norm_target_deviations = self.normalize_deviations(target_deviations)

        # MSE loss of deviations
        deviation_mse, deviation_mse_per_channel = stacked_mse(
            norm_prediction_deviations,
            norm_target_deviations,
            loss_weights["deviation_mse"],
        )

        # put it all together now
        loss = jnp.sum(forecast_mse_per_member, axis=0) + deviation_mse
        loss_per_channel = {
            "forecast_mse": jnp.sum(forecast_mse_per_member_per_channel, axis=0),
            "deviation_mse": deviation_mse_per_channel,
        }
        return (loss, loss_per_channel), predictions
