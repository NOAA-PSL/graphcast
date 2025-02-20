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
    """In addition to the normalization and residual framework, compute diagnostics
    from predictions and targets that get added to the loss function.

    This is tricky because:
    * loss is computed in normalized space
    * diagnostics are computed in un-normalized space

    So the loss function has to have access to
    * predictions: normalized and un-normalized
    * targets: same
    * predicted_diagnostics: normalized (and un-normalized for the routine returning predictions)
    * predicted_targets: normalized

    While it is easy to pass the prediction routine through the Bfloat16 casting,
    it is not easy to do this for the loss function calculation...
    So for now, predictions might be in half precision, but loss is in single.
    This should be fine, esp. since the prediction part is where memory matters most anyway.
    """

    def __init__(
        self,
        predictor: StackedPredictor, # either StackedBfloat16Cast or StackedGraphCast
        stddev_by_level: dict,
        mean_by_level: dict,
        diffs_stddev_by_level: dict,
        spread_by_level: chex.Array, # since this is only for the target space
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
        self._deviation_scales = spread_by_level
        self._checkit(self._deviation_scales)

    def normalize_deviations(self, deviations):
        return normalize(deviations, self._deviation_scales, self._deviation_locations)

    def loss(
        self,
        inputs: tuple[chex.Array],
        targets: tuple[chex.Array],
        weights: chex.Array,
        deviation_weights: chex.Array,
    ) -> StackedLossAndChannelLoss:
        (loss, loss_per_channel), _ = self.loss_and_predictions(inputs, targets, weights, deviation_weights)
        return loss, loss_per_channel

    def _isel(index, array):
        return array[:, index, ...]

    def loss_and_predictions(  # pytype: disable=signature-mismatch  # jax-ndarray
        self,
        inputs: tuple[chex.Array],
        targets: tuple[chex.Array],
        weights: chex.Array,
        deviation_weights: chex.Array,
    ) -> Tuple[StackedLossAndChannelLoss, chex.Array]:
        """
        I'm hackily assuming that inputs and targets are tuples of length 2 with the ICs and targets
        for the ensemble member pairs

        Note that inputs and targets have shape:
            [n_samples_per_batch, n_members, n_latitude, n_longitude, n_channels]
        """

        # compute normalized and unnormalized prediction for each initial condition
        norm_predictions = tuple(
            self.normalized_predict(inputs[:,0,...]),
            self.normalized_predict(inputs[:,1,...]),
        )
        predictions = tuple(
            self._unnormalize_prediction_and_add_input(inputs[:, 0, ...], norm_predictions[0]),
            self._unnormalize_prediction_and_add_input(inputs[:, 1, ...], norm_predictions[1]),
        )

        # compute deviation in un-normalized space
        prediction_deviations = predictions[1] - predictions[0]
        norm_prediction_deviations = self.normalize_deviations(prediction_deviations)

        # prepare targets
        norm_target_residuals = tuple(
            self._subtract_input_and_normalize_target(inputs[:, 0, ...], targets[:, 0, ...]),
            self._subtract_input_and_normalize_target(inputs[:, 1, ...], targets[:, 1, ...]),
        )
        target_deviations = targets[1] - targets[0]
        norm_target_deviations = self.normalize_deviations(target_deviations)

        # MSE loss function from each member
        mse_per_member = []
        mse_per_member_per_channel = []
        for pp, tt in zip(norm_predictions, norm_target_residuals):
            loss1, loss2 = stacked_mse(pp, tt, weights)
            mse_per_member.append(loss1)
            mse_per_member_per_channel.append(loss2)

        # MSE loss of deviations
        deviation_mse, deviation_mse_per_channel = stacked_mse(pp, tt, deviation_weights)

        # put it all together now
        loss = jnp.sum(mse_per_member, axis=0) + deviation_mse
        loss_per_channel = jnp.sum(mse_per_member_per_channel, axis=0) + deviation_mse_per_channel
        return (loss, loss_per_channel), predictions
