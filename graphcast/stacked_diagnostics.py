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

from graphcast.stacked_predictor_base import StackedPredictor, StackedLossAndChannelLoss
from graphcast.stacked_normalization import normalize, unnormalize, StackedInputsAndResiduals
from graphcast import xarray_tree
import xarray

class StackedInputsResidualsDiagnostics(StackedInputsAndResiduals):
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
        stddev_by_level: dict[chex.Array, chex.Array],
        mean_by_level: dict[chex.Array, chex.Array],
        diffs_stddev_by_level: dict[chex.Array, chex.Array],
        last_input_channel_mapping: dict,
        mappings: dict,
        masks: dict,
    ):
        assert not isinstance(predictor, StackedInputsAndResiduals)
        self._predictor = predictor
        self._scales = stddev_by_level
        self._locations = mean_by_level
        self._residual_scales = diffs_stddev_by_level
        self._residual_locations = {"inputs": None, "targets": None, "diagnostics": None}
        self._last_input_channel_mapping = last_input_channel_mapping

        self._checkit(self._scales)
        self._checkit(self._locations)
        self._checkit(self._residual_scales)

    @staticmethod
    def _checkit(attr):
        if attr is not None:
            assert isinstance(attr, dict)
            assert "inputs" in attr.keys()
            assert "targets" in attr.keys()
            assert "diagnostics" in attr.keys()
            assert len(attr.keys()) == 3

    def calc_diagnostics(self, inputs, outputs):
        return jnp.concatenate(
            [func(inputs, outputs, self.masks) for func in self.mappings.values()],
            axis=-1,
        )

    def normalize_diagnostics(self, outputs):
        return normalize(outputs, self._scales["diagnostics"], self._locations["diagnostics"])

    def __call__(
        self,
        inputs: chex.Array,
    ) -> chex.Array:
        predictions = super().__call__(inputs)
        diagnostics = self.calc_diagnostics(inputs, predictions)
        return jnp.concatenate(
            [predictions, diagnostics],
            axis=-1,
        )

    def loss(
        self,
        inputs: chex.Array,
        targets: chex.Array,
        weights: chex.Array,
    ) -> StackedLossAndChannelLoss:
        """Returns the loss computed on normalized inputs and targets."""
        (loss, loss_by_channel), _ = self.loss_and_predictions(inputs, targets, weights)
        return loss, loss_by_channel

    def loss_and_predictions(  # pytype: disable=signature-mismatch  # jax-ndarray
        self,
        inputs: chex.Array,
        targets: chex.Array,
        weights: chex.Array,
    ) -> Tuple[StackedLossAndChannelLoss, chex.Array]:
        """Note that the weights have to include the diagnostic channels too"""

        # prepare normalized predictions with normalized diagnostics
        norm_predictions = self.normalized_predict(inputs)
        predictions = self._unnormalize_prediction_and_add_input(inputs, norm_predictions)
        prediction_diagnostics = self.calc_diagnostics(inputs, predictions)

        norm_predictions = jnp.concatenate(
            [norm_predictions, self.normalize_diagnostics(prediction_diagnostics)],
            axis=-1,
        )

        # prepare normalized targets with normalized target diagnostics
        norm_target_residuals = self._subtract_input_and_normalize_target(inputs, targets)
        target_diagnostics = self.calc_diagnostics(inputs, targets)
        norm_targets = jnp.concatenate(
            [norm_target_residuals, self.normalize_diagnostics(target_diagnostics)],
            axis=-1,
        )

        # compute loss
        loss, loss_per_channel = stacked_mse(norm_predictions, norm_targets, weights)
        predictions_with_diagnostics = jnp.concatenate(
            [predictions, prediction_diagnostics],
            axis=-1,
        )
        return (loss, loss_per_channel), predictions_with_diagnostics
