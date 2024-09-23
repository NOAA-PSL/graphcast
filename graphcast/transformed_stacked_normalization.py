import chex
from typing import Optional, Tuple

from graphcast.stacked_predictor_base import StackedPredictor, StackedLossAndDiagnostics
from graphcast.stacked_normalization import StackedInputsAndResiduals

def transform(
    values: chex.Array,
    transforms: dict,
) -> chex.Array:
    """transforms should be a mapping from channel_index -> transform_function
    so that we return the transformed value for each channel_index
    """
    result = values.copy()
    for channel, transform_function in transforms.items():
        result = result.at[..., channel].set(transform_function(values[..., channel]))
    return result

class TransformedStackedInputsAndResiduals(StackedInputsAndResiduals):
    """Same as StackedInputsAndResiduals, but perform a transform function on
    inputs, and inverse transform on output

    Note that this does NOT transform any of the normalization statistics, since
    this may not be the appropriate thing to do if the mapping is nonlinear.
    It is assumed that the user has thought about this.
    """

    def __init__(
        self,
        predictor: StackedPredictor,
        stddev_by_level: dict[chex.Array, chex.Array],
        mean_by_level: dict[chex.Array, chex.Array],
        diffs_stddev_by_level: dict[chex.Array, chex.Array],
        last_input_channel_mapping: dict,
        input_transforms: dict,
        output_transforms: dict,
        target_transforms: dict,
        ):
        super().__init__(
            predictor=predictor,
            stddev_by_level=stddev_by_level,
            mean_by_level=mean_by_level,
            diffs_stddev_by_level=diffs_stddev_by_level,
            last_input_channel_mapping=last_input_channel_mapping,
        )
        self.input_transforms = input_transforms
        self.output_transforms = output_transforms
        self.target_transforms = target_transforms

    def __call__(
        self,
        inputs: chex.Array,
        **kwargs
        ) -> chex.Array:
        transformed_inputs = transform(inputs, self.input_transforms)
        transformed_predictions = super().__call__(transformed_inputs, **kwargs)
        predictions = transform(transformed_predictions, self.output_transforms)
        return predictions

    def loss(
        self,
        inputs: chex.Array,
        targets: chex.Array,
        **kwargs,
        ) -> StackedLossAndDiagnostics:
        """Returns the loss computed on normalized inputs and targets."""
        transformed_inputs = transform(inputs, self.input_transforms)
        transformed_targets = transform(targets, self.target_transforms)
        return super().loss(transformed_inputs, transformed_targets, **kwargs)

    def loss_and_predictions(  # pytype: disable=signature-mismatch  # jax-ndarray
        self,
        inputs: chex.Array,
        targets: chex.Array,
        **kwargs,
        ) -> Tuple[StackedLossAndDiagnostics, chex.Array]:
        """Returns the loss computed on normalized inputs and targets."""
        transformed_inputs = transform(inputs, self.input_transforms)
        transformed_targets = transform(targets, self.target_transforms)
        (loss, scalars), transformed_predictions = super().loss_and_predictions(
            transformed_inputs,
            transformed_targets,
        )
        predictions = transform(transformed_predictions, self.output_transforms)
        return (loss, scalars), predictions
