# Copyright 2023 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import xarray
from graphcast import graphcast, xarray_tree

class ClipCast(graphcast.GraphCast):
  """Clip values after prediction

  """

  def __init__(self, model_config: graphcast.ModelConfig, task_config: graphcast.TaskConfig):
    """Initializes the predictor."""
    super().__init__(model_config, task_config)
    self.clip_min = {
        "spfh": 0.,
        "o3mr": 0.,
        "clwmr": 0.,
        "grle": 0.,
        "icmr": 0.,
        "rwmr": 0.,
        "snmr": 0.,
        "ntrnc": 0.,
        "nicp": 0.,
        "soill1": 0.,
        "soilw1": 0.,
        "snod": 0.,
        "weasd": 0.,
        "f10m": 0.,
        "sfcr": 0.,
        "prateb_ave": 0.,
        "tmp": 0.,
        "tmp2m": 0.
    }
    self.clip_max = {
        "soill1": 1.,
        "soilw1": 1.,
    }


  def __call__(self,
               inputs: xarray.Dataset,
               targets_template: xarray.Dataset,
               forcings: xarray.Dataset,
               is_training: bool = False,
               ) -> xarray.Dataset:
    predictions = super().__call__(
        inputs=inputs,
        targets_template=targets_template,
        forcings=forcings,
        is_training=is_training,
    )
    #def clipvals(preds, clip_min, clip_max):
    if not set(self.clip_min.keys()).issubset(set(predictions.keys())):
      raise ValueError(
          'Passing a weight that does not correspond to any variable '
          f'{set(self.clip_min.keys())-set(predictions.keys())}')
    if not set(self.clip_max.keys()).issubset(set(predictions.keys())):
      raise ValueError(
          'Passing a weight that does not correspond to any variable '
          f'{set(self.clip_max.keys())-set(predictions.keys())}')
    for key, minval in self.clip_min.items():
      predictions[key] = predictions[key].where(predictions[key] > minval, minval).astype(predictions[key].dtype)

    for key, maxval in self.clip_max.items():
      predictions[key] = predictions[key].where(predictions[key] < maxval, maxval).astype(predictions[key].dtype)
    return predictions

    #return clipvals(predictions, self.clip_min, self.clip_max)
