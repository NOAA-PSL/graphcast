
import xarray
import chex
import jax.numpy as jnp
import numpy as np

from graphcast import graphcast
from graphcast import predictor_base
from graphcast import xarray_jax

class StackedGraphCast(graphcast.GraphCast):

    def __init__(
        self,
        model_config: graphcast.ModelConfig,
        task_config: graphcast.TaskConfig
        ):
        super().__init__(model_config=model_config, task_config=task_config)

        # since we don't use xarray DataArrays as inputs, we have to
        # establish the grid somehow. Seems easiest to pass it via task_config
        # just like the pressure levels
        self._init_grid_properties(
            grid_lat=np.array(task_config.latitude),
            grid_lon=np.array(task_config.longitude),
        )


    def __call__(
        self,
        inputs: chex.Array,
        ) -> chex.Array:

        self._maybe_init()

        # Convert all input data into flat vectors for each of the grid nodes.
        # xarray (batch, time, lat, lon, level, multiple vars, forcings)
        # -> [num_grid_nodes, batch, num_channels]
        grid_node_features = self._inputs_to_grid_node_features(inputs)

        # Transfer data for the grid to the mesh,
        # [num_mesh_nodes, batch, latent_size], [num_grid_nodes, batch, latent_size]
        (latent_mesh_nodes, latent_grid_nodes
         ) = self._run_grid2mesh_gnn(grid_node_features)

        # Run message passing in the multimesh.
        # [num_mesh_nodes, batch, latent_size]
        updated_latent_mesh_nodes = self._run_mesh_gnn(latent_mesh_nodes)

        # Transfer data frome the mesh to the grid.
        # [num_grid_nodes, batch, output_size]
        output_grid_nodes = self._run_mesh2grid_gnn(
            updated_latent_mesh_nodes, latent_grid_nodes)

        # Conver output flat vectors for the grid nodes to the format of the output.
        # [num_grid_nodes, batch, output_size] ->
        # xarray (batch, one time step, lat, lon, level, multiple vars)
        return self._grid_node_outputs_to_prediction(output_grid_nodes)

    def loss_and_predictions(
        self,
        inputs: chex.Array,
        targets: chex.Array,
        ) -> tuple[predictor_base.LossAndDiagnostics, chex.Array]:
        # Forward pass
        predictions = self(inputs)

        # bump to xarray.DataArray in order to hookup to losses module
        dims = ("lat", "lon", "channels")

        # Compute loss
        loss = losses.weighted_mse_per_level(
            predictions,
            targets,
            per_variable_weights=dict(),
        )
        return loss, predictions


    def _maybe_init(self):
        if not self._initialized:
            self._init_mesh_properties()
            # grid properties initialized at __init__
            self._grid2mesh_graph_structure = self._init_grid2mesh_graph()
            self._mesh_graph_structure = self._init_mesh_graph()
            self._mesh2grid_graph_structure = self._init_mesh2grid_graph()

            self._initialized = True


    def _inputs_to_grid_node_features(
        self,
        inputs: chex.Array,
        ) -> chex.Array:
        """inputs expected to be as [lat, lon, ...]"""

        shape = (-1,) + inputs.shape[2:]
        result = xarray_jax.unwrap(inputs)
        result = result.reshape(shape)
        return result

    def _grid_node_outputs_to_prediction(
        self,
        grid_node_outputs: chex.Array,
        ) -> chex.Array:
        """returned as [lat, lon, ...]"""

        assert self._grid_lat is not None and self._grid_lon is not None
        grid_shape = (self._grid_lat.shape[0], self._grid_lon.shape[0])

        # result is [lat, lon, batch, channels]
        result = grid_node_outputs.reshape(
            grid_shape + grid_node_outputs.shape[1:],
        )
        return result
