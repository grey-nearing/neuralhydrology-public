from typing import Tuple

import torch

from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.utils.config import Config
from neuralhydrology.modelzoo.fc import FC
from neuralhydrology.modelzoo.inputlayer import InputLayer


class SuperFlex(BaseModel):
    """..."""

    def __init__(
        self,
        cfg: Config
    ):
        super(SuperFlex, self).__init__(cfg=cfg)

        if len(cfg.target_variables) > 1:
            raise ValueError("Currently, bucket models only support a single target variable.")

       # We require exactly two mass inputs. This is trivial to change.
        self._n_mass_vars = len(cfg.mass_inputs)
        if self._n_mass_vars != 4:
            raise ValueError("Superflex requires exactly three mass inputs: P, E, & Tmin, Tmax.")

        # The universal embedding network. This network can't transform the dynamic inputs
        # because that would change units of the conserved inputs (P & E).
        # This embedding network could be re-designed so that it acts on just statics, or on
        # statics and auxilary inputs. Auxilary inputs are dynamic inputs other than the mass-
        # conserved inputs.
        if cfg.dynamics_embedding is not None:
            raise ValueError("Embedding for dynamic inputs is not supported with the current bucket-style models.")

        self.embedding_net = InputLayer(cfg)

        # Build the parameterization network. This takes static catchment attributes as inputs and estiamtes
        # all of the parameters for all of the different model components at once. Parameters will be extracted
        # from the output vector of this model sequentially.
        statics_input_size = len(cfg.static_attributes + cfg.hydroatlas_attributes + cfg.evolving_attributes)
        # If the user requests using one-hot encoding to identify basins, this will be added to static inputs.
        if cfg.use_basin_id_encoding:
            statics_input_size += cfg.number_of_basins

        # Just assigning these by hand here for a quick test. The full model specification should be added
        # to the config file. This will require automating graph development.
        total_parameters = 12
        hidden_sizes = [20]
        hidden_sizes.append(total_parameters)

        self.parameterization = FC(
            input_size=statics_input_size,
            hidden_sizes=hidden_sizes,
            dropout=0.,
        )

        # Hand-defined graph strucutre. Automate this.
        # This ad hoc example has two thresholds in parallel and three routers in sequence. 
        num_threshold_reservoirs = 2
        num_routing_reservoirs = 3
        self.evap_partitioner = FluxPartition(cfg=cfg, num_inputs=statics_input_size+4, num_outputs=num_threshold_reservoirs)
        self.melt_partitioner = FluxPartition(cfg=cfg, num_inputs=statics_input_size+4, num_outputs=num_threshold_reservoirs)
        self.liquid_partitioner = FluxPartition(cfg=cfg, num_inputs=statics_input_size+4, num_outputs=num_threshold_reservoirs)
        self.snow_reservoir = SnowReservoir(cfg=cfg)
        self.threshold_reservoirs = [ThresholdReservoir(cfg=cfg) for _ in range(num_threshold_reservoirs)]
        self.routing_reservoirs = [RoutingReservoir(cfg=cfg) for _ in range(num_routing_reservoirs)]
        self.lag_function = LagFunction(cfg=cfg, timesteps=6)

    def _execute_graph(
        self,
        parameters: torch.Tensor,
        x_s: torch.Tensor,
        precip: torch.Tensor,
        tmin: torch.Tensor,
        tmax: torch.Tensor,
        evap: torch.Tensor
    ) -> torch.Tensor:
        """..."""
        
        # Initialize parameter extractor.
        parameter_idx = 0

        # 1) Snow Bucket(s).
        liquid, melt = self.snow_reservoir(x_s=x_s, precip=precip, tmin=tmin, tmax=tmax, rate=parameters[:, parameter_idx])
        parameter_idx += 1
        
        # 2) Partition fluxes.
        flux_partition_inputs = torch.cat([x_s, precip, tmin, tmax, evap], dim=-1)
        evaps = self.evap_partitioner(x=flux_partition_inputs, flux=evap)
        liquids = self.evap_partitioner(x=flux_partition_inputs, flux=liquid)
        melts = self.evap_partitioner(x=flux_partition_inputs, flux=melt)

        # 3) Execute parallel threshold reservoirs.        
        output = []
        for idx, threshold_reservoir in enumerate(self.threshold_reservoirs):
            x_in = torch.unsqueeze(liquids[:, idx] + melts[:, idx], dim=-1)
            x_out = torch.unsqueeze(evaps[:, idx], dim=-1)
            output.append(threshold_reservoir(x_in=x_in, x_out=x_out, height=parameters[:, parameter_idx]))
            parameter_idx += 1
        output = torch.sum(torch.stack(output, dim=1), dim=1)
        
        # 4) Execute sequence routing reservoirs.
        for routing_reservoir in self.routing_reservoirs:
            output = routing_reservoir(x_in=output, rate=parameters[:, parameter_idx])
            parameter_idx += 1

        # 5) Execute lag.
        output = self.lag_function(x_in=output, weights=parameters[:, parameter_idx:])

        return output
        

    def forward(
        self,
        data: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """..."""
        # Prepare inputs through any universal embedding layer that might exist.
        # Since we don't allow embedding transforms in the current version, really
        # all this is doing is separating the dynamic and static inputs in a
        # way that conforms to NeuralHydrology convention.
        #  - x_d are dynamic inputs.
        #  - x_s are static inputs.
        x_d, x_s = self.embedding_net(data, concatenate_output=False)

        # Dimensions.
        time_steps, batch_size, _ = x_d.shape

        # Estimate model parameters.
        parameters = self.parameterization(x=x_s)

        # Initialize storage in all model components.
        self.snow_reservoir.initialize_bucket(batch_size=batch_size, device=x_d.device)
        [reservoir.initialize_bucket(batch_size=batch_size, device=x_d.device) for reservoir in self.threshold_reservoirs]
        [reservoir.initialize_bucket(batch_size=batch_size, device=x_d.device) for reservoir in self.routing_reservoirs]
        self.lag_function.initialize_bucket(batch_size=batch_size, device=x_d.device)

        # Execute time loop.
        output = []
        for t in range(time_steps):

            # We assume that the dynamic inputs are in the following order.
            # TODO: Enforce this.
            precip = torch.unsqueeze(x_d[t, :, 0], -1)
            evap = torch.unsqueeze(x_d[t, :, 1], -1)
            tmin = torch.unsqueeze(x_d[t, :, 2], -1)
            tmax = torch.unsqueeze(x_d[t, :, 3], -1)

            output.append(
                self._execute_graph(
                    parameters=parameters,
                    x_s=x_s,
                    precip=precip,
                    tmin=tmin,
                    tmax=tmax,
                    evap=evap,
                )
            )

        return {'y_hat': torch.stack(output, 1)}


class ThresholdReservoir(torch.nn.Module):
    """Initialize a threshold bucket node.

    A threshold bucket is a bucket with finite height and no drain. Outflow is from overflow.
    It has one parameter: bucket height. The parameter is treated dynamically, instead of
    as a fixed parameter, which allows the parameter to be either learned or estimated with
    an external parameterization network. The threshold bucket has prescribed source and sink
    fluxes.

    Parameters
    ----------
    cfg : Config
        Configuration of the run, read from the config file with some additional keys (such as
        number of basins).
    """

    def __init__(
        self,
        cfg: Config
    ):
        super(ThresholdReservoir, self).__init__()
        self.number_of_parameters = 1

    def initialize_bucket(
        self,
        batch_size: int,
        device: torch.device
    ) -> torch.Tensor:
        """Initializes a bucket during runtime.
        
        Initialization must happen at runtime so that we know the batch size and device.
        """
        # We can do this however we want, but for now let's just start with every bucket being empty.
        self.storage = torch.zeros([batch_size, 1], device=device)
        
        # Initialize a tensor of zeros to use in the threshold calculation.
        self._zeros = torch.zeros_like(self.storage)

    def forward(
        self,
        x_in: torch.Tensor,
        x_out: torch.Tensor,
        height: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass for a threshold reservoir."""
        # Account for prescribed fluxes (e.g., E, P)
        self.storage = self.storage.clone() + x_in
        self.storage = self.storage.clone() - torch.minimum(x_out, self.storage)

        # Ensure that the bucket height parameter is positive.
        height = torch.unsqueeze(torch.abs(height), dim=-1)

        # Calculate bucket overflow.
        overflow = torch.maximum(self.storage - height, self._zeros)
        self.storage = self.storage.clone() - overflow

        return overflow


class RoutingReservoir(torch.nn.Module):
    """Initialize a routing bucket node.

    A routing bucket is a bucket with infinite height and a drain.
    It has one parameter: outflow rate. The parameter is treated dynamically, instead of
    as a fixed parameter, which allows the parameter to be either learned or estimated with
    an external parameterization network. The routing bucket only has a prescribed source
    flux.

    Parameters
    ----------
    cfg : Config
        Configuration of the run, read from the config file with some additional keys (such as
        number of basins).
    """

    def __init__(
        self,
        cfg: Config
    ):
        super(RoutingReservoir, self).__init__()
        self.number_of_parameters = 1

    def initialize_bucket(
        self,
        batch_size: int,
        device: torch.device
    ) -> torch.Tensor:
        """Initializes a bucket during runtime.
        
        Initialization must happen at runtime so that we know the batch size and device.
        """
        # We can do this however we want, but for now let's just start with every bucket being empty.
        self.storage = torch.zeros([batch_size, 1], device=device)

    def forward(
        self,
        x_in: torch.Tensor,
        rate: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass for a routing reservoir."""
        # Account for the source flux.
        self.storage = self.storage.clone() + x_in

        # Ensure that the bucket rate parameter is in (0, 1).
        rate = torch.unsqueeze(torch.sigmoid(rate), dim=-1)

        # Outflow from leaky bucket.
        outflow = rate * self.storage
        self.storage = self.storage - outflow

        return outflow


class SnowReservoir(torch.nn.Module):
    """Initialize a snow bucket node.

    A snow bucket is a bucket with infinite height and a drain, where the input is partitioned
    into one flux that goes into the bucket and one flux that misses the bucket. This bucket has
    two parameters: flux partition and outflow rate. The parameters are treated dynamically, instead
    of being fixed, which allows them to be either learned or estimated with an external
    parameterization network. The snow bucket only has a prescribed source flux.

    Parameters
    ----------
    cfg : Config
        Configuration of the run, read from the config file with some additional keys (such as
        number of basins).
    """

    def __init__(
        self,
        cfg: Config
    ):
        super(SnowReservoir, self).__init__()
        self.number_of_parameters = 1

        # Network for converting temperature, precip, and static attributes into a partitioning
        # coefficient between liquid and solid precip.
        statics_input_size = len(cfg.static_attributes + cfg.hydroatlas_attributes + cfg.evolving_attributes)
        if cfg.use_basin_id_encoding:
            statics_input_size += cfg.number_of_basins

        self.precip_partitioning = FluxPartition(
            cfg=cfg,
            num_inputs=3+statics_input_size,
            num_outputs=2,
        )

    def initialize_bucket(
        self,
        batch_size: int,
        device: torch.device
    ) -> torch.Tensor:
        """Initializes a bucket during runtime.
        
        Initialization must happen at runtime so that we know the batch size and device.
        """
        # We can do this however we want, but for now let's just start with every bucket being empty.
        self.storage = torch.zeros([batch_size, 1], device=device)

    def forward(
        self,
        x_s: torch.Tensor,
        precip: torch.Tensor,
        tmin: torch.Tensor,
        tmax: torch.Tensor,
        rate: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for a snow reservoir."""
        # Partition the in-flux.
        partition = self.precip_partitioning(x=torch.cat([x_s, tmin, tmax, precip], dim=-1), flux=precip)
        miss_flux = torch.unsqueeze(partition[:, 0], -1)
        self.storage = self.storage.clone() + torch.unsqueeze(partition[:, 1], -1)

        # Outflow from leaky bucket is snowmelt.
        # The rate parameter is in (0, 1).
        rate = torch.unsqueeze(torch.sigmoid(rate), dim=-1)       
        snowmelt = rate * self.storage
        self.storage = self.storage.clone() - snowmelt

        return miss_flux, snowmelt


class LagFunction(torch.nn.Module):
    """Initialize a lag function.

    A generic lag function as a convolution. We use a storage vector to make this easy to plug into
    a model graph that operates over a single timestep.

    Parameters
    ----------
    cfg : Config
        Configuration of the run, read from the config file with some additional keys (such as
        number of basins).
    """

    def __init__(
        self,
        cfg: Config,
        timesteps: int
    ):
        super(LagFunction, self).__init__()
        # The number of timesteps in a lag function must be set.
        self.timesteps = timesteps
        self.number_of_parameters = timesteps

    def initialize_bucket(
        self,
        batch_size: int,
        device: torch.device
    ) -> torch.Tensor:
        """Initializes a bucket during runtime.
        
        Initialization must happen at runtime so that we know the batch size, convolution width,
        and device.
        """
        # We can do this however we want, but for now let's just start with every bucket being empty.
        self.storage = torch.zeros([batch_size, self.timesteps], device=device)

    def forward(
        self,
        x_in: torch.Tensor,
        weights: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass for a lag function."""
        if weights.shape[-1] != self.timesteps:
            raise ValueError(f'Convolution weights must be the same dimenions as the conv filter. '
                             f'Expected{self.timesteps}, received {weights.shape[-1]}.')

        # Add to the storage in the filter.
        self.storage = self.storage.clone() + x_in * weights

        # Shift the filter.
        outflow = torch.unsqueeze(self.storage[:, -1], dim=-1)
        self.storage[:, 1:] = self.storage[:, :-1].clone()
        self.storage[:, 0] = self.storage.new_zeros([self.storage.shape[0]])

        return outflow


class FluxPartition(torch.nn.Module):
    """Fully connected layer with N normalized outputs.

    Parameters
    ----------
    cfg : Config
        Configuration of the run, read from the config file with some additional keys (such as number of basins).
    num_inputs: Number of inputs to the fully connected layer.
    num_outputs: Number of normalized outputs.

    Raises
    ------
    ValueError
        Given an unsupported ativation function.
    ----------
    """
    def __init__(
        self,
        cfg: Config,
        num_inputs: int,
        num_outputs: int,
        activation: str = 'sigmoid'
    ):
        super(FluxPartition, self).__init__()
        
        self.fc = torch.nn.Linear(in_features=num_inputs, out_features=num_outputs)

        if activation.lower() == "sigmoid":
            self.activation = torch.nn.Sigmoid()
        elif activation.lower() == "relu":
            self.activation = torch.nn.ReLU()
        else:
            raise ValueError(
                f'Flux partitioning currently only works with sigmoid and relu activation functions. '
                f'Got {activation}. This is necessary becasue the activations must always be positive.'
            )

        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.orthogonal_(self.fc.weight)
        torch.nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor, flux: torch.Tensor) -> torch.Tensor:
        """Perform forward pass through the normalized gate"""
        weigths = self.activation(self.fc(x))
        normalized_weights = torch.nn.functional.normalize(weigths, p=1, dim=-1)

        if flux.shape[-1] != 1:
            raise ValueError('FluxPartition network can only partition a scaler.')

        return normalized_weights * flux
