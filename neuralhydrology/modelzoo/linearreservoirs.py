from typing import Dict

import torch

from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.modelzoo.fc import FC
from neuralhydrology.modelzoo.inputlayer import InputLayer
from neuralhydrology.utils.config import Config


class LinearReservoirs(BaseModel):
    """Initialize a linear reservoir model.

    This model consists of fully connected layers of linear (leaky) buckets. Each bucket is parameterized by a height
    and rate parameter, and those parameters are learned either with a neural network acting on static inputs, or
    trained directly via backpropagation. Each bucket includes a neural network that learns a time-dynamic source/sink
    term representing ET (or any combination of unobserved source/sinks). Between each bucket and the next layer of
    buckets is a flux partitioning network, which defines how the output of an individual bucket is partitioned into
    the next layer of buckets.

    To use this model class, you have to specify the name of the mass input using the `mass_input` config argument.
    Additionally, the mass input and target variable should *not* be normalized. Use the config argument
    `custom_normalization` and set the `centering` and `scaling` key for both to `None` (see
    :doc:`config arguments </usage/config>` for more details on `custom_normalization`).

    Additional config arguments necessary for this include:
        - `bucket_layers`: A list of the number of buckets in each layer.
        - `leaky_bucket_parameterization_hidden_sizes`: A list of the number of neurons in the (fully connected) network
        used to estimate height and rate parameters for each bucket.
        - `leaky_bucket_parameterization_activation`: Activation function applied to all neurons (except the last layer)
        in each bucket's parameterization network.
        - `leaky_bucket_parameterization_dropout`: Dropout rate at the end of each bucket's parameterization network.
        - `leaky_bucket_sink_source_hidden_sizes`: A list of the number of neurons in the (fully connected) network
        used to estimate the time-variable sink/source term for each bucket.
        - `leaky_bucket_sink_source_activation`: Activation function applied to all neurons (except the last layer)
        in each bucket's sink/source network.
        - `leaky_bucket_sink_source_dropout`: Dropout rate at the end of each bucket's sink/source network.
        - `leaky_bucket_flux_partition_hidden_sizes`: A list of the number of neurons in the (fully connected) network
        used to partition the output flux from each bucket.
        - `leaky_bucket_flux_partition_activation`: Activation function applied to all neurons (except the last layer)
        in each bucket's flux partition network.
        - `leaky_bucket_flux_partition_dropout`: Dropout rate at the end of each bucket's flux partition network.

    Parameters
    ----------
    cfg : Config
        Configuration of the run, read from the config file with some additional keys (such as number of basins).

    Raises
    ------
    ValueError
        If no or more than one `mass_input` is specified in the config. 
        If the number of bucket layers is less than 1 or if any bucket layer is empty. 
        If the number of `target_variables` is greater than 1.
        If `dynamics_embedding` is specified in the config, which is (currently) not supported for this model class.
        If there are multiple time frequencies in the data (MTS).
    """
    # Specify submodules of the model that can later be used for finetuning. Names must match class attributes.
    module_parts = ['embedding_net', 'mass_input_partitioning', 'nashlike_cascade', 'flux_partitioning_layers']

    def __init__(self, cfg: Config):

        super(LinearReservoirs, self).__init__(cfg=cfg)    

        if len(cfg.target_variables) > 1:
            raise ValueError("Currently, bucket models only support a single target variable.")

       # For now, we can only handle one mass input. This could be changed later.
        self._n_mass_vars = len(cfg.mass_inputs)
        if self._n_mass_vars > 1:
            raise ValueError("Currently, bucket-style models only support a single mass input.")
        elif self._n_mass_vars == 0:
            raise ValueError("No mass input specified. Specify mass input variable using `mass_inputs` in the config file.")

        # Get the size of different types of inputs from config file.
        # Currently we do not support bucket models in a multiple timestep (MTS) setting.        
        if isinstance(cfg.dynamic_inputs, dict):
            frequencies = list(cfg.dynamic_inputs.keys())
            if len(frequencies) > 1:
                raise ValueError('Bucket models only support single-frequency data')
            dynamics_input_size = len(cfg.dynamic_inputs[frequencies[0]])
        else:
            dynamics_input_size = len(cfg.dynamic_inputs)

        statics_input_size = len(cfg.static_attributes + cfg.hydroatlas_attributes + cfg.evolving_attributes)
        # If the user requests using one-hot encoding to identify basins, this will be added to static inputs.
        if cfg.use_basin_id_encoding:
            statics_input_size += cfg.number_of_basins

        # The universal embedding network. This network can't transform the dynamic inputs
        # because that would change units of the conserved input (precipitation).
        # This embedding network could be re-designed so that it acts on just statics, or on
        # statics and auxilary inputs. Auxilary inputs are dynamic inputs other than the mass-
        # conserved input (precip).
        if cfg.dynamics_embedding is not None:
            raise ValueError("Embedding for dynamic inputs is not supported with the current bucket-style models.")

        self.embedding_net = InputLayer(cfg)

        # We need a flux partitioning network to partition rainfall into the first layer of buckets.
        # Inputs to this network are: (1) all statics, (2) all dynamics, (3) current storages of all
        # buckets in the first layer.
        self.mass_input_partitioning = _FluxPartition(
            num_inputs=dynamics_input_size + statics_input_size + cfg.bucket_layers[0],
            num_outputs=cfg.bucket_layers[0],
            activation=cfg.leaky_bucket_flux_partition_activation,
        )

        # Set up the graph strcuture of leaky buckets. This graph is a series of fully connected layers,
        # where each node is a leaky bucket plus a flux partitioning network to tell how to partition the
        # outflow from each individual bucket into all of the buckets in the next layer.
        if len(cfg.bucket_layers) <= 1:
            raise ValueError("Bucket model has no layers.")
        if any([lyr < 1 for lyr in cfg.bucket_layers]):
            raise ValueError("At least one of the bucket model layers has no buckets.")

        self.nashlike_cascade = []
        self.flux_partitioning_layers = []
        for layer_index, _ in enumerate(cfg.bucket_layers):

            # This is the layer of leaky buckets.
            self.nashlike_cascade.append([_SingleLeakyBucket(cfg) for _ in range(cfg.bucket_layers[layer_index])])
            
            # This is flux partioning for each bucket in this layer of leaky buckets.
            # Inputs to each of these flux partitioning networks are: (1) all statics, (2) all dynamics,
            # (3) current storages of the current bucket, (4) current storages of all buckets in the next layer.
            if layer_index < len(cfg.bucket_layers) - 1:
                self.flux_partitioning_layers.append(
                    [
                        _FluxPartition(
                            num_inputs=dynamics_input_size + statics_input_size + cfg.bucket_layers[layer_index + 1] + 1,
                            num_outputs=cfg.bucket_layers[layer_index + 1],
                            activation=cfg.leaky_bucket_flux_partition_activation,
                        ) for _ in range(cfg.bucket_layers[layer_index])
                    ]
                )
            else:
                # Don't add a flux partitioning network to the last layer. In this case,
                # the outputs from all buckets will be summed to represent streamflow.
                self.flux_partitioning_layers.append(None)
       
    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        # Prepare inputs through any universal embedding layer that might exist.
        # Since we don't allow embedding transforms in the current version, really
        # all this is doing is separating the dynamic and static inputs in a
        # way that conforms to NeuralHydrology convention.
        #  - x_d are dynamic inputs.
        #  - x_s are static inputs.
        x_d, x_s = self.embedding_net(data, concatenate_output=False)

        # x_m are mass inputs, x_a are auxilary inputs. Conservation
        # laws are enforced on mass inputs, while auxilary inputs are
        # used in whatever neural networks exist in the model.
        # The BaseDataset stores the mass input at the beginning of the
        # array. Remeber that there can be only one, so this is a little
        # redundant, but it will allow for possible expansion to
        # multiple mass inputs in the future.
        x_m = x_d[:, :, :self._n_mass_vars]
        x_a = x_d[:, :, self._n_mass_vars:]

        # Dimensions.
        time_steps, batch_size, _ = x_d.shape

        # Initizlize all storage states. Notice that this must be done at runtime
        # instead of init since we do not know the batch size until we get an input
        # tensor.
        previous_storages = []
        for bucket_layer in self.nashlike_cascade:
            for bucket in bucket_layer:
                bucket.init_storage(batch_size=batch_size, device=x_d.device)
            previous_storages.append(torch.cat([bucket.storage for bucket in bucket_layer], dim=-1))

        # Initialize output storage.
        output = x_m.new_zeros([batch_size, time_steps, 1])

        # Run through the time sequence. This is a mega-slow way to unfold this network, but
        # for experiment and development, it's intuitive.
        for t in range(time_steps):
            x_at = x_a[t]
            x_mt = x_m[t]

            # Run through the bucket graph strcuture.
            for layer_idx, (bucket_layer, flux_partinioning_layer) in enumerate(
                zip(self.nashlike_cascade, self.flux_partitioning_layers)):
                
                # Mass input fluxes for each bucket are either: (1) a fraction of the precipitation
                # at this timestep if the bucket is in the first layer, or (2) the sum of all partitioned
                # fluxes that are aimed at this bucket from the previous layer.
                if layer_idx == 0:
                    layer_storages = torch.cat([bucket.storage for bucket in bucket_layer], dim=-1)
                    # Normalize layer storages, so that growing states over the sequence don't cause problems in the networks.
                    layer_storages = torch.nn.functional.normalize(layer_storages, p=1, dim=-1)
                    inputs = torch.cat([x_at, x_s, layer_storages], dim=-1)
                    mass_inputs = self.mass_input_partitioning(x=inputs, flux=x_mt)
                else:
                    mass_inputs = torch.sum(torch.stack(partitioned_fluxes, dim=1), dim=1)
                
                # Slow but explicit loop to do stuff with each bucket in the layer.
                partitioned_fluxes = []
                for bucket_idx, bucket in enumerate(bucket_layer):

                    # Calculate responses of each bucket in the layer. This returns a dict of:
                    # outflow, storage state, and source/sink term for each bucket.
                    bucket_response = bucket(x_m=torch.unsqueeze(mass_inputs[:, bucket_idx], dim=-1), x_s=x_s, x_a=x_at)

                    # Partition the outputs from each bucket into fluxes for the next layer,
                    # or sum as the final output of the model.
                    if layer_idx < len(self.nashlike_cascade) - 1:
                        partitioner = flux_partinioning_layer[bucket_idx]
                        partitioner_inputs = torch.cat([x_at, x_s, previous_storages[layer_idx+1], bucket_response['storage']], dim=-1)
                        partitioned_fluxes.append(partitioner(x=partitioner_inputs, flux=bucket_response['outflow']))
                    else:
                        output[:, t] += bucket_response['outflow']
                        
        return {'y_hat': output}


class _SingleLeakyBucket(BaseModel):
    """Initialize a single leaky bucket.

    Parameters
    ----------
    cfg : Config
        Configuration of the run, read from the config file with some additional keys (such as number of basins).

    Raises
    ------
    ValueError
        If there are multiple time frequencies in the data (MTS).
    """

    def __init__(self, cfg: Config):
        super(_SingleLeakyBucket, self).__init__(cfg=cfg)

        # Get the size of different types of inputs from config file.
        # Currently we do not support bucket models in a multiple timestep (MTS) setting.        
        if isinstance(cfg.dynamic_inputs, dict):
            frequencies = list(cfg.dynamic_inputs.keys())
            if len(frequencies) > 1:
                raise ValueError('Bucket models only support single-frequency data')
            dynamics_input_size = len(cfg.dynamic_inputs[frequencies[0]])
        else:
            dynamics_input_size = len(cfg.dynamic_inputs)

        statics_input_size = len(cfg.static_attributes + cfg.hydroatlas_attributes + cfg.evolving_attributes)
        # If the user requests using one-hot encoding to identify basins, this will be added to static inputs.
        if cfg.use_basin_id_encoding:
            statics_input_size += cfg.number_of_basins

        # Layer to learn bucket parameters from static attributes.
        # The final layer size is the number of parameters to learn (i.e., 2: height & rate).
        hidden_sizes = cfg.leaky_bucket_parameterization_hidden_sizes
        hidden_sizes.append(2)
        self._parameterization_layer = FC(
            input_size=statics_input_size,
            hidden_sizes=hidden_sizes,
            activation=cfg.leaky_bucket_parameterization_activation,
            dropout=cfg.leaky_bucket_parameterization_dropout,
        )

        # Layer to estimate a total source/sink term from meteorological inputs (e.g., ET + baseflow + recharge).
        # This term is dynamic (time-variable), and is estimated from both static and dynamic inputs, plus the
        # current storage state of this bucket. In the future, inputs to this layer could include all system memory
        # i.e., the storate state of all buckets in the system.
        hidden_sizes = cfg.leaky_bucket_sink_source_hidden_sizes
        hidden_sizes = hidden_sizes.append(1)
        self._unobserved_sink_source_layer = FC(
            input_size=statics_input_size + dynamics_input_size + 1,
            hidden_sizes=cfg.leaky_bucket_sink_source_hidden_sizes,
            activation=cfg.leaky_bucket_sink_source_activation,
            dropout=cfg.leaky_bucket_sink_source_dropout,
        )

    def init_storage(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initializes the storage in a bucket."""
        # We can do this however we want, but for now let's just start with every bucket being empty.
        self.storage = torch.zeros([batch_size, 1], device=device)
        self._zeros = torch.zeros_like(self.storage)

    def forward(self, x_m: torch.Tensor, x_s: torch.Tensor, x_a: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Apologies for the poor naming convention of input tensors. This convention is consistent throughout
        # the NeuralHydrology codebase.
        # - x_d = dynamic inputs (inputs that change with time, mostly meteorological variables)
        # - x_s = static inputs (modely geophysical catchment characteristics)
        # - x_m = mass inputs (inputs that participate in a mass balance, if such a thing exists in a particular model, like the MC-LSTM)
        # - x_a = auxilary inputs (inputs that do not participate in a mass balance, if such exists in a particular model)

        # Add mass inputs to the bucket.
        self.storage = self.storage.clone() + x_m

        # Estimate net unobserved source and sink fluxes using all input data at this timestep.
        # This flux can be positive or negative. I would expect it to be negative in most (realistic)
        # cases, since it probably mostly represents ET + recharge, although recharge could be
        # negative due to baseflow.
        inputs = torch.cat([x_s, x_a, self.storage], dim=-1)
        unobserved_source_sink_fluxes = self._unobserved_sink_source_layer(x=inputs)
        self.storage = self.storage.clone() + torch.maximum(unobserved_source_sink_fluxes, self.storage)

        # Estimate height and rate paramters from static inputs only.
        # rate = parameters[0], height = parameters[1].
        # Height must be > 0. Rate must be in (0, 1).
        # TODO: There is a danger here that we are asking the height parameter to migrate to
        # reasonable values in the space of some non-normalized physical units. This needs
        # to be looked at closely.
        parameters = self._parameterization_layer(x=x_s)
        height = torch.unsqueeze(torch.abs(parameters[:, 1]), dim=-1)
        rate = torch.unsqueeze(torch.sigmoid(parameters[:, 0]), dim=-1)

        # Outflow from leaky bucket.
        outflow = rate * self.storage
        self.storage = self.storage.clone() - outflow

        # Assess whether bucket overflows.
        overflow = torch.maximum(self.storage - height, self._zeros)
        self.storage = self.storage.clone() - overflow

        # Create output dict in the format the NeuralHydrology expects.
        return {
            'outflow': outflow + overflow,
            'storage': self.storage,
            'source_sinks': unobserved_source_sink_fluxes,
        }


class _FluxPartition(torch.nn.Module):
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
    def __init__(self, num_inputs: int, num_outputs: int, activation: str):
        super(_FluxPartition, self).__init__()
        
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

