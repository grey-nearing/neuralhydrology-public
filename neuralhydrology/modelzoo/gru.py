from typing import Dict

import torch
from torch import nn

from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.modelzoo.inputlayer import InputLayer
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.utils.config import Config


class GRU(BaseModel):
    """Gated Recurrent Unit (GRU) class based on the PyTorch GRU implementation.

    This class implements the standard GRU combined with a model head, as specified in the config. All features
    (time series and static) are concatenated and passed to the GRU directly.
    The `GRU` class only supports single-timescale predictions.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """
    # specify submodules of the model that can later be used for finetuning. Names must match class attributes
    module_parts = ['embedding_net', 'gru', 'head']

    # names of state variables in the returned dictionary of the forward function
    state_var_names = ['h_n']

    def __init__(self, cfg: Config):

        super(GRU, self).__init__(cfg=cfg)

        self.embedding_net = InputLayer(cfg)

        self.gru = nn.GRU(input_size=self.embedding_net.output_size, hidden_size=cfg.hidden_size)

        self.dropout = nn.Dropout(p=cfg.output_dropout)

        self.head = get_head(cfg=cfg, n_in=cfg.hidden_size, n_out=self.output_size)

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform a forward pass on the GRU model.

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            Dictionary, containing input features as key-value pairs. If the dictionary includes 'h_n' this tensor will 
            be used as initial hidden state. Otherwise, the initial hidden state defaults to a vector or zeros. The 
            shape of the initial hidden state has to be [1, batch size, hidden size].

        Returns
        -------
        Dict[str, torch.Tensor]
            Model outputs and states as a dictionary.
                - `y_hat`: model predictions of shape [batch size, sequence length, number of target variables].
                - `h_n`: hidden state at the last time step of the sequence of shape [batch size, 1, hidden size].
        """
        # possibly pass dynamic and static inputs through embedding layers, then concatenate them
        x_d = self.embedding_net(data, concatenate_output=True)

        # check if data contains initial hidden state, otherwise create a vector or zeros
        batch_size = x_d.shape[1]
        if 'h_n' in data.keys():
            h_0 = data['h_n']
        else:
            h_0 = x_d.new_zeros((1, batch_size, self.gru.hidden_size))

        # run the actual GRU
        gru_output, h_n = self.gru(x_d, h_0)

        # reshape to [batch_size, 1, n_hiddens]
        h_n = h_n.transpose(0, 1)

        pred = {'h_n': h_n}

        # add the final output as it's returned by the head to the prediction dict
        # (this will contain the 'y_hat')
        pred.update(self.head(self.dropout(gru_output.transpose(0, 1))))

        return pred
