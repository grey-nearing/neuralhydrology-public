from typing import Dict

import torch
import torch.nn as nn

from neuralhydrology.modelzoo.inputlayer import InputLayer
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.utils.config import Config


class CudaLSTM(BaseModel):
    """LSTM model class, which relies on PyTorch's CUDA LSTM class.

    This class implements the standard LSTM combined with a model head, as specified in the config. Depending on the
    embedding settings, static and/or dynamic features may or may not be fed through embedding networks before being
    concatenated and passed through the LSTM.
    To control the initial forget gate bias, use the config argument `initial_forget_bias`. Often it is useful to set
    this value to a positive value at the start of the model training, to keep the forget gate closed and to facilitate
    the gradient flow.
    The `CudaLSTM` class only supports single-timescale predictions. Use `MTSLSTM` to train a model and get
    predictions on multiple temporal resolutions at the same time.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """
    # specify submodules of the model that can later be used for finetuning. Names must match class attributes
    module_parts = ['embedding_net', 'lstm', 'head']

    # names of state variables in the returned dictionary of the forward function
    state_var_names = ['h_n', 'c_n']

    def __init__(self, cfg: Config):
        super(CudaLSTM, self).__init__(cfg=cfg)

        self.embedding_net = InputLayer(cfg)

        self.lstm = nn.LSTM(input_size=self.embedding_net.output_size, hidden_size=cfg.hidden_size)

        self.dropout = nn.Dropout(p=cfg.output_dropout)

        self.head = get_head(cfg=cfg, n_in=cfg.hidden_size, n_out=self.output_size)

        self._reset_parameters()

    def _reset_parameters(self):
        """Special initialization of certain model weights."""
        if self.cfg.initial_forget_bias is not None:
            self.lstm.bias_hh_l0.data[self.cfg.hidden_size:2 * self.cfg.hidden_size] = self.cfg.initial_forget_bias

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform a forward pass on the CudaLSTM model.

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            Dictionary, containing input features as key-value pairs. If the dictionary includes 'h_n' and/or 'c_n', 
            this tensor will be used as initial hidden state and cell state, respectively. Otherwise, the initial hidden
            and/or cell state defaults to a vector or zeros. The shape of the initial hidden an cell state has to be 
            [1, batch size, hidden size].

        Returns
        -------
        Dict[str, torch.Tensor]
            Model outputs and intermediate states as a dictionary.
                - `y_hat`: model predictions of shape [batch size, sequence length, number of target variables].
                - `h_n`: hidden state at the last time step of the sequence of shape [1, batch size, hidden size].
                - `c_n`: cell state at the last time step of the sequence of shape [1, batch size, hidden size].
        """
        # possibly pass dynamic and static inputs through embedding layers, then concatenate them
        x_d = self.embedding_net(data)

        # check if data contains initial hidden state, otherwise create a vector or zeros
        batch_size = x_d.shape[1]
        if 'h_n' in data.keys():
            h_0 = data['h_n']
        else:
            h_0 = x_d.new_zeros((1, batch_size, self.lstm.hidden_size))

        # check if data contains initial cell state, otherwise create a vector or zeros
        if 'c_n' in data.keys():
            c_0 = data['c_n']
        else:
            c_0 = x_d.new_zeros((1, batch_size, self.lstm.hidden_size))

        lstm_output, (h_n, c_n) = self.lstm(x_d, (h_0, c_0))

        # reshape to [batch_size, seq, n_hiddens]
        lstm_output = lstm_output.transpose(0, 1)

        pred = {'lstm_output': lstm_output, 'h_n': h_n, 'c_n': c_n}
        pred.update(self.head(self.dropout(lstm_output)))

        return pred
