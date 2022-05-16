from typing import Dict

import pandas as pd
import torch
import torch.nn as nn

from neuralhydrology.evaluation.metrics import calculate_metrics, get_available_metrics
from neuralhydrology.training import get_loss_obj, get_optimizer, get_regularization_obj, loss
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.utils.assimilationconfig import AssimilationConfig


class Assimilation(object):

    def __init__(self, cfg: AssimilationConfig):
        self.cfg = cfg

        # start and end index of the assimilation period
        self._end_timestep = cfg.seq_length - cfg.assimilation_lead_time
        self._start_timestep = self._end_timestep - (cfg.history * cfg.assimilation_window)

        if self._end_timestep > cfg.seq_length:
            raise ValueError("The sum of the warmup and assimilation period has to be smaller than the sequence length")

        # initialize loss object
        # TODO: if we use the Config later, make sure to set predict_last_n to the full window size before calling
        # this function.
        self._loss_obj = get_loss_obj(cfg)
        self._loss_obj.set_regularization_terms(get_regularization_obj(cfg=cfg))

    def assimilate(self, model: BaseModel, data: Dict[str, torch.Tensor], verbose: str=False):

        # Check if we use dropout in the model during assimilation as a regularization strategy
        if self.cfg.model_dropout:
            model.train()
        else:
            # we need to track gradients through all parts of the model. However, cudNN layers only track gradients in
            # train mode. So we either set dropout to zero everywhere and run train mode, or only set the cudnn model
            # parts to train mode, so we don't have to touch dropout in e.g. the head and embedding layer.
            for p in dir(model):
                if isinstance(getattr(model, p), nn.modules.rnn.LSTM) or isinstance(getattr(model, p),
                                                                                    nn.modules.rnn.GRU):
                    getattr(model, p).train()

        # during assimilation, we can also use dropout on the timesteps, to add additional stochasticity
        if self.cfg.timestep_dropout > 0.0:
            timestep_dropout = True
            mask_sampler = torch.distributions.bernoulli.Bernoulli(torch.tensor([1 - self.cfg.timestep_dropout]))
        else:
            timestep_dropout = False

        # perform forward pass through the warmup period until the first assimilation step
        assim_data = {k: (v.clone() if k != 'x_d' else v[:, :self._start_timestep, :].clone()) for k, v in data.items()}
        pred = model(assim_data)

        # store the predictions of all time steps in a list, which we concatenate at the end
        y_hat = [pred['y_hat'].detach().clone()]

        # we need the initial learning rate for our manual learning rate scheduling
        learning_rate = self.cfg.learning_rate[0]

        # Verbose output is useful to see how DA is working for your particular problem -- this is not
        # just a tool for debugging code.
        # Calculates background loss on the actual target data before any assimilation.
        if verbose:
            test_pred = model(data)
            no_da_test_loss = self._loss_obj(test_pred, data)
            test_mask = ~torch.isnan(data["y"][:,-1,:]).any(-1)
            no_da_nse = 1 - torch.mean((test_pred['y_hat'][test_mask,-1,:] - data['y'][test_mask,-1,:])**2) / torch.mean((data['y'][test_mask,-1,:] - torch.mean(data['y'][test_mask,-1,:]))**2)

        for timestep in range(self._start_timestep, self._end_timestep, self.cfg.assimilation_window):

            # slice input sequence only for the specific assimilation window
            assim_data['x_d'] = data['x_d'][:, timestep:timestep + self.cfg.assimilation_window].clone()
            assim_data['y'] = data['y'][:, timestep:timestep + self.cfg.assimilation_window].clone()

            # required for generating dropout masks
            if timestep_dropout:
                batch_size, sequence_length, n_targets = assim_data["y"].shape

            # add states of the last time step to the input data dictionary
            for var in model.state_var_names:
                assim_data[var] = pred[var].detach().clone()

            # make sure we calculate gradients for the assimilation targets
            for var in self.cfg.assimilation_targets:
                assim_data[var].requires_grad = True

            # create a copy of the data, so we can go back one iteration step, if the update makes the results worse
            last_assim_data = {k: v.clone() for k, v in assim_data.items()}

            # create an optimizer instsance
            optimizer = get_optimizer([assim_data[var] for var in self.cfg.assimilation_targets], self.cfg)

            # set initial learning rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate

            # compute predictions with unchanged inputs
            pred = model(assim_data)

            # we use this counter variable for our manual learning rate decrease after a set number of succesful updates
            counter = 0

            # Verbose output: Calculates background loss on test data using results from last timestep. 
            # This is to check that assimilation is carrying information correctly between timesteps.
            if verbose:
                test_assim_data = {k: v.detach().clone() for k, v in data.items()}
                test_assim_data['x_d'] = test_assim_data['x_d'][:, timestep + self.cfg.assimilation_window:, :]
                test_assim_data['y'] = test_assim_data['y'][:, timestep + self.cfg.assimilation_window:, :]
                test_assim_data['c_n'] = pred['c_n'].detach().clone()
                test_assim_data['h_n'] = pred['h_n'].detach().clone()
                test_pred = model(test_assim_data)
                initial_test_loss = self._loss_obj(test_pred, test_assim_data)
                test_initial_nse = 1 - torch.mean((test_pred['y_hat'][test_mask,-1,:] - data['y'][test_mask,-1,:])**2) / torch.mean((data['y'][test_mask,-1,:] - torch.mean(data['y'][test_mask,-1,:]))**2)
                print(timestep, no_da_test_loss.item(), initial_test_loss.item(), no_da_nse.item(), test_initial_nse.item())

            for epoch in range(self.cfg.epochs):

                # perform one update step
                optimizer.zero_grad()

                mask = ~torch.isnan(assim_data["x_d"]).any(1).any(1)
                # mask all outputs required for loss computation. Currently exclude states, since those have shape
                # [1, batch, hidden] for cudalstm and would require some extra steps
                masked_pred = {k: pred[k][mask] for k in self._loss_obj._prediction_keys}
                # same as above, exclude states
                masked_data = {k: assim_data[k][mask] for k in data.keys()}

                if timestep_dropout:
                    # sample and apply dropout mask. Not sure why the sampling results always has a trailing dim of 1...
                    timestep_mask = mask_sampler.sample(masked_data["y"].shape).bool().squeeze(-1).to(data["y"].device)

                    # We "apply" dropout by setting the targets to NaN, which will exclude those timesteps in the loss
                    masked_data["y"][timestep_mask] = torch.tensor(float("nan")).to(data["y"].device)

                initial_loss_value = self._loss_obj(masked_pred, masked_data)
                initial_loss_value.backward()
                optimizer.step()

                # perform another forward pass with updated model inputs and compute the loss after the update
                pred = model(assim_data)
                mask = ~torch.isnan(assim_data["x_d"]).any(1).any(1)
                masked_pred = {k: pred[k][mask] for k in self._loss_obj._prediction_keys}
                masked_data = {k: assim_data[k][mask] for k in data.keys()}

                if timestep_dropout:
                    # Same mask, to make the results comparable
                    masked_data["y"][timestep_mask] = torch.tensor(float("nan")).to(data["y"].device)

                loss_after_update = self._loss_obj(masked_pred, masked_data)

                # Verbose output: Calculates analysis loss on test data using results from this epoch. 
                # This is to check whether assimilation is reducing loss on predict_last_n.
                if verbose:
                    test_assim_data = {k: v.detach().clone() for k, v in data.items()}
                    test_assim_data['x_d'] = test_assim_data['x_d'][:, timestep + self.cfg.assimilation_window:, :]
                    test_assim_data['y'] = test_assim_data['y'][:, timestep + self.cfg.assimilation_window:, :]#[:, -1, :]
                    test_assim_data['c_n'] = pred['c_n'].detach().clone()
                    test_assim_data['h_n'] = pred['h_n'].detach().clone()
                    test_pred = model(test_assim_data)
                    test_loss = self._loss_obj(test_pred, test_assim_data)
                    test_assim_nse = 1 - torch.mean((test_pred['y_hat'][test_mask,-1,:] - data['y'][test_mask,-1,:])**2) / torch.mean((data['y'][test_mask,-1,:] - torch.mean(data['y'][test_mask,-1,:]))**2)
                    print(timestep, epoch, no_da_test_loss.item(), initial_test_loss.item(), test_loss.item(), no_da_nse.item(), test_initial_nse.item(), test_assim_nse.item())

                # print(f"Initial loss: {initial_loss_value.item()}, loss after update: {loss_after_update.item()}")

                # if the loss is increasing rather than decreasing, reset the inputs to the previous state and reduce
                # the learning rate
                if initial_loss_value < loss_after_update:
                    assim_data = {k: v.clone() for k, v in last_assim_data.items()}
                    learning_rate = learning_rate * self.cfg.learning_rate_drop_factor
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = learning_rate
                    # print(f"Got worse, reduce learning rate to {learning_rate}")
 
                    # print(f"Reduce learning rate due to constant epoch drop factor to {learning_rate}")
                    # reset counter
                    counter = 0
                else:
                    # store last "good" data so we can go back if loss increases on subsequent epoch
                    counter += 1
                    last_assim_data = {k: v.detach().clone() for k, v in assim_data.items()}

                    # check the counter with the user defined constant learning rate step size and optionally reduce lr
                    if counter == self.cfg.learning_rate_epoch_drop:
                        learning_rate = learning_rate * self.cfg.learning_rate_drop_factor
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = learning_rate

                        # print(f"Reduce learning rate due to constant epoch drop factor to {learning_rate}")
                        # reset counter
                        counter = 0

                # heuristic values for early stopping
                if learning_rate < 1e-5 or loss_after_update < 1e-5:
                    break

                # we don't want to overfit to earlier parts of the assimilation period. those we only allow a limited
                # number of epochs unti we reached the final assimilation window.
                if (timestep < self._end_timestep - self.cfg.assimilation_window) and (epoch >= 5):
                    break

            # TODO: Unclear why this is different than assim_window
            y_hat.append(pred["y_hat"])

        # assim_data = {k: v.detach().clone() for k, v in data.items()}
        # assim_data['x_d'] = assim_data['x_d'][:, timestep + self.cfg.assimilation_window:, :]
        # assim_data['c_n'] = pred['c_n'].detach().clone()
        # assim_data['h_n'] = pred['h_n'].detach().clone()
        # test_pred = model(assim_data)

        # add states of the last time step to the input data dictionary
        assim_data = {k: v.detach().clone() for k, v in data.items()}
        for var in model.state_var_names:
            assim_data[var] = pred[var].detach().clone()
        
        # finally, make predictions beyond the assimilation period
        assim_data['x_d'] = data['x_d'][:, self._end_timestep:, :]
        pred = model(assim_data)

        # Final metric at the end of assimilation 
        if verbose:
            test_mask = ~torch.isnan(data["y"][:,-1,:]).any(-1)
            test_assim_nse = 1 - torch.mean((pred['y_hat'][test_mask,-1,:] - data['y'][test_mask,-1,:])**2) / torch.mean((data['y'][test_mask,-1,:] - torch.mean(data['y'][test_mask,-1,:]))**2)
            print(test_assim_nse)

        return {'y_hat': torch.cat(y_hat + [pred["y_hat"]], 1)}
