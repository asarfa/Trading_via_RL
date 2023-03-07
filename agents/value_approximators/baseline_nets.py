import torch
import torch.nn as nn
from torch.optim import Adam, SGD
import os
import numpy as np
import time


class Net:
    """
    This class allows to compute the main methods to fit a model and predict output on testing set
    """

    def __init__(self, model, lr: float = 0.001, opt: str = 'Adam',
                 name: str = None, verbose: bool = False, seed: int = None):
        self.verbose = verbose
        self.seed = seed
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = nn.MSELoss()
        self.lr = lr
        self.opt = opt
        self.optimizer = None
        self.train_loss = None
        self.val_loss = None
        self.path = None
        self.name = name
        self.__set_seed()
        self.__instantiate_model(model)
        self.__instantiate_optimizer()

    def __set_seed(self):
        torch.manual_seed(self.seed)

    def __instantiate_model(self, model):
        self.model = Utils.to_device(model, self.device)

    def set(self, path, episode: int = 0):
        last_episode = Utils.find_last_episode(path.replace('EpisodeNone', '')) if episode==0 else episode
        self.path = path.replace('None', str(last_episode)) + '_model'
        self.model, self.optimizer = Utils.load(self.model, self.optimizer, self.path, self.device)
        return last_episode

    def __instantiate_optimizer(self):
        if self.opt == 'Adam':
            self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        elif self.opt == 'SGD':
            self.optimizer = SGD(self.model.parameters(), lr=self.lr)
        else:
            raise ValueError(f'{self.optimizer} has not been implemented')

    @staticmethod
    def __transf(state: np.ndarray, target: np.ndarray = None):
        state = torch.tensor(state, dtype=torch.float32)
        if len(state.shape) in [1, 2]:
            state = state.unsqueeze(0)
        if target is not None:
            target = torch.tensor(target, dtype=torch.float32)
            return state, target
        return state

    def __train_model(self, state: np.ndarray, target: torch.tensor):
        # set the model in training mode
        self.model.train()
        # send input to device
        state, target = self.__transf(state, target)
        state, target = Utils.to_device((state, target), self.device)
        # zero out previous accumulated gradients
        self.optimizer.zero_grad() #not needed as no batch
        # perform forward pass and calculate accuracy + loss
        all_Q_values = self.model(state)
        Q_values = all_Q_values
        loss = self.criterion(Q_values, target)
        # perform backpropagation and update model parameters
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def __evaluate_model(self, state: np.ndarray, idmax: bool=None):
        # set the model in eval mode
        self.model.eval()
        # send input to device
        state = self.__transf(state)
        state = Utils.to_device(state, self.device)
        output = self.model(state)
        if idmax: output = Utils.argmax(output)
        return output

    def __compute_verbose_train(self, epoch, start_time, train_loss):
        print("Epoch [{}] took {:.2f}s | train_loss: {:.4f}".format(epoch, time.time() - start_time, train_loss))

    def fit(self, state: np.ndarray, target: torch.tensor):

        start_time = time.time()
        train_loss = self.__train_model(state, target)

        if self.verbose:
            self.__compute_verbose_train(1, start_time, train_loss)

        self.train_loss = train_loss

    def save_args(self, path: str = None):
        if path is not None: self.path = path + '_model'
        Utils.save(self.model, self.optimizer, self.path)

    def predict(self, state: np.ndarray, idmax: bool=None):
        prediction = self.__evaluate_model(state, idmax)
        return prediction


class Utils:
    """
    This class allows to contains all the utility function needed for neural nets
    """

    @staticmethod
    def to_device(data, device):
        if isinstance(data, (list, tuple)):
            return [Utils.to_device(x, device) for x in data]
        return data.to(device, non_blocking=True)

    @staticmethod
    def argmax(outputs):
        return torch.argmax(outputs).cpu().detach().item()

    @staticmethod
    def save(model, optimizer, path):
        torch.save({"model_state_dict": model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                    path)

    @staticmethod
    def find_last_episode(path):
        try:
            last_ep = max([int(f.split('_')[0].replace('Episode', '')) for f in os.listdir(path)])
        except ValueError:
            last_ep = 0
        return last_ep

    @staticmethod
    def load(model, optimizer, path, device):
        if os.path.exists(path):
            if device.type == 'cpu':
                checkpoint = torch.load(path, map_location=torch.device('cpu'))
            else:
                checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return model, optimizer
        else:
            return model, optimizer