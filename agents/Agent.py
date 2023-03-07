import abc
import numpy as np
from mygym.FinancialEnvironment import FinancialEnvironment
from mygym.utils import plot_per_episode, plot_final, done_inf
from collections import deque
from pylab import plt, mpl
from copy import deepcopy
import time
import os
import pickle

plt.style.use('seaborn')
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'


class ActionSpace:
    """
    Agent's exploration policy
    Simple uniform random policy between each action possible
    """

    def __init__(self, n):
        self.n = n

    def sample(self) -> int:
        return np.random.randint(0, self.n)


class Agent(metaclass=abc.ABCMeta):
    def __init__(
            self,
            learn_env: FinancialEnvironment = None,
            test_env: FinancialEnvironment = None,
            learning_agent: bool = None,
            episodes: int = None,
            epsilon: float = None,
            epsilon_min: float = None,
            epsilon_decay: float = None,
            gamma: float = None,
            batch_size: int = None,
            seed: int = 42
    ):
        self.learn_env = learn_env
        self.test_env = test_env
        self.learning_agent = learning_agent
        self.episodes = episodes
        self.seed = seed
        self._set_learning_args(epsilon, epsilon_min, epsilon_decay, gamma, batch_size)
        self.num_actions = 3
        self.step_info_per_episode = dict(map(lambda i: (i, None), range(1, self.episodes + 1)))
        self.step_info_per_eval_episode = deepcopy(self.step_info_per_episode)
        self.done_info = {'pnl': [], 'sharpe': [], 'aum': [], 'trades': [], 'depth': []}
        self.done_info_eval = deepcopy(self.done_info)
        self.len_learn = '?'
        self.len_eval = '?'

    def _set_seed_np(self):
        np.random.seed(self.seed)

    def _set_learning_args(self, epsilon: float, epsilon_min: float, epsilon_decay: float, gamma: float,
                           batch_size: int):
        self._set_seed_np()
        if self.learning_agent:
            self.epsilon = epsilon  # Initial exploration rate
            self.epsilon_min = epsilon_min  # Minimum exploration rate
            self.epsilon_decay = epsilon_decay  # Decay rate for exploration rate, interval must be a pos function of the nb of xp
            self.gamma = gamma  # Discount factor for delayed reward
            self.batch_size = batch_size  # Batch size for replay
            self.memory = deque(maxlen=int(10e4))  # deque collection for limited history to train agent
        else:
            self.episodes = 1
            self.epsilon = 0 #useless just for verbose
            self.learn_env.aum_threshold = self.test_env.aum_threshold = -np.inf

    @property
    def actions(self):
        return list(range(self.num_actions))

    @property
    def action_space(self):
        return ActionSpace(len(self.actions))

    @abc.abstractmethod
    def get_action(self, state: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def get_name(self):
        pass

    def _greedy_policy(self, state: np.ndarray):
        if np.random.random() <= self.epsilon:
            return self.action_space.sample()
        return self.get_action(state)

    def _play_one_step(self, state: np.ndarray):
        state = state.copy()
        action = self._greedy_policy(state) if self.learning_agent else self.get_action(state)
        next_state, reward, done, info = self.learn_env.step(action)
        if self.learning_agent:
            self.memory.append(
                [state, action, reward, next_state, done])
        return next_state, done

    @abc.abstractmethod
    def replay(self):
        pass

    def learn(self):
        start, start_ = time.time(), time.time()
        last_ep = self._set_args()
        for episode in range(last_ep, self.episodes + 1):
            state = self.learn_env.reset()
            while self.learn_env.end_of_trading >= self.learn_env.state.now_is:
                state, done = self._play_one_step(state)
                if done:
                    self.step_info_per_episode[episode] = self.learn_env.info_calculator
                    self._compute_done(self.step_info_per_episode, episode, self.done_info)
                    break
            self._evaluate(episode)
            if self.learning_agent and episode >= last_ep:
                self._save_args(episode)
            if self.learning_agent and len(self.memory) > self.batch_size:
                self.replay()
            if (episode - 1) % 10 == 0:
                plot_per_episode(self.learn_env.ticker, self.get_name(),
                                 self.learn_env.step_size, self.step_info_per_episode,
                                 self.step_info_per_eval_episode, episode, self.done_info, self.done_info_eval,
                                 self.learn_env.manage_risk, self.test_env.database.data[self.test_env.ticker])
                print(f'Time elapsed for 10 episodes: {round((time.time() - start_) / 60, 3)} minutes')
                start_ = time.time()
            if self.learning_agent and episode >= 50 and (episode - 1) % 10 == 0:
                plot_final(self.done_info, self.done_info_eval, self.learn_env.ticker, self.get_name(),
                           self.learn_env.step_size, self.learn_env.manage_risk)
        best_info_eval = done_inf(self.done_info_eval).sort_values('pnl', ascending=False).iloc[:10]
        print(f'Top 10 episodes on testing according to maximal cumulative reward (PnL):\n  {best_info_eval.to_string()}')
        self._set_best_ep()
        self.evaluate(self.test_env)
        print(f'Total time elapsed: {round((time.time() - start) / 3600, 3)} hours')

    def _evaluate(self, episode: int):
        """
        Method to validate the performance of the DQL agent.
        only relies on the exploitation of the currently optimal policy
        """
        state = self.test_env.reset()
        while self.test_env.end_of_trading >= self.test_env.state.now_is:
            action = self.get_action(state)
            state, reward, done, info = self.test_env.step(action)
            if done:
                self.step_info_per_eval_episode[episode] = self.test_env.info_calculator
                self._compute_done(self.step_info_per_eval_episode, episode, self.done_info_eval)
                break

    def _compute_done(self, info, episode: int, done_info: dict):
        info = info[episode]
        done_info['pnl'].append(info.pnls[-1])
        done_info['sharpe'].append(info.sharpe)
        done_info['aum'].append(info.aums[-1])
        done_info['trades'].append(info.n_trades)
        bar = len(info.pnls)
        done_info['depth'].append(bar)
        if (episode - 1) % 10 == 0:
            templ = '\nepisode: {:2d}/{} | bar: {:2d}/{} | epsilon: {:5.2f}\n'
            templ += 'pnl: {:5.2f} | sharpe: {:5.2f} | n_trades: {:2d}\n'
            templ += 'net wealth: {:5.2f} | success: {} \n'
            if done_info is self.done_info:
                print(50 * '*')
                print(f'           Training of {self.get_name()}      ')
                print(f'    Start of trading: {self.learn_env.start_of_trading} ')
                success = (self.learn_env.end_of_trading <= self.learn_env.state.now_is)
                if success: self.len_learn = bar
                print(templ.format(episode, self.episodes, bar, self.len_learn, self.epsilon,
                                   info.pnls[-1], info.sharpe, info.n_trades, info.aums[-1], success))
            else:
                print(f'          Evaluation of {self.get_name()}      ')
                print(f'    Start of trading: {self.test_env.start_of_trading} ')
                success = (self.test_env.end_of_trading <= self.test_env.state.now_is)
                if success: self.len_eval = bar
                print(templ.format(episode, self.episodes, bar, self.len_eval, self.epsilon,
                                   info.pnls[-1], info.sharpe, info.n_trades, info.aums[-1], success))
                print(50 * '*')

    def _get_path(self, episode: str = None):
        ticker = self.learn_env.ticker
        base = os.path.dirname(os.path.abspath(__file__))
        agent = os.path.join(base, 'savings', self.get_name(), ticker)
        if not os.path.exists(agent):
            os.makedirs(agent)
        return os.path.join(agent, 'Episode'+str(episode))

    def _save_args(self, episode: int = None):
        path = self._get_path(episode)
        self._delete_earlier_args(path, episode)
        model_agent = self.model
        self.model = None
        with open(f'{path}_agent.pkl', "wb") as file:
            pickle.dump(self, file)
        model_agent.save_args(path)
        self.model = model_agent

    def _delete_earlier_args(self, path, episode):
        #keep only previous models
        args_file = [d for d in os.listdir(path.replace('Episode' + str(episode), '')) if d.find('model')==-1]
        for file in args_file:
            delete_filename = os.path.join(path.replace('Episode' + str(episode), file))
            open(delete_filename, 'w').close()
            os.remove(delete_filename)

    def _set_args(self):
        if self.learning_agent:
            path = self._get_path()
            last_ep = self.model.set(path)
            if last_ep >= 1:
                mymodel = self.model
                print(f'Loading arguments from episode {last_ep} savings')
                with open(path.replace('None', str(last_ep)) + '_agent.pkl', "rb") as file:
                    self.__dict__ = pickle.load(file).__dict__
                self.model = mymodel
        else:
            last_ep = 0
        return last_ep + 1 #to not redo the same ep

    def _set_best_ep(self):
        """
        After learn() has been called over the 1000 episodes (self.episode), set the current model to the best one
        """
        if self.learning_agent:
            path = self._get_path()
            best_ep = done_inf(self.done_info_eval).sort_values('pnl', ascending=False).index[0]
            print(f'Set the current model to that of the best episode i.e. the one with the maximum PnL on the test set --> ({best_ep}) ')
            _ = self.model.set(path, best_ep)

    def evaluate(self, test_env: FinancialEnvironment):
        self.test_env = test_env
        self.test_env.aum_threshold = -np.inf
        print('------------------------------------------------------------------------')
        print(f'----------------Evaluation of {self.test_env.ticker}------------------')
        print('------------------------------------------------------------------------')
        self.done_info_eval = {'pnl': [], 'sharpe': [], 'aum': [], 'trades': [], 'depth': []}
        self.step_info_per_eval_episode = dict(map(lambda i: (i, None), range(1, self.episodes + 1)))
        episode = 1
        self._evaluate(episode)
        plot_per_episode(self.test_env.ticker, self.get_name(),
                         self.test_env.step_size, None,
                         self.step_info_per_eval_episode, episode, None, self.done_info_eval,
                         self.learn_env.manage_risk, self.test_env.database.data[self.test_env.ticker])


