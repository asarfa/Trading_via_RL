import argparse
from helpers.main_helper import add_env_args, get_env_configs
from mygym.utils import env_creator
from agents.baseline_agents import FixedActionAgent, RandomAgent, LstmAgent, DnnAgent
from utils.utils import split_dates
from database.HistoricalDatabase import HistoricalDatabase

if __name__ == '__main__':

    tickers = ['AAPL', 'TSLA']
    name = 'tech_stocks'
    step_size_in_hour = 1.0
    database = HistoricalDatabase(tickers, name)
    dates = split_dates(split=0.7, start_date=database.start_date, end_date=database.end_date)

    """
    With Feed forward network agent as DQN 
    """
    parser = argparse.ArgumentParser(description="")
    add_env_args(parser, step_size_in_hour, tickers[0], dates, lags=0, manage_risk=False)
    args = vars(parser.parse_args())

    train_env_config, eval_env_config = get_env_configs(args)
    train_env = env_creator(train_env_config, database)
    eval_env = env_creator(eval_env_config, database)
    agent = DnnAgent(train_env, eval_env)
    agent.learn()

    """
    With LSTM network agent as DQN
    """
    lags = 7*5 #7hours of trading per day, 5 business day in a week --> results in lags over the week
    parser = argparse.ArgumentParser(description="")
    add_env_args(parser, step_size_in_hour, tickers[0], dates, lags, manage_risk=False)
    args = vars(parser.parse_args())

    train_env_config, eval_env_config = get_env_configs(args)
    train_env = env_creator(train_env_config, database)
    eval_env = env_creator(eval_env_config, database)
    agent = LstmAgent(train_env, eval_env)
    agent.learn()


    """
    #work with random agent
    train_env_config, eval_env_config = get_env_configs(args)
    train_env = env_creator(train_env_config, database)
    eval_env = env_creator(eval_env_config, database)
    agent = RandomAgent(train_env, eval_env)
    agent.learn()

    #work with fixed agent (fixed action --> long, neutral or short)
    for i in range(3):
        train_env_config, eval_env_config = get_env_configs(args)
        train_env = env_creator(train_env_config, database)
        eval_env = env_creator(eval_env_config, database)
        agent = FixedActionAgent(i, train_env, eval_env)
        agent.learn()
    """

    print('End')