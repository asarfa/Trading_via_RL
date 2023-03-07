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
    Evaluate on new tickers
    """
    for ticker in tickers[1:]:
        #dates[2] = dates[0]
        parser = argparse.ArgumentParser(description="")
        add_env_args(parser, step_size_in_hour, ticker, dates, lags=0, manage_risk=False)
        args = vars(parser.parse_args())

        _, eval_env_config = get_env_configs(args)
        eval_env = env_creator(eval_env_config, database)
        agent.evaluate(eval_env)


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