from datetime import timedelta

from database.HistoricalDatabase import HistoricalDatabase
from mygym.FinancialEnvironment import FinancialEnvironment
from rewards.RewardFunctions import PnL

from features.Features import Portfolio

# import sns
from pylab import plt
import pandas as pd
import numpy as np
import os


def get_reward_function(reward_function: str):
    if reward_function == "PnL":
        return PnL()


def env_creator(env_config, database: HistoricalDatabase = None):
    if env_config["features"] == "agent_state":
        features = FinancialEnvironment.get_default_features(
            step_size=timedelta(hours=env_config["step_size"]),
            normalisation_on=env_config["normalisation_on"],
        )[-1:]

    elif env_config["features"] == "market_state":
        features = FinancialEnvironment.get_default_features(
            step_size=timedelta(hours=env_config["step_size"]),
            normalisation_on=env_config["normalisation_on"],
        )[:-1]

    elif env_config["features"] == "full_state":
        features = FinancialEnvironment.get_default_features(
            step_size=timedelta(hours=env_config["step_size"]),
            normalisation_on=env_config["normalisation_on"],
        )

    env = FinancialEnvironment(
        database=database,
        features=features,
        ticker=env_config["ticker"],
        step_size=timedelta(hours=env_config["step_size"]),
        start_of_trading=env_config["start_trading"],
        end_of_trading=env_config["end_trading"],
        per_step_reward_function=get_reward_function(env_config["per_step_reward_function"]),
        n_lags_feature=env_config["n_lags_feature"],
        manage_risk=env_config["manage_risk"]
    )
    return env

def done_inf(dct):
    df = pd.DataFrame.from_dict(dct, orient='index').T
    df.index = list(df.index+1)
    return np.round(df, 1)

def plot_per_episode(
        ticker,
        agent_name,
        step_size,
        step_info_per_episode,
        step_info_per_eval_episode,
        episode,
        done_info,
        done_info_eval,
        manage_risk,
        eval_data
):
    step_info_per_episode = step_info_per_episode[episode] if step_info_per_episode is not None else None
    step_info_per_eval_episode = step_info_per_eval_episode[episode]

    def join_df(step_info_per_episode, step_info_per_eval_episode, train_metric, val_metric):
        if step_info_per_episode is not None:
            train_metric = pd.DataFrame([train_metric], index=['training'],
                                        columns=step_info_per_episode.__dict__['dates'][-len(train_metric):]).T
            val_metric = pd.DataFrame([val_metric], index=['testing'],
                                      columns=step_info_per_eval_episode.__dict__['dates'][-len(val_metric):]).T
            metrics = pd.concat([train_metric, val_metric])
            assert(len(metrics.T)==2)
            assert(len(metrics)==(len(train_metric)+len(val_metric)))
        else:
            train_metric = pd.DataFrame()
            val_metric = pd.DataFrame([val_metric], index=['testing'],
                                      columns=step_info_per_eval_episode.__dict__['dates'][-len(val_metric):]).T
            metrics = val_metric
        return train_metric, val_metric, metrics

    def graph_per_episode(step_info_per_episode, step_info_per_eval_episode, metric: str = None):
        train_metric = step_info_per_episode.__dict__[metric] if step_info_per_episode is not None else None
        val_metric = step_info_per_eval_episode.__dict__[metric]
        _, _, metrics = join_df(step_info_per_episode, step_info_per_eval_episode, train_metric, val_metric)
        return metrics

    def info_reward(step_info_per_episode, step_info_per_eval_episode):
        train_reward = np.diff(step_info_per_episode.__dict__['pnls']) if step_info_per_episode is not None else None
        val_reward = np.diff(step_info_per_eval_episode.__dict__['pnls'])
        train_reward, test_reward, rewards = join_df(step_info_per_episode, step_info_per_eval_episode, train_reward, val_reward)
        stats = pd.DataFrame(rewards).describe()
        stats = np.round(stats)
        stats = stats.astype(int)
        return stats, rewards

    def info_returns(step_info_per_episode, step_info_per_eval_episode, aum_key: str, window: str):
        train_returns = pd.Series(step_info_per_episode.__dict__[aum_key][:-1]).pct_change().dropna().values if step_info_per_episode is not None else None
        val_returns = pd.Series(step_info_per_eval_episode.__dict__[aum_key][:-1]).pct_change().dropna().values
        train_returns, test_returns, returns = join_df(step_info_per_episode, step_info_per_eval_episode, train_returns, val_returns)
        returns_roll_mean = pd.concat([train_returns.rolling(window).mean()[100:],
                                      test_returns.rolling(window).mean()[100:]]).dropna(how='all')
        returns_roll_vol = pd.concat([train_returns.rolling(window).std()[100:],
                                     test_returns.rolling(window).std()[100:]]).dropna(how='all')
        returns_sharpe = returns_roll_mean/returns_roll_vol*(252**0.5)
        stats = pd.DataFrame(returns).describe()
        stats = np.round(stats)
        stats = stats.astype(int)
        return stats, returns_sharpe

    def info_actions(step_info_per_episode, step_info_per_eval_episode):
        actions_train = pd.DataFrame(step_info_per_episode.__dict__['positions'], columns=['training']) if step_info_per_episode is not None else pd.DataFrame()
        actions_test = pd.DataFrame(step_info_per_eval_episode.__dict__['positions'], columns=['test'])
        return actions_train.value_counts(), actions_test.value_counts()

    fig = plt.figure(constrained_layout=True, figsize=(10, 15))

    if step_info_per_episode is not None:
        ax_dict = fig.subplot_mosaic(
            """
            ZY
            AA
            CD
            EE
            GH
            """
        )
    else:
        ax_dict = fig.subplot_mosaic(
            """
            ZY
            AA
            CD
            EE
            FF
            """
        )



    eval_str = 'Testing: ' if done_info is None else ''
    name = f"{eval_str}{ticker} - {agent_name} - Episode_{episode} \n step size: {step_size.seconds//3600}h | " \
        + f"reward: PnL with trading fees | managing risk: {manage_risk}\n"

    plt.suptitle(name)

    done_info = done_inf(done_info).iloc[episode-1].to_frame().T if done_info is not None else pd.DataFrame(np.zeros(4)).T
    done_info_eval = done_inf(done_info_eval).iloc[episode-1].to_frame().T

    if step_info_per_episode is not None:
        table = ax_dict["Z"].table(
            cellText=done_info.values,
            colLabels=done_info_eval.columns,
            loc="center",
        )
        table.set_fontsize(6.5)
        #table.scale(0.5, 1.1)
        ax_dict["Z"].set_axis_off()
        ax_dict["Z"].title.set_text("Agent's characteristics training")

    table = ax_dict["Y"].table(
        cellText=done_info_eval.values,
        colLabels=done_info_eval.columns,
        loc="center",
    )
    table.set_fontsize(6.5)
    #table.scale(0.5, 1.1)
    ax_dict["Y"].set_axis_off()
    ax_dict["Y"].title.set_text("Agent's characteristics testing")

    aum_curve = graph_per_episode(step_info_per_episode, step_info_per_eval_episode, 'aums')
    aum_market_curve = graph_per_episode(step_info_per_episode, step_info_per_eval_episode, 'market_aums')
    curves_aum = pd.concat([aum_curve, aum_market_curve], axis=1)
    curves_aum.columns = ['agent_training', 'agent_testing', 'mkt_training', 'mkt_testing'] if step_info_per_episode is not None else ['agent_testing', 'mkt_testing']
    curves_aum.plot(ax=ax_dict["A"], ylabel='$',
                      title=f'Agent vs Market Net Wealth through time')

    stats_rewards, rewards = info_reward(step_info_per_episode, step_info_per_eval_episode)

    table = ax_dict["C"].table(
        cellText=stats_rewards.values,
        rowLabels=stats_rewards.index,
        colLabels=stats_rewards.columns,
        loc="center",
    )
    table.set_fontsize(6.5)
    ax_dict["C"].set_axis_off()
    ax_dict["C"].title.set_text("Satistics of agent's reward (PnL)")

    non_null_reward = rewards[rewards != 0].dropna(how='all')
    non_null_reward.plot.hist(ax=ax_dict["D"], title="Non null rewards histogram", bins=50, alpha=0.5)

    window = '30d'
    stats_returns, sharpe_returns = info_returns(step_info_per_episode, step_info_per_eval_episode, 'aums', window)
    stats_mkt_returns, sharpe_mkt_returns = info_returns(step_info_per_episode, step_info_per_eval_episode, 'market_aums', window)
    sharpe = pd.concat([sharpe_returns, sharpe_mkt_returns], axis=1)
    sharpe.columns = ['agent_training', 'agent_testing', 'mkt_training', 'mkt_testing'] if step_info_per_episode is not None else ['agent_testing', 'mkt_testing']
    sharpe.plot(ax=ax_dict["E"], title=f'Agent vs Market Annual Sharpe ratio rolling {window}')

    actions_train, actions_test = info_actions(step_info_per_episode, step_info_per_eval_episode)
    actions = pd.concat([actions_train, actions_test], axis=1) if step_info_per_episode is not None else actions_test.to_frame()
    actions.columns = ['training', 'testing'] if step_info_per_episode is not None else ['testing']
    actions.sort_values(by='testing', inplace=True)
    if step_info_per_episode is not None:
        actions.plot.barh(ax=ax_dict["G"], title="Count agent's position")
    else:
        actions.plot.barh(ax=ax_dict["Z"], title="Count agent's position")

    prices = eval_data.loc[step_info_per_eval_episode.dates].Close
    plt.plot(prices)
    short_ticks = []
    long_ticks = []
    neutral_ticks = []
    last_position = None
    for i, tick in enumerate(step_info_per_eval_episode.dates):
        if step_info_per_eval_episode.positions[i] == "short" and last_position != 'short':
            short_ticks.append(tick)
            last_position = 'short'
        elif step_info_per_eval_episode.positions[i] == "long" and last_position != 'long':
            long_ticks.append(tick)
            last_position = 'long'
        elif step_info_per_eval_episode.positions[i] == "neutral" and last_position != 'neutral':
            neutral_ticks.append(tick)
            last_position = 'neutral'
    plt.plot(short_ticks, prices[short_ticks], 'ro')
    plt.plot(long_ticks, prices[long_ticks], 'go')
    plt.plot(neutral_ticks, prices[neutral_ticks], 'wo')
    if step_info_per_episode is not None:
        letter = "H"
    else:
        letter = "F"
    plt.plot(ax=ax_dict[letter])
    ax_dict[letter].title.set_text('Agent entry position through time on testing set')

    pdf_path = os.path.join("results", agent_name, ticker)
    os.makedirs(pdf_path, exist_ok=True)
    subname = name.replace(ticker, '').replace('\n', '').replace(' ', '_').replace(':', '').replace('|','')
    pdf_filename = os.path.join(pdf_path, f"Ep_{episode}_{subname}.jpg")
    # Write plot to pdf
    fig.savefig(pdf_filename)
    plt.close(fig)


def plot_final(
        done_info,
        done_info_eval,
        ticker,
        agent_name,
        step_size,
        manage_risk,
):
    def graph_final(done_info, done_info_eval, metric):
        metrics = pd.DataFrame([done_info[metric], done_info_eval[metric]], index=['training', 'testing']).T
        return metrics

    fig = plt.figure(constrained_layout=True, figsize=(15, 15))
    ax_dict = fig.subplot_mosaic(
        """
        AB
        CD
        """
    )
    name = f"{ticker} - {agent_name}\n step size: {step_size.seconds//3600}h | reward: PnL with trading fees | managing risk: {manage_risk}\n"
    plt.suptitle(name)
    pdf_path = os.path.join("results", agent_name, ticker)
    subname = name.replace(ticker, '').replace('\n', '').replace(' ', '_').replace(':', '').replace('|','')
    pdf_filename = os.path.join(pdf_path, f"final_{subname}.pdf")

    metrics = ['pnl', 'sharpe', 'trades', 'depth']
    for metric, ax in zip(metrics, ["A", "B", "C", "D"]):
        graph = graph_final(done_info, done_info_eval, metric)
        graph.plot(ax=ax_dict[ax], ylabel=metric, xlabel='n-th episodes', title=f'{metric} through episodes')

    fig.savefig(pdf_filename)
    plt.close(fig)