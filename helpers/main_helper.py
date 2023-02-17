from copy import deepcopy
from datetime import datetime


def add_env_args(parser, step_in_hour: float, ticker: str, dates: list, lags: int, manage_risk: bool):
    parser.add_argument("-sz", "--step_size", default=step_in_hour, help="Step size in hours.", type=float)
    parser.add_argument("-t", "--ticker", default=ticker, help="Specify stock ticker.", type=str)
    parser.add_argument("-n", "--normalisation_on", default=True, help="Normalise features.", type=bool)

    parser.add_argument(
        "-f",
        "--features",
        default="full_state",
        choices=["agent_state", "market_state", "full_state"],
        help="Agent state, market state or full state.",
        type=str,
    )
    parser.add_argument("-nlf", "--n_lags_feature", default=lags, help="Number of lags per feature", type=int)

    parser.add_argument("-starttrain", "--start_trading_train", default=dates[0], help="Start trading train.", type=datetime)
    parser.add_argument("-endtrain", "--end_trading_train", default=dates[1], help="End trading train.", type=datetime)
    parser.add_argument("-endeval", "--start_trading_eval", default=dates[2], help="Start trading eval.", type=datetime)
    parser.add_argument("-starteval", "--end_trading_eval", default=dates[3], help="End trading eval.", type=datetime)


    parser.add_argument(
        "-psr",
        "--per_step_reward_function",
        default="PnL",
        choices=["PnL"],
        help="Per step reward function: pnl",
        type=str,
    )

    parser.add_argument("-risk", "--manage_risk", default=manage_risk, help="Managing risks.", type=bool)


def get_env_configs(args):
    env_config = {
        "ticker": args["ticker"],
        "start_trading": args["start_trading_train"],
        "end_trading": args["end_trading_train"],
        "step_size": args["step_size"],
        "features": args["features"],
        "normalisation_on": args["normalisation_on"],
        "per_step_reward_function": args["per_step_reward_function"],
        "n_lags_feature": args["n_lags_feature"],
        "manage_risk": args["manage_risk"]
    }

    eval_env_config = deepcopy(env_config)
    eval_env_config["start_trading"] = args["start_trading_eval"]
    eval_env_config["end_trading"] = args["end_trading_eval"]

    return env_config, eval_env_config


