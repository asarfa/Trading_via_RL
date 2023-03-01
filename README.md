# Trading via Reinforcement Learning

The aim of this project is to create a Reinforcement Learning strategy through a trading bot which take action 
(long, neutral or short) by interacting with a financial market environment and then backtesting this strategy.
Rewards are obviously the monetary gains and losses with transaction fees.
It is set up to use intraday data (1h tick) provided by [YAHOO_Finance](https://finance.yahoo.com/).

Environment : Financial market

State : All relevant parameters that describe the current state of the environment
this include historical price levels and financial indicators

Agent : Represents a trader placing bets on rising or falling markets

Action: Going long, neutral or short

Step: Given an action of an agent, the state of the environment is updated

Reward: Depending on the action an agent chooses, a PnL value is computed

Target : Specifies what the agent tries to maximize, there is the accumulated trading profit (PnL)

Policy : Defines which action an agent takes given a certain state of the environment, using DQN

Episode : Set of steps from the initial state of the environment until success is achieved or failure is observed