# Reinforcement-Learning-in-Bitcoin-Option-Pricing <br>

***Environment:***

Jupyter notebook <br>
Python 3 <br>
Numpy <br>
Scipy <br>
Pandas <br>
bspline <br>
pymongo <br>

## Abstract <br>

In 2019 Hack Arizona, our project's topic is Reinforcement Learning in Bitcoin Option Pricing.  We used Q leaning to research the proper bitcoin option price at Jex Exchange(jex.com). Q leaning is model free and off policy reinforcement learing method. The real historical bitcoin option and spot price data is from Jex Exchange(jex.com) <br>

The reason we did this topic is that the option market of bitcoin and other new cryptocurrencies are still new. We are not sure if BS model is a good choise for option pricing in such new market (high volatility). Thus, we researched reinforcement learning in BS and compared our final RL result to BS model and real market data. In the jupyter notebook, we detailed the math derivation of BS model and dynamic programming solution for q learning and the way we computed the optimal q-function with DP approach. <br>

## Result


![alt text](https://github.com/HuaizheXu/Reinforcement-Learning-in-Bitcoin-Option-Pricing/blob/master/Result-RL-Bitcoin-Option.png)

## Keywords: <br>
 Black-Sholes Model <br>
 Dynamic Programming solution for Q Learning in BS Model <br>
 Compute optimal hedge and portfolio value <br>
 Compute the optimal Q-function with the DP approach <br>
