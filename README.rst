===============
regime_switch_model
===============
regime_switch_model is a set of algorithms for learning and inference
of the Regime-Switching Model.


Regime-Switching Model
===============
 Let :math:`y_t` be a :math:`p\times 1` observed time series and :math:`h_t` be a homogenous and stationary hidden Markov
chain taking values in :math:`\{1, 2, \dots, m\}` with transition probabilities
.. math::

    w_{kj} = P(h_{t+1}=j\mid h_t=k), \quad k,j=1, \dots, m

where the number of hidden states $m$ is known. It is assumed that the financial market in
each period can be realized as one of $m$ regime. Furthermore, the regimes are characterized
by a set of $J$ risk factors, which represent broad macro and micro economic indicators. Let $F_{tj}$ be the value of the $j$th risk factor $(j=1, \dots, J)$ in period $t$. Correspondingly, $F_t$ is the vector of risk factors in period $t$. We assumes that, for $t=1, \dots, n$, when the market is in regime $h_t$ in period $t$,
    \begin{equation}
    y_t = u_{h_t} + B_{h_t}F_t + \Gamma_{h_t}\epsilon_t,
    \end{equation}
where $\epsilon_t \sim N(0,I)$. The model parameters $\{u_{h_t}, B_{h_t}, \Gamma_{h_t}\}$ depend on the regime $h_t$. $u_{h_t}$ is the state-dependent intercepts of the linear factor model. The matrix $B_{h_t}$ defines the sensitivities of asset returns to the common risk factors in state $h_t$ and is often called the loading matrix.

`regime_switch_model` solves the following fundamental problems:
* Given the observed data, estimate the model parameters
* Given the model parameters and observed data, estimate the optimal sequence of hidden states


Important links
===============

* Official source code repo: https://github.com/Liuyi-Hu/regime_switch_model

Dependencies
============

The required dependencies to use regime_switch_model are

* Python >= 2.7
* NumPy (tested to work with >=1.13.1)
* SciPy (tested to work with >=0.19.0)
* scikit-learn >= 0.18.1


Installation
============

First make sure you have installed all the dependencies listed above. Then run
the following command::

    pip install -U --user regime_switch_model


Reference
============
* Ma, Ying, Leonard MacLean, Kuan Xu, and Yonggan Zhao. "A portfolio optimization model with regime-switching risk factors for sector exchange traded funds." Pac J Optim 7, no. 2 (2011): 281-296.

Credits
============
Part of the code are from Python package 'hmmlearn'. Their source code can be found here: https://github.com/hmmlearn/hmmlearn
