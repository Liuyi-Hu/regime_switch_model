
Tutorial
========

Regime-Switching Model
----------------------

``regime_switch_model`` is a set of algorithms for learning and
inference on regime-switching model. Let :math:`y_t` be a
:math:`p\times 1` observed time series and :math:`h_t` be a homogenous
and stationary hidden Markov chain taking values in
:math:`\{1, 2, \dots, m\}` with transition probabilities

.. raw:: latex

   \begin{equation}
       w_{kj} = P(h_{t+1}=j\mid h_t=k), \quad k,j=1, \dots, m
       \end{equation}

where the number of hidden states :math:`m` is known. It is assumed that
the financial market in each period can be realized as one of :math:`m`
regime. Furthermore, the regimes are characterized by a set of :math:`J`
risk factors, which represent broad macro and micro economic indicators.
Let :math:`F_{tj}` be the value of the :math:`j`\ th risk factor
:math:`(j=1, \dots, J)` in period :math:`t`. Correspondingly,
:math:`F_t` is the vector of risk factors in period :math:`t`. We
assumes that, for :math:`t=1, \dots, n`, when the market is in regime
:math:`h_t` in period :math:`t`,

.. raw:: latex

   \begin{equation}
       y_t = u_{h_t} + B_{h_t}F_t + \Gamma_{h_t}\epsilon_t,
       \end{equation}

where :math:`\epsilon_t \sim N(0,I)`. The model parameters
:math:`\{u_{h_t}, B_{h_t}, \Gamma_{h_t}\}` depend on the regime
:math:`h_t`. :math:`u_{h_t}` is the state-dependent intercepts of the
linear factor model. The matrix :math:`B_{h_t}` defines the
sensitivities of asset returns to the common risk factors in state
:math:`h_t` and is often called the loading matrix.

``regime_switch_model`` solves the following fundamental problems: \*
Given the observed data, estimate the model parameters \* Given the
model parameters and observed data, estimate the optimal sequence of
hidden states

The implementation of code is based on the well-known Baum-Welch
algorithm and Viterbi algorithm that are widely used in hidden Markov
model.

.. code:: ipython2

    import numpy as np
    import pandas as pd
    from regime_switch_model.rshmm import *

Generate samples based on the regime-switching model
----------------------------------------------------

.. code:: ipython2

    model = HMMRS(n_components=2)
    # startprob
    model.startprob_ = np.array([0.9, 0.1])
    
    # transition matrix
    model.transmat_ = np.array([[0.9, 0.1], [0.6, 0.4]])
    # risk factor matrix
    # read file from Fama-French three-factor data
    Fama_French = pd.read_csv('Global_ex_US_3_Factors_Daily.csv', skiprows=3)
    Fama_French.rename(columns={'Unnamed: 0': 'TimeStamp'}, inplace=True)
    Fama_French.replace(-99.99, np.nan);
    Fama_French.replace(-999, np.nan);
    
    # select data 
    #Fama_French_subset = Fama_French[(Fama_French['TimeStamp'] >= 20150101) & (Fama_French['TimeStamp'] <= 20171231)]
    Fama_French_subset = Fama_French
    Fama_French_subset.drop(['TimeStamp', 'RF'], axis=1, inplace=True)
    F = np.hstack((np.atleast_2d(np.ones(Fama_French_subset.shape[0])).T, Fama_French_subset))
    
    # loading matrix with intercept
    loadingmat1 = np.array([[0.9, 0.052, -0.02], 
                            [0.3, 0.27, 0.01], 
                            [0.12, 0.1, -0.05], 
                            [0.04, 0.01, -0.15], 
                            [0.15, 0.04, -0.11]])
    intercept1 = np.atleast_2d(np.array([-0.015, -0.01, 0.005, 00.1, 0.02])).T
    
    model.loadingmat_ = np.stack((np.hstack((intercept1, loadingmat1)), 
                                  np.hstack((0.25*intercept1, -0.5* loadingmat1))), axis=0)
    
    # covariance matrix
    n_stocks = 5
    rho = 0.2
    Sigma1 = np.full((n_stocks, n_stocks), rho) + np.diag(np.repeat(1-rho, n_stocks))
    model.covmat_ = np.stack((Sigma1, 10*Sigma1), axis=0)
    
    save = True
    # sample
    Y, Z = model.sample(F)


Split data into training and test
---------------------------------

.. code:: ipython2

    # Use the last 300 day as the test data
    Y_train = Y[:-300,:]
    Y_test = Y[-300:,:]
    F_train = F[:-300,:]
    F_test = F[-300:,:]

Fitting Regime-Switch Model
---------------------------

.. code:: ipython2

    remodel = HMMRS(n_components=2, verbose=True)
    remodel.fit(Y_train, F_train)
    Z2, logl, viterbi_lattice = remodel.predict(Y_train, F_train)


.. parsed-literal::

             1      -63535.1590              nan
             2      -60972.1968        2562.9622
             3      -59533.7367        1438.4601
             4      -56005.9127        3527.8240
             5      -54584.0500        1421.8628
             6      -54259.0186         325.0314
             7      -54199.8384          59.1802
             8      -54192.7580           7.0804
             9      -54192.0477           0.7103
            10      -54191.9793           0.0684
            11      -54191.9727           0.0065
            12      -54191.9721           0.0006
            13      -54191.9720           0.0001


Examine model parameters
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython2

    np.set_printoptions(precision=2)
    
    print("Number of data points = ", Y_train.shape[0])
    print(" ")
    print("Starting probability")
    print(remodel.startprob_)
    print(" ")
    
    print("Transition matrix")
    print(remodel.transmat_)
    print(" ")
    
    print("Means and vars of each hidden state")
    for i in range(remodel.n_components):
        print("{0}th hidden state".format(i))
        print("loading matrix = ", remodel.loadingmat_[i])
        print("covariance = ", remodel.covmat_[i])
        print(" ")


.. parsed-literal::

    ('Number of data points = ', 6723)
     
    Starting probability
    [  1.78e-13   1.00e+00]
     
    Transition matrix
    [[ 0.37  0.63]
     [ 0.11  0.89]]
     
    Means and vars of each hidden state
    0th hidden state
    ('loading matrix = ', array([[-0.14, -0.53, -0.15, -0.03],
           [-0.04, -0.04, -0.17,  0.06],
           [ 0.04, -0.15, -0.23, -0.02],
           [ 0.02,  0.02,  0.19, -0.  ],
           [ 0.01,  0.07, -0.04,  0.24]]))
    ('covariance = ', array([[ 10.55,   1.92,   2.14,   1.56,   1.37],
           [  1.92,   9.53,   1.68,   2.19,   1.74],
           [  2.14,   1.68,   9.44,   1.85,   1.71],
           [  1.56,   2.19,   1.85,   9.89,   2.62],
           [  1.37,   1.74,   1.71,   2.62,  10.24]]))
     
    1th hidden state
    ('loading matrix = ', array([[ -6.79e-03,   8.93e-01,   3.71e-02,   4.66e-02],
           [ -2.98e-02,   2.93e-01,   2.42e-01,   1.15e-01],
           [  7.01e-04,   1.09e-01,   6.24e-02,  -2.15e-02],
           [  9.63e-02,   4.09e-02,  -1.29e-02,  -1.78e-01],
           [  1.62e-02,   1.48e-01,  -2.12e-04,  -1.60e-01]]))
    ('covariance = ', array([[ 0.98,  0.2 ,  0.24,  0.19,  0.2 ],
           [ 0.2 ,  0.98,  0.21,  0.21,  0.17],
           [ 0.24,  0.21,  1.01,  0.19,  0.21],
           [ 0.19,  0.21,  0.19,  0.94,  0.19],
           [ 0.2 ,  0.17,  0.21,  0.19,  0.98]]))
     


Examine the predicted hidden state
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython2

    print("Prediction accuracy of the hidden states = ", np.mean(np.equal(Z[:-300], 1-Z2)))


.. parsed-literal::

    ('Prediction accuracy of the hidden states = ', 0.9840844860925182)

