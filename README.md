**Machine Learning Option Pricing:**  
An empirical approach based on market data

# Table of Contents
1. [Introduction](#1-introduction)
2. [Pricing Model Specification](#2-pricing-model-specification)
   - 2.1 [Pricing Model](#21-pricing-model)
   - 2.2 [Volatility Estimation](#22-volatility-estimation)
   - 2.3 [Assumptions](#23-assumptions)
3. [Neural Network Specification](#3-neural-network-specification)
   - 3.1 [Scope](#31-scope)
   - 3.2 [Data Generation Method](#32-data-generation-method)
4. [References](#4-references)



# 1. Introduction

The main scope of this paper is to devise a data generation method that can be reliably used to train machine learning algorithms in pricing options under stochastic volatility. It has been demonstrated by Frey et al. (2022) that Single Layer and Deep Neural networks are able to accurately predict vanilla option prices given a large synthetic training dataset. 

The data generation method proposed is an extension of that in Frey et al. (2022) with additional considerations regarding the feasibility of all feature combinations. The machine learning model specification is nearly identical to that of Frey et al. (2022) with minor exceptions including the use of relative absolute pricing error against relative moneyness as a performance metric, as well as exploration of additional parameterisation.

Following this eniqury into the felixbility of this method, it was considered whether ScikitLearn Neural Network models could be used to price path-dependent exotic options. Thus, we aim to generalize a data generation routine utilising generic at-the-money volatilites which will allow us to train the model on data strictly adhering to desired market conditions while allowing for accurately calibrated stochastic volatility for each trading day simulated in part.


# 2. Pricing Model Specification
## 2.1 Pricing Model

To model the logarithmic price of our underlying security, we use the Heston (1993) model, described by the pair of stochastic differential equations:

$$
dX_t = \left( r - \frac{v_t}{2} \right) dt + \sqrt{v_t} \left( \rho dW_t + \sqrt{1 - \rho^2} dB_t \right) \quad (1)
$$

$$
\hspace{1.9cm}  dv_t = \kappa (\theta - v_t) dt + \eta \sqrt{v_t} dW_t \hspace{1.8cm} \quad (1.1)
$$


where
- $v_0$ represents the initial variance,
- $\theta$ is the long-run variance,
- $\rho$ is the correlation between the log-price process and its volatility,
- $\kappa$ is the mean reversion of the variance to **𝜃**,
- $\eta$ is the volatility of the variance process, and 
- $B_t$ , $W_t$ are continuous random walks. 

## 2.2 Volatility Estimation
The model becomes suitable for fitting to our proposed method via approximation of implied volatilities as devised by Derman (2008):

$$
\hspace{1cm} \sigma(K, t_0) = \sigma_{\text{atm}}(S_0, t_0) - b(t_0)(K - S_0) \hspace{1cm} \quad (2)
$$

<br>

## 2.3 Assumptions
While the specification in $$(1)$$ and $$(1.1)$$ is flexible enough to permit stochastic volatility with correlation between the asset process and its variance process, we make a limiting assumption around the topological nature of the volatility surface being constant with $$(2)$$.

# 3. Neural Network Specification
## 3.1 Scope
The scope of our proposed historical simulation method is to test whether machine learning estimations of pricing functions can reliably price large volumes of exotic options in as close of a realistic trading scenario as our methods permit. The main considerations will be around frequency of retraining and choice of model features. In all cases, the model will have the minimum of four features: underlying spot price, strike price, days to maturity, and a categorical 'put'/'call' flag classified via one-hot-encoding. Further, the model will include the additional features of barrier level and barrier type and potentially Heston pricing parameters. Later, the model will be extended to Asian arithmetic and geometric options with similar considerations.

## 3.2 Data Generation Method
In order to achieve homogenous results, where each trading day is distributionally comparative to the others, we calibrate a set of global parameters across a number of strikes and set maturities for every trading day simulated  (i.e., one set of $$\kappa$$, $$\theta$$, $$\rho$$, $$\eta$$ and $$v_0$$ across a maximum of five strikes and six maturities, total of thirty options) This method will yield lower calibration accuracy but will allow us to create a dataset that represents the option price as a functional of its features across multiple trading scenarios. With calibrated parameters for every trading day, a dataset is generated to represent all pheasible contract feature combinations for a given option. This process is iterated over every historical trading day, resulting in a market-based distirbution of historical option prices accompanied by all repsective pricing inputs.

# 4. References
Blanda, V. (2023). FX Barrier Option Pricing. 
<br> https://www.imperial.ac.uk/media/imperial-college/faculty-of-natural-sciences/department-of-mathematics/math-finance/212252650---VALENTIN-BLANDA---BLANDA_VALENTIN_02293988.pdf

De La Rosa, A. (2024) HESTON MODEL CALIBRATION USING QUANTLIB IN PYTHON
<br>https://medium.com/@aaron_delarosa/heston-model-calibration-using-quantlib-in-python-0089516430ef

Derman, E. (2008). Lecture 9: Patterns of Volatility Change.
<br> https://emanuelderman.com/wp-content/uploads/2013/09/smile-lecture9.pdf 

Frey, C., Scheuch, C., Voigt, S. and Weiss, P. (2022). Option Pricing via Machine Learning with Python. Tidy Finance.
<br> https://www.tidy-finance.org/python/option-pricing-via-machine-learning.html

Gavin, H. (2024) The Levenberg-Marquardt algorithm for nonlinear least squares curve-fitting problems.
<br> https://people.duke.edu/~hpgavin/lm.pdf

Heston, S. (1993). A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options.
<br> https://www.ma.imperial.ac.uk/~ajacquie/IC_Num_Methods/IC_Num_Methods_Docs/Literature/Heston.pdf

Lewis, S., Kwon, Y. (2000). Pricing Barrier Option Using Finite Difference Method and MonteCarlo Simulation.
<br> https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=d72456ecdcb08ae63f5738e41771ccd16dd2b53e

Schönbucher, P. (1998) A Market Model for Stochastic Implied Volatility.
<br> https://papers.ssrn.com/sol3/papers.cfm?abstract_id=182775
<br>
