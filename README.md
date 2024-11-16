**Machine Learning Option Pricing:**  
An empirical approach based on SPX options trade data

# Table of Contents
1. [Introduction](#1-introduction)
2. [Pricing Model](#2-pricing-model)<br>
   2.1. [Specification](#21-specification)<br>
   2.2. [Historical Parameter Retrieval](#22-historical-parameter-retrieval)
3. [Data Generation](#3-data-generation)<br>
   3.1. [Barrier Options](#31-barrier-options)<br>
   3.2. [Asian Options](#32-asian-options)
5. [Reference List](#6-reference-list)



# 1. Introduction

In this paper we will explore a proposed method of pricing exotic options via multi-layer-perceptron-based approximations derived from the simulation of a multidimensional space representing an option's price as a functional form of its features. To achieve this, we calibrate historical Heston (1993) parameters using market observed risk-free and dividend rates accompanied by live options trade data, thereby effectively simulating, in the case of this paper, the SPX index options market. This paper serves as a framework and demonstration of a generalized estimation process for barrier and Asian options along with a model specification and retraining analysis of both pricing models. We will explore the estimation of Barrier and Asian options priced via Finite Difference and Monte Carlo simulation, respectively.

# 2. Pricing Model

## 2.1. Specification

To model the logarithmic price of our underlying security, we use the Heston (1993) model, described by the pair of stochastic differential equations:

$$
dX_t = \left( r - \frac{v_t}{2} \right) dt + \sqrt{v_t} \left( \rho dW_t + \sqrt{1 - \rho^2} dB_t \right) \quad (2.1)
$$

$$
\hspace{1.9cm}  dv_t = \kappa (\theta - v_t) dt + \eta \sqrt{v_t} dW_t \hspace{1.8cm} \quad (2.2)
$$


where
- $v_0$ represents the initial variance,
- $\theta$ is the long-run variance,
- $\rho$ is the correlation between the log-price process and its volatility,
- $\kappa$ is the mean reversion of the variance to **𝜃**,
- $\eta$ is the volatility of the variance process, and 
- $B_t$ , $W_t$ are continuous random walks. 

Heston (1993) famously derives the above model as an extension of the previously established Black and Scholes (1973) model for pricing options while lifting the well-known assumption of constant volatility. The addition of an auxiliary variance process replacing the otherwise assumed constant volatility parameter $\sigma$ in the renowned Black-Scholes formula allows for time-dependent, discretely measurable volatility. Consequently, implementation of the Heston (1993) model for governing the underlying log price is imperative to the functionality of our model as we are aiming to estimate prices of path-dependent options which may require discrete monitoring of the spot price throughout a contract's tenor.

## 2.2. Historical Parameter Retrieval

We aim to create a dataset of historical parameter sets in order to simulate a market with minimal to no assumptions relating to the bounds, dispersion, or any other statistical feature of the data. Synthetic sample generation is typically more popular and has been explored in detail by Liu et. al. (2019) in estimating implied volatility using artificial neural networks. However, by exploiting relatively small amounts amount of live trades data, one is able to attempt the automated reconstruction of volatility surfaces via discretization of the data and logic applied to trade times and volumes. One such proprietary method permitted the extraction of c. 1600 individual calibrations for as many unique live underlying spot prices from 2012 to 2024, which is far beyond the requirements for an accurate model as described by our specifications. The time continuity of the data is not necessarily guaranteed, however, for the purposes of our approximation we are more concerned with ensuring each calibration surface contains enough skew in the form of multiple strikes spread both above and below the corresponding underlying price. In our method, this is accounted for by disregarding all reconstructed surfaces which do not have at least two strikes on each side of the spot price with a minimum of five contracts as needed to calibrate the Heston (1993) stochastic volatility model. The optimization algorithm of choice in our study is that of the Levenberg-Marquard algorithm as described in detail by Gavin (2024) and implemented via QuantLib's Heston Model Calibration Helper.

# 3. Data Generation

In the spirit of Liu et. al. (2019) and Frey et. al. (2022) we will generate a development dataset by simulating possible parameter combinations for a given security. Liu et. al. (2019) demonstrate a considerable increase in computational efficiency with retention of low errors for the estimation of implied volatilities via artificial neural networks by considering the relative spot price (i.e., the spot price $S$ divided by the strike price $K$) and the relative option price (i.e., the option's price $C$ divided by its strike $K$) as opposed to their levels, a method we will be borrowing for our estimation. Frey et. al. (2022) propose a data generation method via Cartesian product to create a sample space of vanilla option pricing features to estimate the price level ($S$) using a multi-layer perceptron model. Testing of this method considering exotic options did not retain pricing accuracy as evidenced by high Root Mean Squared Error (RMSE), high Mean Absolute Error (MSE), and high partial dependence of the target price in relation to the underlying spot price level $S$ and the initial variance $v_0$. We therefore propose a new method combining the Cartesian product approach to retain control over feature combinations while considering the option's relative price ($C/K$) and any other linear features also scaled by the strike price $K$.

## 3.1. Barrier Options

In the case of barrier options, we begin generating the development dataset by iterating through the historical spot prices and volatility parameters and for each observation, performing the Cartesian product:

$$
S \times K \times T \times B \times R = \set{ (s, k, t, b, r_{\text{rebate}}) | \ s \in S, \ k \in K, \ t \in T, \ b \in B, \ \text{and} \ r_{\text{rebate}} \in R\} \quad (3.1)
$$

where
- $S$ is a single element set consisting of the underying spot price, <br>
- $K$ is a set of strikes spread around the spot, <br>
- $T$ is a set of maturities, <br>
- $B$ is a set of barrier levels, and <br>
- $R$ is a set of rebates which for the purposes of this study in a set consisting of only the element $0$ (zero)

## 3.2. Asian Options
For the Asian option counterpart, we perform a similar iteration of our historical data, for each observation performing the Cartesian product:

$$
S \times K \times T \times A \times P = \set{ (s,k,t,a,p) | \ s \in S, \ k \in K, \ t \in T, \ a \in A, \text{and} \ p \in P \} \quad (3.2)
$$

where
- $S$ is a single element set consisting of the underying spot price, <br>
- $K$ is a set of strikes spread around the spot, <br>
- $T$ is a set of maturities, <br>
- $A$ is a set of time frequencies at which the Asian option fixes (i.e., a frequency which determines the number of monitoring dates), and
- $P$ is a set of past fixings which for the purposes of this study is a set consisting of only the element $0$ (zero)

# 4. Model Training

With adequate generated data, we are now able to train a Multi-Layer Perceptron network to numerically approximate the relationship between our pricing features and the target price.


![Graph of model specification](README/MLP.png)


# 6. Reference list

<a href="https://www.mybib.com/b/G0Rbd7">View Bibliography</a>
