**Machine Learning Option Pricing:**  
An empirical approach based on market data

# Table of Contents
1. [Introduction](#1-introduction)
2. [Pricing Model](#2-pricing-model)
3. [Price Estimation](#3-price-estimation)
4. [Reference List](#4-reference-list)



# 1. Introduction

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

# 3. Price Estimation


# 4. Reference list
Blanda, V. (2023). FX Barrier Option Pricing. [online] <br> Available at: https://www.imperial.ac.uk/media/imperial-college/faculty-of-natural-sciences/department-of-mathematics/math-finance/212252650---VALENTIN-BLANDA---BLANDA_VALENTIN_02293988.pdf.

Frey, C., Scheuch, C., Voigt , S. and Weiss, P. (2022). Option Pricing via Machine Learning with Python. [online] Tidy Finance. <br> Available at: https://www.tidy-finance.org/python/option-pricing-via-machine-learning.html.

Gavin, H. (2024). The Levenberg-Marquardt algorithm for nonlinear least squares curve-fitting problems. [online] <br> Available at: https://people.duke.edu/~hpgavin/lm.pdf.

Heston, S.L. (1993). A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options. Review of Financial Studies, 6(2), pp.327–343. <br> doi:https://doi.org/10.1093/rfs/6.2.327.

Liu, S., Oosterlee, C. and Bohte, S. (2019). Pricing Options and Computing Implied Volatilities using Neural Networks. Risks, 7(1), p.16. <br> doi:https://doi.org/10.3390/risks7010016.

Schönbucher, P.J. (1999). A Market Model for Stochastic Implied Volatility. SSRN Electronic Journal, 21(4). <br> doi:https://doi.org/10.2139/ssrn.182775.

Van Wieringen, W. (2021). Lecture notes on ridge regression. [online] <br> Available at: https://arxiv.org/pdf/1509.09169.
