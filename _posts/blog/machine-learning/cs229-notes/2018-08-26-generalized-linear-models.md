---
layout: post
title: Generalized Linear Models
permalink: blog/machine-learning/cs229-notes/generalized-linear-models
categories: [Machine Learning, CS229 Notes]
---

## 1\. The Exponential Family

We shall refer to probability distributions of the following form to be members of the exponential family:

<center>$$ p(y;\eta) = b(y)exp(\eta^TT(y)-a(\eta)) $$</center>

where the $$p(y)$$ is being parameterized by the variable natural (or canonical) parameter $$\eta$$; $$T(y)$$ is the sufficient statistic (for the distribution) and; $$a(\eta)$$ is the log partition function. The term $$exp{(-a(\eta))}$$ essentially plays the role of a normalization constant, ensuring that $$p(y;\eta)$$ remains within the bounds of $$0$$ and $$1$$.

A fixed choice of $$b(y)$$, $$T(y)$$ and $$a(\eta)$$ defines a family (or set) of distributions. Varying $$\eta$$ then gives different distributions within that family of distributions.

We shall now show that the Bernoulli distribution $$y;\phi \sim Bernoulli(\phi)$$ and the Gaussian distribution $$y; \mu,\sigma \sim \mathcal{N} (\mu, \sigma^2)$$ are members of the exponential family.

### (i) Bernoulli Distribution

The random variable of a Bernoulli distribution can take on only two values $$0$$ or $$1$$. Therefore, $$y \sim Bernoulli(\phi)$$ means:

<center>$$ \begin{eqnarray} p(y=1;\phi) &=& \phi \\ p(y=0;\phi) &=& 1-\phi \end{eqnarray} $$</center>

Or, compactly:

<center>$$ p(y;\phi) = (\phi)^y(1-\phi)^{1-y} $$</center>

As $$z = exp(log(z))$$, we have:

<center>$$ \begin{eqnarray} p(y;\phi) &=& exp(ylog(\phi)+(1-y)log(1-\phi)) \\ &=& exp\left(\left(log\left(\frac{\phi}{1-\phi}\right)\right)y + log(1-\phi)\right) \end{eqnarray} $$</center>

Comparing the above equation with the general equation of the exponential family yields:

<center>$$ \begin{eqnarray} \eta &=& log\left(\frac{\phi}{1-\phi}\right)\\ b(y) &=& 1\\ T(y) &=& y\\ a(\eta) &=& -log(1-\phi) \end{eqnarray} $$</center>

Note that:

<center>$$ \begin{eqnarray} \phi &=& \frac{1}{1+exp(-\eta)}\\ a(\eta) &=& -log\left(1-\frac{1}{1+exp(-\eta)}\right)\\ &=& -log\left(1-\frac{exp(\eta)}{1+exp(\eta)}\right)\\ &=& log(1+exp(\eta)) \end{eqnarray} $$</center>

### (ii) Gaussian Distribution

For linear regression models we showed that $$y \vert x; \theta \sim \mathcal{N}(\mu,\sigma^2)$$ under a certain set of assumptions. We also showed that the both the resulting hypothesis $$h_\theta(x)$$ and $$\theta$$ itself were independent of the value of $$\sigma^2$$. Therefore, to simplify our discussion below we shall set $$\sigma^2 = 1$$:

<center>$$ \begin{eqnarray} p(y;\mu) &=& \frac{1}{\sqrt{2\pi}}exp\left(-\frac{(y-\mu)^2}{2}\right)\\ &=& \frac{1}{\sqrt{2\pi}}exp\left(-\frac{y^2}{2}\right)exp\left(\mu y -\frac{\mu^2}{2}\right) \end{eqnarray} $$</center>

Therefore:

<center>$$ \begin{eqnarray} \eta &=& \mu \\ b(y) &=& \frac{1}{\sqrt{2\pi}}exp\left(-\frac{y^2}{2}\right) \\ T(y) &=& y \\ a(y) &=& \frac{\mu^2}{2} = \frac{\eta^2}{2} \end{eqnarray} $$</center>

## 2\. Generalized Linear Models (GLMs)

In order to define generalized linear models we shall make the following assumptions:

1.  $$y \vert x; \theta \sim Exponential\ Family(\eta)$$ where $$\eta$$ may depend on $$\theta$$ and $$x$$.
2.  $$h_\theta(x) = \mathbb{E}[T(y) \vert x; \theta]$$, i.e., we need to predict the expected value of the sufficient statistic $$T(y)$$ given $$x$$ and the parameter $$\theta$$. This is a reasonable assumption because $$T(y)$$, being the sufficient statistic, contains all the ‘information’ regarding this particular distribution. Note that, by definition, a statistic is sufficient with respect to a statistical model and its unknown parameter if no other statistic derived from the same sample provides additional information as to the value of that parameter.
3.  $$\eta = \theta^Tx$$ or $$\eta_i = \theta_i^Tx$$ if $$\eta$$ is a vector (this may also be thought of a design choice rather than an assumption).

We shall now first derive the linear regression and logistic regression models from the Gaussian and Bernoulli distributions respectively using this definition of GLMs and then introduce the softmax regression model that is based on the multinomial distribution and is a generalization of logistic regression.

In the following discussion, we shall refer to the function $$g(\eta) = \mathbb{E}[T(y);\eta]$$ as the canonical response function and its inverse $$g^{-1}$$ as the canonical link function.

### (i) Linear Regression

Let $$y \vert x \sim \mathcal{N}(\mu,\sigma^2)$$. We have from earlier that (for the Gaussian distribution) $$T(y)=y$$. Therefore, $$\mathbb{E}[T(y) \vert x; \theta] = \mathbb{E}[y \vert x; \theta] = \mu$$. We also know that $$\mu = \eta$$. As per the third assumption, $$\eta = \theta^Tx$$. As a result of the second assumption, we have:

<center>$$ \begin{eqnarray} h_\theta(x) &=& \mathbb{E}[T(y) \vert x;\theta]\\ &=& \theta^Tx \end{eqnarray} $$</center>

which is the linear regression model we presented earlier.

Note that the canonical response function $$g(\eta)$$ is just the identity function.

### (ii) Logistic Regression

Let $$y \vert x \sim Bernoulli(\phi)$$. We have from earlier that (for the Bernoulli distribution) $$T(y)=y$$. Therefore:

<center>$$ \begin{eqnarray} \mathbb{E}[T(y) \vert x; \theta] &=& \mathbb{E}[y \vert x; \theta] \\ &=& \sum_{a \in Val(y)} a\ p(y=a \vert x; \theta) \\ &=& (0)p(y=0 \vert x; \theta) + (1)p(y=1 \vert x; \theta) \\ &=& \phi \end{eqnarray} $$</center>

We also know that $$\phi=\frac{1}{1+exp(-\eta)}$$. Therefore:

<center>$$ \begin{eqnarray} h_\theta(x) &=& \mathbb{E}[T(y) \vert x; \theta] \\ &=& \frac{1}{1+exp(-\eta)}\\ &=& \frac{1}{1+exp(-\theta^Tx)} \end{eqnarray} $$</center>

which is the logistic regression model we presented earlier.

Note that the canonical response function $$g(\eta) = \frac{1}{1+exp(-\eta)}$$.

### (iii) Softmax Regression

Let $$y \in \{1,...,k\}$$ for a given $$x$$. Let there also be a set $$\{\phi_1, ..., \phi_{k-1}\}$$ such that the probability that $$y$$ takes on value $$i$$, where $$i \in \{1,...,k-1\}$$ is given by $$\phi_i$$. The probability that $$y=k$$ is then given by $$\phi_k = 1-\sum_{i=0}^{k-l}\phi_i$$. Let us define an indicator function $$1\{\mathbb{o}\}$$ that is $$1$$ when its argument is $$True$$ and $$0$$ when it is $$False$$. Finally, let us define $$T(y)$$ to be a vector of length $$k-1$$, i.e. $$T(y) \in \mathbb{R}^{k-1}$$ such that $$T(y)_i = 1\{y=i\}$$. Also, for notational convenience define $$T(y)_k=1\{y=k\}$$. Therefore:

<center>$$ \begin{eqnarray} p(y \vert x; \theta) &=& \phi_1^{T(y)_1}\phi_2^{T(y)_2}...\phi_{k-1}^{T(y)_{k-1}}\phi_k^{T(y)_k} \\ &=& \phi_1^{T(y)_1}\phi_2^{T(y)_2}...\phi_{k-1}^{T(y)_{k-1}}\phi_k^{1-\sum_{i=1}^{k-1}T(y)_i} \end{eqnarray} $$</center>

This is the multinomial distribution - a generalization of the Bernoulli distribution to $$k$$ classes. We shall now show that the multinomial distribution is a member of the Exponential Family.

<center>$$ \begin{eqnarray} p(y \vert x; \theta) &=& exp\left(log(\phi_1^{T(y)_1}...\phi_{k-1}^{T(y)_{k-1}}\phi_k^{1-\sum_{i=1}^{k-1}T(y)_i})\right) \\ &=& exp\left(T(y)_1log(\phi_1)+...+T(y)_{k-1}log(\phi_{k-1})+\left({1-\sum_{i=1}^{k-1}T(y)_i}\right)log(\phi_k)\right) \\ &=& exp\left(T(y)_1log\left(\frac{\phi_1}{\phi_k}\right)+...+T(y)_{k-1}log\left(\frac{\phi_{k-1}}{\phi_k}\right)+log(\phi_k)\right) \\ \end{eqnarray} $$</center>

Let:

<center>$$ \eta = \begin{bmatrix} log{\frac{\phi_1}{\phi_k}}\\ . \\ . \\ . \\ log{\frac{\phi_{k-1}}{\phi_k}} \end{bmatrix} $$</center>

For notational convenience we shall also define $$\eta_k = log\left(\frac{\phi_k}{\phi_k}\right)=0$$.

Therefore:

<center>$$ p(y \vert x; \theta) = exp\left(\eta^TT(y) + log(1-\sum_{i=1}^{k-1}\phi_i)\right) $$</center>

This shows that the multinomial distribution does indeed belong to the Exponential Family. Note:

<center>$$ \begin{eqnarray} a(\eta) &=& -log\left(1-\sum_{i=1}^{k-1}\phi_i\right) \\ b(y) &=& 1 \end{eqnarray} $$</center>

The canonical link function is thus given by:

<center>$$ \eta_i = log\left(\frac{\phi_i}{\phi_k}\right) $$</center>

Note that the canonical response function $$g(\eta)$$ expresses the expectation $$\mathbb{E}[T(y); \eta]$$ in terms of $$\eta$$ whereas the canonical link function $$g^{-1}$$ expresses $$\eta$$ in terms of the expectation $$\mathbb{E}[T(y); \eta]$$. Note that the above equation satisfies this condition because $$\mathbb{E}[T(y) _i; \eta_i] = \phi_i$$.

Therefore:

<center>$$ \begin{eqnarray} exp(\eta_i) &=& \frac{\phi_i}{\phi_k} \\ \phi_kexp(\eta_i) &=& \phi_i \\ \phi_k\sum_{i=1}^{k}exp(\eta_i) &=& \sum_{i=1}^{k} \phi_i \\ \phi_k &=& \frac{1}{\sum_{i=1}^{k}exp(\eta_i)} \end{eqnarray} $$</center>

Substituting the value of $$\phi_k$$ in the second equation above gives:

<center>$$ \phi_i = \frac{exp(\eta_i)}{\sum_{j=1}^{k}exp(\eta_j)} $$</center>

From our third assumption we have $$\eta_i=\theta^T_ix$$. Therefore:

<center>$$ \phi_i = \frac{exp(\theta_i^Tx)}{\sum_{j=1}^{k}exp(\theta_j^Tx)} $$</center>

This is the softmax regression model - a generalization of the logistic regression model to $$k$$ classes.

As per the second assumption, our hypothesis $$h_\theta(x) = \mathbb{E}[T(y) \vert x]$$. Therefore:

<center>$$ \begin{eqnarray} h_\theta(x) &=& \begin{bmatrix}E(y=1|x;\theta)\\E(y=2|x;\theta)\\.\\.\\.\\E(y=k-1|x;\theta)\end{bmatrix} \\ &=& \begin{bmatrix} \phi_1 \\ \phi_2 \\.\\.\\.\\ \phi_{k-1} \end{bmatrix} \\ &=& \begin{bmatrix} \frac{exp(\theta_1^Tx)}{\sum_{j=1}^{k}exp(\theta_j^Tx)} \\ \frac{exp(\theta_2^Tx)}{\sum_{j=1}^{k}exp(\theta_j^Tx)} \\.\\.\\.\\ \frac{exp(\theta_{k-1}^Tx)}{\sum_{j=1}^{k}exp(\theta_j^Tx)} \end{bmatrix} \end{eqnarray} $$</center>

Suppose that we are given a training set $$\{(x^{(i)}, y^{(i)}\}$$ where $$i \in \{1,...,m\}$$ and our goal is to find the parameter $$\theta$$ that maps the $$x^{(i)}$$’s to the $$y^{(i)}$$’s. Let us define a likelihood function as follows:

<center>$$ \begin{eqnarray} L(\theta) &=& \prod_{i=1}^{m} p(y^{(i)} \vert x^{(i)}; \theta) \\ &=& \prod_{i=1}^{m} \prod_{j=1}^{k} \phi_j^{1\{y^{(i)}=j\}} \\ &=& \prod_{i=1}^{m} \prod_{j=1}^{k} \frac{exp(\theta_j^Tx)}{\sum_{l=1}^{k}exp(\theta_l^Tx)} \end{eqnarray} $$</center>

We can now use gradient descent or the Netwon-Raphson method to find the value of $$\theta$$ that maximizes $$\mathcal{l}(\theta) = log(L(\theta))$$.