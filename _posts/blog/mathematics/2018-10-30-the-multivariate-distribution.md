---
layout: post
title: The Multivariate Distribution
permalink: blog/mathematics/the-multivariate-distribution
categories: [Mathematics]
---

The multivariate distribution of a random variable $$x \in \mathbb{R}^n$$ is written as $$x \sim \mathcal{N}(\mu, \Sigma)$$ where $$\mu \in \mathbb{R}^n$$ is the mean vector and $$\Sigma \in \mathbb{R}^{n \times n}$$ is the covariance matrix. $$\Sigma\geqslant 0$$ is symmetric and positive semi-definite. Note that the multivariate distribution generalizes the one-dimensional normal distribution to n-dimensions. The probability density function is given by:

<center>$$ p(x) = \frac{1}{(2\pi)^{n/2}\vert\Sigma\vert^{1/2}}exp(-\frac{1}{2}(x-\mu)^T\Sigma^{-1} (x-\mu)) $$</center>

For this multivariate distribution the expectation is given by:

<center>$$ \mathbb{E}[x] = \mu $$</center>

Also, the covariance matrix, which generalizes the notion of variance, is given by:

<center>$$ \begin{eqnarray} \Sigma &=& \mathbb{E}[(x-\mathbb{E}[x])(x-\mathbb{E}[x])^T] \\ &=& \mathbb{E}[xx^T-x\mathbb{E}[x^T]-\mathbb{E}[x]x^T+\mathbb{E}[x]\mathbb{E}[x]^T]\\ &=& \mathbb{E}[xx^T]-\mathbb{E}[x]\mathbb{E}[x^T]-\mathbb{E}[x]\mathbb{E}[x^T]+\mathbb{E}[x]\mathbb{E}[x^T]\mathbb{E}[1]\\ &=& \mathbb{E}[xx^T]-\mathbb{E}[x]\mathbb{E}[x^T] \end{eqnarray} $$</center>

A multivariate distribution with a zero mean vector and an identity matrix as the covariance matrix is known as the standard normal distribution.

## Marginals and Conditionals of Gaussians

Suppose that we have a vector-valued random variable $$x\sim \mathcal{N}(\mu,\Sigma)$$ where:

<center>$$ x = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} \\ \mu = \begin{bmatrix} \mu_1 \\ \mu_2 \end{bmatrix}\\ \Sigma = \begin{bmatrix} \Sigma_{11} & \Sigma_{12} \\ \Sigma_{21} & \Sigma_{22} \end{bmatrix}\\ $$</center>

Here $$x_1, \mu_1 \in \mathcal{R}^{n_1}$$, $$x_2, \mu_2 \in \mathcal{R}^{n_2}$$, $$x, \mu \in \mathcal{R}^{n_1+n_2}$$, $$\Sigma_{11} \in \mathcal{R}^{n_1 \times n_1}$$, $$\Sigma_{12} \in \mathcal{R}^{n_1 \times n_2}$$, $$\Sigma_{21} \in \mathcal{R}^{n_2 \times n_1}$$, $$\Sigma_{22} \in \mathcal{R}^{n_2 \times n_2}$$ and $$\Sigma \in \mathcal{R}^{(n_1+n_2) \times (n_1+n_2)}$$.

Note that $$\mathbb{E}(x_1)=\mu_1$$and $$\mathbb{E}(x_2)=\mu_2$$. Also, note that:

<center>$$ \begin{eqnarray} Cov(x) &=& \Sigma\\ &=& \mathbb{E}[(x-\mu)(x-\mu^T)]\\ &=& \mathbb{E}\left[\begin{bmatrix} x_1 - \mu_1 \\ x_2 - \mu_2\end{bmatrix}\begin{bmatrix} x_1 - \mu_1 \\ x_2 - \mu_2\end{bmatrix}^T\right]\\ &=& \mathbb{E}\left[\begin{bmatrix} x_1 - \mu_1 \\ x_2 - \mu_2\end{bmatrix}\begin{bmatrix} (x_1 - \mu_1)^T & (x_2 - \mu_2)^T\end{bmatrix}\right]\\ &=& \mathbb{E}\left[\begin{bmatrix} (x_1 - \mu_1)(x_1 - \mu_1)^T & (x_1 - \mu_1)(x_2 - \mu_2)^T \\ (x_2 - \mu_2)(x_1 - \mu_1)^T & (x_2 - \mu_2)(x_2 - \mu_2)^T\end{bmatrix}\right]\\ &=& \begin{bmatrix} \Sigma_{11} & \Sigma_{12} \\ \Sigma_{21} & \Sigma_{22} \end{bmatrix} \end{eqnarray} $$</center>

Note that $$Cov(x_1) = (x_1 - \mu_1)(x_1 - \mu_1)^T = \Sigma_{11}$$ and similarly $$Cov(x_2) = \Sigma_{22}$$.

Note that $$x$$ represents the joint multivariate density of $$x_1$$ and $$x_2$$. We have also found the marginal distributions of $$x_1$$ and $$x_2$$ to be $$\mathcal{N}(\mu_1,\Sigma_{11})$$ and $$\mathcal{N}(\mu_2,\Sigma_{22})$$ respectively.

It may also be shown that the conditional distribution $$x_1 \vert x_2$$ is given by $$\mathcal{N}(\mu_{1 \vert 2},\Sigma_{1 \vert 2})$$, where:

<center>$$ \begin{eqnarray} \mu_{1 \vert 2} &=& \mu_1 + \Sigma_{12}\Sigma_{22}^{-1}(x_2 - \mu_2)\\ \Sigma_{1 \vert 2} &=& \Sigma_{11} - \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21} \end{eqnarray} $$</center>