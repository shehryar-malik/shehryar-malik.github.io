---
layout: post
title: Generative Learning Algorithms
permalink: blog/machine-learning/cs229-notes/generative-learning-algorithms
categories: [Machine Learning, CS229 Notes]
---

So far we have looked at discriminative algorithms that try to model the distribution $$p(y \vert x)$$. We shall now study generative learning algorithms that model the distribution $$p(x \vert y)$$ and then use Bayes’ Rule to find $$p(y \vert x)$$:

<center>$$ p(y \vert x) = \frac{p(x \vert y)p(y)}{p(x)} $$</center>

Note that we want $$\underset{y}{argmax}$$ $$p(y \vert x)$$. Therefore:

<center>$$ \begin{eqnarray} \underset{y}{argmax}\ p(y \vert x) &=& \underset{y}{argmax}\ \frac{p(x \vert y)p(y)}{p(x)}\\ &=& \underset{y}{argmax}\ p(x \vert y)p(y)\\ &=& \underset{y}{argmax}\ p(x, y) \end{eqnarray} $$</center>

We shall now discuss two such algorithms.

## 1\. Gaussian Discriminant Analysis

Let us first introduce the multivariate distribution $$x \sim \mathcal{N}(\mu, \Sigma)$$ where $$x \in \mathbb{R}^n$$ is a random variable, $$\mu \in \mathbb{R}^n$$ is the mean vector and $$\Sigma \in \mathbb{R}^{n \times n}$$ is the covariance matrix. $$\Sigma\geqslant 0$$ is symmetric and positive semi-definite. The probability density function is given by:

<center>$$ p(x) = \frac{1}{(2\pi)^{n/2}\vert\Sigma\vert^{1/2}}exp(-\frac{1}{2}(x-\mu)^T\Sigma^{-1} (x-\mu)) $$</center>

For this multivariate distribution the expectation is given by:

<center>$$ \mathbb{E}[x] = \mu $$</center>

Also, the covariance matrix, which generalizes the notion of variance, is given by:

<center>$$ \begin{eqnarray} \Sigma &=& \mathbb{E}[(x-\mathbb{E}[x])(x-\mathbb{E}[x])^T] \\ &=& \mathbb{E}[(xx^T-x\mathbb{E}[x^T]-\mathbb{E}[x]x^T+\mathbb{E}[x]\mathbb{E}[x]^T)\\ &=& \mathbb{E}[xx^T]-\mathbb{E}[x]\mathbb{E}[x^T]-\mathbb{E}[x]\mathbb{E}[x^T]+\mathbb{E}[x]\mathbb{E}[x^T]\mathbb{E}[1]\\ &=& \mathbb{E}[xx^T]-\mathbb{E}[x]\mathbb{E}[x^T] \end{eqnarray} $$</center>

A multivariate distribution with a zero mean vector and an identity matrix as the covariance matrix is known as the standard normal distribution.

Suppose that we have a training set $$\{(x^{(i)},y^{(i)});i=1,...,m\}$$. We shall make the following assumptions:

<center>$$ \begin{eqnarray} p(y; \phi) &\sim& Bernoulli(\phi)\\ p(x \vert y=0; \mu_0, \Sigma) &\sim& \mathcal{N}(\mu_0,\Sigma)\\ p(x \vert y=1; \mu_1, \Sigma) &\sim& \mathcal{N}(\mu_1,\Sigma) \end{eqnarray} $$</center>

Therefore:

<center>$$ \begin{eqnarray} p(y^{(i)}; \phi) &=& \phi^{y{(i)}}(1-\phi)^{1-y{(i)}} \\ p(x^{(i)} \vert y^{(i)}=0; \mu_0, \Sigma) &=& \frac{1}{(2\pi)^{n/2}\vert\Sigma\vert^{1/2}}exp(-\frac{1}{2}(x^{(i)}-\mu_0)^T\Sigma^{-1}(x^{(i)}-\mu_0)) \\ p(x^{(i)} \vert y^{(i)}=1; \mu_1, \Sigma) &=& \frac{1}{(2\pi)^{n/2}\vert\Sigma\vert^{1/2}}exp(-\frac{1}{2}(x^{(i)}-\mu_1)^T\Sigma^{-1}(x^{(i)}-\mu_1)) \end{eqnarray} $$</center>

We may thus a define log likelihood function for this model:

<center>$$ \begin{eqnarray} \mathcal{l}(\phi,\mu_0,\mu_1,\Sigma) &=& log\left(\prod_{i=1}^{m} p(x^{(i)}, y^{(i)}; \phi,\mu_0,\mu_1,\Sigma)\right)\\ &=& log\left(\prod_{i=1}^{m} p(x^{(i)} \vert y^{(i)}; \mu_0,\mu_1,\Sigma)p(y^{(i)}; \phi)\right) \end{eqnarray} $$</center>

We shall now take the derivative of $$\mathcal{l}$$ with respect to the parameters $$\phi, \mu_0, \mu_1, \Sigma$$, collectively referred to as $$\theta$$:

<center>$$ \begin{eqnarray} \mathcal{l(\theta}) &=& log\left(\prod_{i=1}^{m} p(x^{(i)} \vert y^{(i)}; \mu_0,\mu_1,\Sigma)p(y^{(i)}; \phi)\right)\\ &=& \sum_{i=1}^{m}log\left(p(x^{(i)} \vert y^{(i)}; \mu_0,\mu_1,\Sigma)p(y^{(i)}; \phi)\right)\\ &=& \sum_{i=1}^{m} log\left(p(x^{(i)} \vert y^{(i)}=0)^{1-y^{(i)}}p(x^{(i)} \vert y^{(i)}=1)^{y^{(i)}}\phi^{y^{(i)}}(1-\phi)^{1-y^{(i)}}\right)\\ &=& \sum_{i=1}^{m}(1-y^{(i)})log\left(p(x^{(i)} \vert y^{(i)}=0)\right) + y^{(i)}log\left(p(x^{(i)} \vert y^{(i)}=1)\right) + y^{(i)}log\left(\phi\right) + (1- y^{(i)})log\left(1-\phi\right) \end{eqnarray} $$</center>

Therefore:

<center>$$ \nabla_\phi \mathcal{l}(\theta) = \sum_{i=1}^{m} y^{(i)}\frac{1}{\phi} - (1-y^{(i)})\frac{1}{1-\phi} $$</center>

Setting this to zero gives:

<center>$$ \begin{eqnarray} \sum_{i=1}^m y^{(i)}\left(\frac{1}{\phi} + \frac{1}{1-\phi}\right) - \frac{1}{1-\phi}&=& 0\\ \sum_{i=1}^{m} y^{(i)}\left(\frac{1}{\phi(1-\phi)}\right) &=& m\frac{1}{1-\phi}\\ \phi &=& \frac{1}{m}\sum_{i=1}^m y^{(i)}\\ \phi &=& \frac{1}{m}\sum_{i=1}^m 1\{y^{(i)}=1\} \end{eqnarray} $$</center>

Also:

<center>$$ \begin{eqnarray} \nabla_{\mu_0} \mathcal{l}(\theta) &=& \sum_{i=1}^m (1-y^{(i)})\frac{1}{p(x^{(i)} \vert y^{(i)}=0)}\nabla_{\mu_0} p(x^{(i)} \vert y^{(i)}=0)\\ &=& \sum_{i=1}^m (1-y^{(i)})\frac{1}{p(x^{(i)} \vert y^{(i)}=0)}\nabla_{\mu_0} \left(\frac{1}{(2\pi)^{n/2}\vert\Sigma\vert^{1/2}}exp(-\frac{1}{2}(x^{(i)}- \mu_0)^T\Sigma^{-1}(x^{(i)}-\mu_0))\right) \\ &=& \sum_{i=1}^m (1-y^{(i)})\frac{1}{p(x^{(i)} \vert y^{(i)}=0)}\left(\frac{1} {(2\pi)^{n/2}\vert\Sigma\vert^{1/2}}exp(-\frac{1}{2}(x^{(i)}-\mu_0)^T\Sigma^{-1}(x^{(i)}- \mu_0))\right)\nabla_{\mu_0} \left(-\frac{1}{2}(x^{(i)}-\mu_0)^T\Sigma^{-1}(x^{(i)}- \mu_0)\right)\\ &=& \sum_{i=1}^m (1-y^{(i)}) \nabla_{\mu_0}\left(-\frac{1}{2}(x^{(i)}-\mu_0)^T\Sigma^{-1} (x^{(i)}-\mu_0)\right)\\ &=& \sum_{i=1}^m (1-y^{(i)})\nabla_{\mu_0}\left(-\frac{1} {2}\left((x^{(i)})^T\Sigma^{-1}x^{(i)}-(x^{(i)})^T\Sigma^{-1}\mu_0- \mu_0^T\Sigma^{-1}x^{(i)}+\mu_0^T\Sigma^{-1}\mu_0\right)\right)\\ &=& \sum_{i=1}^m (1-y^{(i)})\left(-\frac{1}{2}\left(-\left((x^{(i)})^T\Sigma^{-1}\right)^T- \Sigma^{-1}x^{(i)}+\Sigma^{-1}\mu_0+\left(\mu_0^T\Sigma^{-1}\right)^T\right)\right)\\ &=& \sum_{i=1}^m (1-y^{(i)})\left(-\frac{1}{2}\left(-\Sigma^{-1}x^{(i)}- \Sigma^{-1}x^{(i)}+\Sigma^{-1}\mu_0+\Sigma^{-1}\mu_0\right)\right)\\ &=& \sum_{i=1}^m(1-y^{(i)})\left(\Sigma^{-1}x^{(i)}-\Sigma^{-1}\mu_0\right) \end{eqnarray} $$</center>

Note that $$\Sigma \geqslant 0$$ is positive semi-definite (and thus symmetric) and so $$\Sigma^T = \Sigma$$ and $$\left(\Sigma^{-1}\right)^T=\left(\Sigma^{-1}\right)$$. We have used these facts in the derivation above. Setting $$\nabla_{\mu_0}\mathcal{l}(\theta)$$ to zero gives:

<center>$$ \begin{eqnarray} \sum_{i=1}^m(1-y^{(i)})\left(\Sigma^{-1}x^{(i)}-\Sigma^{-1}\mu_0\right) &=& 0\\ \sum_{i=1}^m (1-y^{(i)})\left(\Sigma^{-1}\mu_0\right) &=& \sum_{i=1}^m (1- y^{(i)})\left(\Sigma^{-1}x^{(i)}\right)\\ \mu_0 &=& \frac{\sum_{i=1}^m 1\{y^{(i)}=0\}x^{(i)}}{\sum_{i=1}^m 1\{y^{(i)}=0\}} \end{eqnarray} $$</center>

Similarly, it can be shown that:

<center>$$ \mu_1 = \frac{\sum_{i=1}^m 1\{y^{(i)}=1\}x^{(i)}}{\sum_{i=1}^m 1\{y^{(i)}=1\}} $$</center>

For $$\nabla_\Sigma \mathcal{l}(\theta)$$ we have:

<center>$$ \nabla_\Sigma \mathcal{l}(\theta) = \nabla_\Sigma \sum_{i=1} ^m(1-y^{(i)})log\left(p(x^{(i)} \vert y^{(i)}=0)\right) + y^{(i)}log\left(p(x^{(i)} \vert y^{(i)}=1)\right) $$</center>

We have:

<center>$$ \begin{eqnarray} \nabla_\Sigma log\left(p(x^{(i)} \vert y^{(i)}=0)\right) &=& \frac{1}{p(x^{(i)} \vert y^{(i)}=0)}\nabla_\Sigma \left(\frac{1}{(2\pi)^{n/2}\vert\Sigma\vert^{1/2}}exp\left(- \frac{1}{2}(x^{(i)}-\mu_0)^T\Sigma^{-1}(x^{(i)}-\mu_0)\right)\right)\\ &=& \frac{1}{p(x^{(i)} \vert y^{(i)}=0)}\left(\nabla_\Sigma\left(\frac{1} {(2\pi)^{n/2}\vert\Sigma\vert^{1/2}}\right)exp\left(-\frac{1}{2}(x^{(i)}-\mu_0)^T\Sigma^{-1} (x^{(i)}-\mu_0)\right) + \left(\frac{1} {(2\pi)^{n/2}\vert\Sigma\vert^{1/2}}\right)\nabla_\Sigma \left(exp\left(-\frac{1}{2} (x^{(i)}-\mu_0)^T\Sigma^{-1}(x^{(i)}-\mu_0)\right)\right)\right)\\ &=& \left((2\pi)^{n/2}\vert\Sigma\vert^{1/2}\right)\nabla_\Sigma \left(\frac{1} {(2\pi)^{n/2}\vert\Sigma\vert^{1/2}}\right) + \nabla_\Sigma \left(-\frac{1}{2}(x^{(i)}- \mu_0)^T\Sigma^{-1}(x^{(i)}-\mu_0)\right)\\ &=& \left((2\pi)^{n/2}\vert\Sigma\vert^{1/2}\right)\left(-\frac{\frac{1}{2}\vert \Sigma \vert^{-1/2}}{(2\pi)^{n/2}(\vert\Sigma\vert^{1/2})^2}\right)\nabla_\Sigma\vert\Sigma\vert - \frac{1}{2}\left(-\Sigma^{-T}(x^{(i)}-\mu_0)(x^{(i)}-\mu_0)^T\Sigma^{-T}\right)\\ &=& -\frac{1}{2\vert\Sigma\vert}\left(\vert\Sigma\vert\Sigma^{-T}\right)+\frac{1} {2}\left(\Sigma^{-T}(x^{(i)}-\mu_0)(x^{(i)}-\mu_0)^T\Sigma^{-T}\right)\\ &=& -\frac{1}{2}\left(\Sigma^{-T}-\Sigma^{-T}(x^{(i)}-\mu_0)(x^{(i)}-\mu_0)^T\Sigma^{- T}\right) \end{eqnarray} $$</center>

Similarly we have:

<center>$$ \nabla_\Sigma log\left(p(x^{(i)} \vert y^{(i)}=1)\right)=-\frac{1}{2}\left(\Sigma^{- T}-\Sigma^{-T}(x^{(i)}-\mu_1)(x^{(i)}-\mu_1)^T\Sigma^{-T}\right) $$</center>

Thus:

<center>$$ \begin{eqnarray} \nabla_\Sigma \mathcal{l}(\theta) &=& -\frac{1}{2}\sum_{i=1}^m (1-y^{(i)})\left(\Sigma^{- T}-\Sigma^{-T}(x^{(i)}-\mu_0)(x^{(i)}-\mu_0)^T\Sigma^{-T}\right)+y^{(i)}\left(\Sigma^{- T}-\Sigma^{-T}(x^{(i)}-\mu_1)(x^{(i)}-\mu_1)^T\Sigma^{-T}\right)\\ &=& -\frac{1}{2} \sum_{i=1}^m \Sigma^{-T}-\Sigma^{-T}(x^{(i)}-\mu_{y^{(i)}})(x^{(i)}- \mu_{y^{(i)}})^T\Sigma^{-T} \end{eqnarray} $$</center>

Setting this to zero gives:

<center>$$ \begin{eqnarray} \sum_{i=1}^m \Sigma^{-T}-\Sigma^{-T}(x^{(i)}-\mu_{y^{(i)}})(x^{(i)}- \mu_{y^{(i)}})^T\Sigma^{-T} &=& 0 \\ \sum_{i=1}^m \Sigma^{-T} &=& \sum_{i=1}^m\Sigma^{-T}(x^{(i)}-\mu_{y^{(i)}})(x^{(i)}- \mu_{y^{(i)}})^T\Sigma^{-T} \\ mI &=& \sum_{i=1}^m I(x^{(i)}-\mu_{y^{(i)}})(x^{(i)}-\mu_{y^{(i)}})^T\Sigma^{-T} \\ \Sigma^T &=& \frac{1}{m} \sum_{i=1}^m (x^{(i)}-\mu_{y^{(i)}})(x^{(i)}-\mu_{y^{(i)}})^T \\ \Sigma &=& \frac{1}{m} \sum_{i=1}^m (x^{(i)}-\mu_{y^{(i)}})(x^{(i)}-\mu_{y^{(i)}})^T \end{eqnarray} $$</center>

It can be shown that:

<center>$$ p(y=1 \vert x; \phi, \mu_0, \mu_1, \Sigma) = \frac{1}{1+exp(-\theta^Tx)} $$</center>

where $$\theta$$ is some appropriate function of $$\phi, \mu_0, \mu_1, \Sigma$$. This is exactly the form that the logistic regression takes. A comparison between the two methods reveals that the Gaussian Discriminant Analysis makes stronger assumptions as compared to logistic regression. Hence, the logistic regression is more robust and less sensitive to incorrect modelling assumptions. However, when $$p(x \vert y)$$ is Gaussian (with shared $$\Sigma$$) it can be shown that the Gaussian Discriminant Analysis is asymptotically efficient, i.e. there is no algorithm that can perform strictly better than it in the limits of very large training sets.

## 2\. Naïve Bayes

Suppose that in our training set $$\{x^{(i)},y^{(i)}\}$$, $$x^{(i)} \in \mathbb{R}^n$$ such that each $$x^{(i)}_j \in \{0,1\}$$ and $$y^{(i)} \in \{0,1\}$$. Let us make a rather strong assumption that the $$x^{(i)}_j$$’s are conditionally independent given $$y^{(i)}$$, i.e. $$p(x^{(i)}_j \vert x^{(i)}_k, y^{(i)} )$$ $$=$$ $$p(x^{(i)}_j \vert y^{(i)} )$$. This is known as the Naïve Bayes assumption. Therefore:

<center>$$ \begin{eqnarray} p(x^{(i)} \vert y^{(i)}) &=& p(x^{(i)}_{n-1},x^{(i)}_{n-2},...,x^{(i)}_1 \vert y^{(i)})\\ &=& p(x^{(i)}_{n-1} \vert x^{(i)}_{n-2},...,x^{(i)}_1, y^{(i)})p(x^{(i)}_{n-2} \vert x^{(i)}_{n-3},...,x^{(i)}_1, y^{(i)})...p(x^{(i)}_1 \vert y^{(i)})\\ &=& p(x^{(i)}_{n-1} \vert y^{(i)})p(x^{(i)}_{n-2} \vert y^{(i)})...p(x^{(i)}_1 \vert y^{(i)})\\ &=& \prod_{j=1}^{n} p(x^{(i)}_{j} \vert y^{(i)}) \end{eqnarray} $$</center>

Let $$\phi_{i \vert y=1} = p(x^{(k)}_i=1 \vert y^{(k)}=1)$$, $$\phi_{i \vert y=0} = p(x^{(k)}_i=1 \vert y^{(k)}=0)$$ and $$\phi_y = p(y^{(i)}=1)$$ be the parameters of our model. We may then define the log likelihood $$\mathcal{l}(\phi_{i \vert y=1}, \phi_{i \vert y=0}, \phi_y)$$ function as follows:

<center>$$ \begin{eqnarray} \mathcal{l}(\phi_{i \vert y=1}, \phi_{i \vert y=0}, \phi_y) &=& log\left(\prod_{i=1}^m p(x^{(i)}, y^{(i)})\right)\\ &=& log\left(\prod_{i=1}^m p(x^{(i)} \vert y^{(i)})p(y^{(i)})\right)\\ &=& log\left(\prod_{i=1}^m \left(\prod_{j=1}^{n} p(x^{(i)}_j \vert y^{(i)}; \phi_{j \vert y=1}, \phi_{j \vert y=0})\right)p(y^{(i)}; \phi_y)\right) \end{eqnarray} $$</center>

The distribution $$p(y \vert x)$$ as described above is the multivariate Bernoulli distribution. Note that:

<center>$$ \begin{eqnarray} \mathcal{l}(\theta) &=& log\left(\prod_{i=1}^m \left(\prod_{j=1}^{n} \phi_{j \vert y=1}^{x^{(i)}_j y^{(i)}} (1-\phi_{j \vert y=1})^{(1-x^{(i)}_j)y^{(i)}}\phi_{j \vert y=0}^{x^{(i)}_j(1-y^{(i)})}(1-\phi_{j \vert y=0})^{(1-x^{(i)}_j)(1- y^{(i)})}\right)\phi_y^{y^{(i)}}(1-\phi_y)^{1-y^{(i)}}\right)\\ &=& \sum_{i=1}^m \left(\sum_{j=1}^{n} x^{(i)}_j y^{(i)}log(\phi_{j \vert y=1}) + (1- x^{(i)}_j)y^{(i)}log(1-\phi_{j \vert y=1})+x^{(i)}_j(1-y^{(i)})log(\phi_{j \vert y=0})+ (1- x^{(i)}_j)(1-y^{(i)})log(1-\phi_{j \vert y=0})\right)+y^{(i)}log(\phi_y)+(1-y^{(i)})log(1- \phi_y) \end{eqnarray} $$</center>

where the parameters of the model are collectively referred to as $$\theta$$. Let us now maximize the log likelihood with respect to each of the parameters in our model:

<center>$$ \begin{eqnarray} \nabla_{\phi_y} \mathcal{l}(\theta) &=& \nabla_{\phi_y} \sum_{i=1}^m y^{(i)}log(\phi_y)+(1- y^{(i)})log(1-\phi_y)\\ &=& \sum_{i=1}^m y^{(i)}\frac{1}{\phi_y}-(1-y^{(i)})\frac{1}{1-\phi_y}\\ &=& \sum_{i=1}^m \frac{y^{(i)}-\phi_y}{\phi_y(1-\phi_y)} \end{eqnarray} $$</center>

Setting this to zero gives:

<center>$$ \begin{eqnarray} \sum_{i=1}^m \frac{y^{(i)}-\phi_y}{\phi_y(1-\phi_y)} &=& 0\\ \phi_y &=& \frac{1}{m}\sum_{i=1}^m 1\{y^{(i)}=1\} \end{eqnarray} $$</center>

Also:

<center>$$ \begin{eqnarray} \nabla_{\phi_{k\vert y=0}} \mathcal{l}(\theta) &=& \nabla_{\phi_{k\vert y=0}}\left(\sum_{i=1}^m \sum_{j=1}^{n} x^{(i)}_j(1-y^{(i)})log(\phi_{j \vert y=0})+(1- x^{(i)}_j)(1-y^{(i)})log(1-\phi_{j \vert y=0})\right)\\ &=& \sum_{i=1}^m x^{(i)}_k(1-y^{(i)})\frac{1}{\phi_{k \vert y=0}}-(1-x^{(i)}_k)(1- y^{(i)})\frac{1}{1-\phi_{k \vert y=0}}\\ &=& \sum_{i=1}^m (1-y^{(i)})\left(\frac{x^{(i)}_k-\phi_{k \vert y=0}}{\phi_{k \vert y=0}(1- \phi_{k \vert y=0})}\right) \end{eqnarray} $$</center>

Setting this to zero gives:

<center>$$ \begin{eqnarray} \sum_{i=1}^m (1-y^{(i)})\left(\frac{x^{(i)}_k-\phi_{k \vert y=0}}{\phi_{k \vert y=0}(1- \phi_{k \vert y=0})}\right) &=& 0\\ \sum_{i=1}^m (1-y^{(i)})x^{(i)}_k &=& \sum_{i=1}^m (1-y^{(i)})\phi_{k \vert y=0}\\ \phi_{k \vert y=0} &=& \frac{\sum_{i=1}^m 1\{x^{(i)}_k=1 \wedge y^{(i)}=0\}}{\sum_{i=1}^m 1 \{y^{(i)}=0\}} \end{eqnarray} $$</center>

where the symbol $$\wedge$$ means ‘and’. Similarly, it can be shown that:

<center>$$ \phi_{k \vert y=1} = \frac{\sum_{i=1}^m 1\{x^{(i)}_k=1 \wedge y^{(i)}=1\}}{\sum_{i=1}^m 1 \{y^{(i)}=1\}} $$</center>

Once the distribution $$p(x \vert y)$$ has been modelled we may find $$p(y=1 \vert x)$$ using Bayes’ rule:

<center>$$ \begin{eqnarray} p(y=1 \vert x) &=& \frac{p(x \vert y=1)p(y=1)}{p(x)}\\ &=& \frac{\prod_{i=1}^n p(x_i \vert y=1)p(y=1)}{\prod_{i=1}^n \left(p(x_i \vert y=1)p(y=1)+p(x_i \vert y=0)p(y=0)\right)} \end{eqnarray} $$</center>

### Laplace Smoothing

Suppose that in a particular training set $$x^{(i)}_j$$ is never $$1$$ for either $$y^{(i)}=0$$ or $$y^{(i)}=1$$ for some index $$j$$. Therefore, we will have $$\phi_{j \vert y=1} = 0$$ and $$\phi_{j \vert y=1} = 0$$. This means that the model will output $$p(y=1 \vert x) = \frac{0}{0}$$.

More broadly, suppose that a random variable $$z \in \{1,..,k\}$$. We may model a multinomial distribution for $$z$$ that is parameterized by $$\phi_j = p(z=j)$$. Maximizing the likelihood function will yield:

<center>$$ \phi_j = \frac{\sum_{i=1}^{m} 1\{z^{(i)}=j\}}{m} $$</center>

However, if $$z$$ is never equal to, say, $$d$$ then $$\phi_d = 0$$, which will result in the model outputting $$p(z) = \frac{0}{0}$$. To solve this problem we introduce Laplace Smoothing. The parameter $$\phi_j$$ is then given by:

<center>$$ \phi_j = \frac{\sum_{i=1}^{m} 1\{z^{(i)}=j\}+1}{m+k} $$</center>

Note that:

<center>$$ \begin{eqnarray} \sum_{i=1}^k \phi_j &=& \sum_{i=1}^k \frac{\sum_{i=1}^{m} 1\{z^{(i)}=j\}+1}{m+k}\\ &=& \frac{\frac{\sum_{i=1}^{m} 1\{z^{(i)}=j\}}{m}}{1+\frac{k}{m}}+\frac{\sum_{i=1}^k(1)} {m+k}\\ &=& \frac{1}{1+\frac{k}{m}}+\frac{k}{m+k}\\ &=& \frac{m}{m+k}+\frac{k}{m+k}\\ &=& 1 \end{eqnarray} $$</center>

We may also extend this to our Naïve Bayes’ model:

<center>$$ \begin{eqnarray} \phi_{k \vert y=0} &=& \frac{\sum_{i=1}^m 1\{x^{(i)}_k=1 \wedge y^{(i)}=0\}+1}{\sum_{i=1}^m 1\{y^{(i)}=0\}+2}\\ \phi_{k \vert y=1} &=& \frac{\sum_{i=1}^m 1\{x^{(i)}_k=1 \wedge y^{(i)}=1\}+1}{\sum_{i=1}^m 1 \{y^{(i)}=1\}+2} \end{eqnarray} $$</center>

### Event Models for Text Classification

Let us describe the multinomial event model. Suppose that in our training set the inputs are texts, each consisting of $$n$$ words (here $$n$$ may be different for different texts). Each word belongs to a vocabulary V of size $$\vert V \vert$$. Let us represent the feature vector $$x^{(i)}$$ for a particular text as $$(x^{(i)}_1, x^{(i)}_2,...,x^{(i)}_n)$$ where $$x^{(i)}_j$$ represents the identity of the $$j$$‘th word of the text in the vocabulary. Our goal is to classify each text in one of the two categories, i.e. $$y \in \{0,1\}$$

Let $$\phi_{i \vert y=0} = p(x^{(i)}_j=i \vert y=0)$$, $$\phi_{i \vert y=1} = p(x^{(i)}_j=i \vert y=1)$$ and $$\phi_y = p(y^{(i)}=1)$$ be the parameters of our model. Assuming that the Naïve Bayes assumption holds, we may then define the log likelihood function $$\mathcal{l}(\theta)$$, where the parameters of the model are collectively referred to as $$\theta$$, as follows:

<center>$$ \begin{eqnarray} \mathcal{l}(\theta) &=& log\left(\prod_{i=1}^m \left( \prod_{j=1}^{n_i} p(x^{(i)}_j \vert y^{(i)};\phi_{i \vert y=0},\phi_{i \vert y=1})\right)p(y;\phi_y)\right)\\ &=& log\left(\prod_{i=1}^m \left( \prod_{j=1}^{n_i}\left( \prod_{k=1}^{\vert V \vert}\phi_{k \vert y=0}^{1\{y^{(i)}=0\}1\{x^{(i)}_j=k\}}(1-\phi_{k \vert y=0})^{1\{y^{(i)}=0\}(1- 1\{x^{(i)}_j=k\})}\phi_{k \vert y=1}^{1\{y^{(i)}=1\}1\{x^{(i)}_j=k\}}(1-\phi_{k \vert y=1})^{1\{y^{(i)}=1\}(1-1\{x^{(i)}_j=k\})}\right)\right)\phi_y^{y^{(i)}}(1-\phi_y)^{1- y^{(i)}}\right)\\ &=& \sum_{i=1}^m \left( \sum_{j=1}^{n_i}\left( \sum_{k=1}^{\vert V \vert}1\{y^{(i)}=0\} 1\{x^{(i)}_j=k\}log(\phi_{k \vert y=0})+1\{y^{(i)}=0\}(1-1\{x^{(i)}_j=k\})log(1-\phi_{k \vert y=0})+1\{y^{(i)}=1\}1\{x^{(i)}_j=k\}log(\phi_{k \vert y=1})+1\{y^{(i)}=1\}(1- 1\{x^{(i)}_j=k\})log(1-\phi_{k \vert y=1})+\right)\right)+y^{(i)}log(\phi_y)+(1- y^{(i)})log(1- \phi_y)\\ \end{eqnarray} $$</center>

Therefore:

<center>$$ \begin{eqnarray} \nabla_{\phi_y} &=& \frac{\sum_{i=1}^m 1\{y^{i}=1\}}{m}\\ \nabla_{\phi_l \vert y=0} &=& \frac{\sum_{i=1}^m\sum_{j=1}^{n_i}1\{x^{(i)}_j=l\} \wedge 1\{y^{(i)}=0\}}{\sum_{i=1}^m 1\{y^{(i)}=0\}n_i}\\ \nabla_{\phi_l \vert y=1} &=& \frac{\sum_{i=1}^m\sum_{j=1}^{n_i}1\{x^{(i)}_j=l\} \wedge 1\{y^{(i)}=1\}}{\sum_{i=1}^m 1\{y^{(i)}=1\}n_i} \end{eqnarray} $$</center>

Using Laplace Smoothing yields:

<center>$$ \begin{eqnarray} \nabla_{\phi_l \vert y=0} &=& \frac{\sum_{i=1}^m\sum_{j=1}^{n_i}1\{x^{(i)}_j=l\} \wedge 1\{y^{(i)}=0\}+1}{\sum_{i=1}^m 1\{y^{(i)}=0\}n_i+\vert V \vert}\\ \nabla_{\phi_l \vert y=1} &=& \frac{\sum_{i=1}^m\sum_{j=1}^{n_i}1\{x^{(i)}_j=l\} \wedge 1\{y^{(i)}=1\}+1}{\sum_{i=1}^m 1\{y^{(i)}=1\}n_i+\vert V \vert} \end{eqnarray} $$</center>
