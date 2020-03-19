---
layout: post
title: Factor Analysis
permalink: blog/machine-learning/cs229-notes/factor-analysis
categories: [Machine Learning, CS229 Notes]
---

Let $$S = \{x^{(1)},x^{(2)},...,x^{(m)}\}$$ be a training set where $$x \in \mathbb{R}^n$$ such that $$n>m$$. Suppose that we wish to model $$S$$ using a Gaussian distribution. Note that the covariance matrix, $$\Sigma$$ $$\in$$ $$\mathbb{R}^{n \times n}$$, is given by:

<center>$$ \Sigma = \frac{1}{m}\sum_{i=1}^m (x^{(i)}-\mu)(x^{(i)}-\mu)^T $$</center>

Note that $$(x^{(i)}-\mu)(x^{(i)}-\mu)^T$$ is essentially:

<center>$$ \begin{bmatrix} (x^{(i)}-\mu)_1(x^{(i)}-\mu)^T \\ (x^{(i)}-\mu)_2(x^{(i)}-\mu)^T \\ .\\ .\\ .\\ (x^{(i)}-\mu)_n(x^{(i)}-\mu)^T \end{bmatrix} $$</center>

where $$(x^{(i)}-\mu)_i$$ is the $$i^{th}$$ element of the vector $$(x^{(i)}-\mu)$$. Note that each row of the matrix above is equal to $$(x^{(i)}-\mu)^T$$ multiplied by some scalar number. Hence, the rank of this matrix is $$1$$. The covariance matrix is thus a summation of $$m$$ matrices, each of rank $$1$$. Note that the $$m^{th}$$ matrix is completely determined by the other $$m-1$$ matrices. This can be shown in the following way:

<center>$$ \begin{eqnarray} \mu &=& \frac{1}{m}\sum_{i=1}^mx^{(i)}\\ x^{(m)} &=& m\mu - \sum_{i=1}^{m-1}x^{(i)} \end{eqnarray} $$</center>

Therefore:

<center>$$ \Sigma = \frac{1}{m}\left(\sum_{i=1}^{m-1}(x^{(i)}-\mu)(x^{(i)}-\mu)^T + (x^{(m)}-\mu)(x^{(m)}-\mu)^T\right) $$</center>

where $$x^{(m)}=m\mu - \sum_{i=1}^{m-1}x^{(i)}$$. Therefore the covariance matrix is a sum of $$m-1$$ linearly independent matrices of rank $$1$$ and another matrix (of rank $$1$$) that can be completely determined by the other $$m-1$$ matrices. We know that the rank of the sum of $$m-1$$ linearly independent matrices is at most the sum of the rank of the $$m-1$$ matrices (see [this]({{site.baseurl}}{%post_url /blog/mathematics/2018-10-22-proofs-of-theorems-on-ranks%})). Therefore, the rank of the covariance matrix, in this case, is at most $$m-1$$. However, because $$n>m$$, the covariance matrix (of size $$n \times n$$) does not have full rank (i.e. not all rows/columns of $$\Sigma$$ are linearly independent). Hence, $$\Sigma$$ is singular, or more specifically $$\vert \Sigma \vert = 0$$. Therefore, $$\Sigma^{-1}$$ does not exist.

The probability density of a random variable $$X$$ that has a Gaussian distribution is given by:

<center>$$ p(X=x) = \frac{1}{(2\pi)^{n/2}\vert\Sigma\vert^{1/2}}exp(-\frac{1}{2}(x-\mu)^T\Sigma^{-1} (x-mu)) $$</center>

However, because $$\Sigma^{-1}$$ does not exist and $$\frac{1}{\vert\Sigma\vert^{1/2}}$$ cannot be evaluated, the probability $$p(X=x)$$ is not defined.

## 1. Restrictions on \\(\Sigma\\)

In order to make sure that the covariance matrix, $$\Sigma$$, is not singular we may put one of the following restrictions on it:

1.  $$\Sigma$$ should be diagonal matrix, where $$\Sigma_{jj} = \frac{1}{m}\sum_{i=1}^m (x^{(i)}_j-\mu_j)^2$$ for $$j=1,...,n$$.
2.  $$\Sigma$$ should be equal to $$\sigma^2I$$, where $$I$$ is an $$n\times n$$ identity matrix and $$\sigma=\frac{1}{mn}\sum_{i=1}^m\sum_{j=1}^n (x^{(i)}_j-\mu_j)^2$$.

Note that while we may obtain a non-singular covariance matrix under either of the two conditions above for $$m \geqslant 2$$ (m=1 would mean that the term $$(x_j-\mu_j)$$ is just $$0$$ because the mean would equal the only example in our training set), restricting $$\Sigma$$ to a diagonal matrix would imply independence and the absence of correlation between different dimensions of $$x$$, which is usually not the case. Below, we describe the factor analysis model that tries to avoid this problem.

## 2. The Factor Analysis Model

Let $$S = \{x^{(1)},...,x^{(m)}\}$$, where $$x^{(i)} \in \mathbb{R}^{n}$$, be a set of training examples. The factor analysis model models the following distribution (in the ensuing discussion we will drop the superscript $$(i)$$ on $$x$$ for conciseness when there is no fear of ambiguity):

<center>$$ \begin{eqnarray} z &\sim& \mathcal{N}(0,I)\\ \epsilon &\sim& \mathcal{N}(0,\Psi)\\ x &=& \mu + \Lambda z + \epsilon \end{eqnarray} $$</center>

where $$z \in \mathbb{R}^k$$, $$\epsilon \in R^{n}$$ are random variables and $$\Lambda \in \mathbb{R}^{n \times k}$$, $$\mu \in \mathbb{R}^n$$ and $$\Psi \in \mathbb{R}^{n \times n}$$.

Consider the equation for $$x$$. Suppose that we hold $$z$$ constant. Then:

<center>$$ \begin{eqnarray} \mathbb{E}[x \vert z] &=& \mathbb{E}[\mu + \Lambda z + \epsilon]\\ &=& \mathbb{E}[\mu + \Lambda z] + \mathbb{E}[\epsilon]\\ &=& \mu + \Lambda z \end{eqnarray} $$</center>

Note that $$\mathbb{E}[x \vert z]$$ indicates that $$z$$ has been fixed to some value. Also note that the last step follows from the fact that $$z$$ is fixed and so the term $$\mu + \Lambda z$$ is constant (and the expectation of a constant is just that constant). Also:

<center>$$ \begin{eqnarray} Cov(x \vert z) &=& Cov(\mu + \Lambda z + \epsilon)\\ &=& Cov(\mu + \Lambda z) + Cov(\epsilon)\\ &=& \Psi \end{eqnarray} $$</center>

Note that the last step follows from the fact that the covariance of a constant is just zero. (To see this consider: $$Cov(a) = \mathbb{E}[(a-\mathbb{E}[a])(a-\mathbb{E}[a])^T]$$ $$=\mathbb{E}[(a-a)(a-a)^T]=0$$, where $$a$$ is a constant).

Therefore, the factor analysis model can equivalently be written as:

<center>$$ \begin{eqnarray} z &\sim& \mathcal{N}(0,I)\\ x \vert z &\sim& \mathcal{N}(\mu + \Lambda z, \Psi) \end{eqnarray} $$</center>

However, in deriving the factor analysis model we will use the initial formulation of the model. Let us define the joint distribution for $$x$$ and $$z$$ as follows:

<center>$$ \begin{bmatrix} z \\ x \end{bmatrix} \sim \mathcal{N}(\mu_{zx}, \Sigma) $$</center>

where $$\mu_{zx} \in \mathbb{R}^{n+k}$$, $$\Sigma \in \mathbb{R}^{(n+k) \times (n+k)}$$. Note that:

<center>$$ \begin{eqnarray} \mathbb{E}[x] &=& \mathbb{E}[\mu + \lambda z + \epsilon]\\ &=& \mu \end{eqnarray} $$</center>

Note that $$z$$ is no longer fixed. Therefore:

<center>$$ \mu_{zx} = \begin{bmatrix} \vec{0} \\ \mu\end{bmatrix} $$</center>

Consider the following:

<center>$$ \begin{eqnarray} Cov(z) &=& \mathbb{E}[(z-\mathbb{E}[z])(z-\mathbb{E}[z])^T]\\ &=& \mathbb{E}[zz^T] \end{eqnarray} $$</center>

Similarly $$Cov(\epsilon)=\mathbb{E}[\epsilon\epsilon^T]$$. In order to derive the joint covariance matrix $$\Sigma$$ we would need $$\Sigma_{zz}$$, $$\Sigma_{zx}$$, $$\Sigma_{xz}$$ and $$\Sigma_{zz}$$ (see [this]({{site.baseurl}}{%post_url /blog/mathematics/2018-10-30-the-multivariate-distribution%})). Clearly $$\Sigma_{zz}$$ $$= I​$$. Also:

<center>$$ \begin{eqnarray} \Sigma_{zx} &=& \mathbb{E}[(z-\mathbb{E}[z])(x-\mathbb{E}[x])^T]\\ &=& \mathbb{E}[(z-\mathbb{E}[z])(\mu+\Lambda z + \epsilon - \mu)^T]\\ &=& \mathbb{E}[(z)(\Lambda z + \epsilon)^T]\\ &=& \mathbb{E}[(zz^T\Lambda^T + z\epsilon^T]\\ &=& \mathbb{E}[zz^T]\Lambda^T + \mathbb{E}[z]\mathbb{E}[\epsilon^T]\\ &=& Cov(z)\Lambda^T\\ &=& \Lambda^T \end{eqnarray} $$</center>

The third-to-last step used the fact that $$z$$ and $$\epsilon$$ are independent and so $$\mathbb{E}[z\epsilon]=\mathbb{E}[z]\mathbb{E}[\epsilon]$$. Similarly, we can show that $$\Sigma_{xz}=\Lambda$$. Lastly:

<center>$$ \begin{eqnarray} \Sigma_{xx} &=& \mathbb{E}[(x-\mathbb{E}[x])(x-\mathbb{E}[x])^T]\\ &=& \mathbb{E}[(\Lambda z + \epsilon)(\Lambda z + \epsilon)^T]\\ &=& \mathbb{E}[\Lambda zz^T \Lambda^T + \Lambda z\epsilon^T + \epsilon z^T \Lambda + \epsilon \epsilon^T]\\ &=& \Lambda \Lambda^T + \Psi \end{eqnarray} $$</center>

Hence:

<center>$$ \begin{bmatrix} z \\ x \end{bmatrix} \sim \mathcal{N}\left(\begin{bmatrix} \vec{0} \\ \mu \end{bmatrix}, \begin{bmatrix} I & \Lambda^T \\ \Lambda & \Lambda\Lambda^T + \Psi \end{bmatrix}\right) $$</center>

From this it follows that $$x \sim \mathcal{N}(\mu, \Lambda\Lambda^T+\Psi)$$. Therefore the log likelihood is given by:

<center>$$ l(\mu,\Lambda,\Psi) = log\prod_{i=1}^m \frac{1}{(2\pi)^{n/2}\vert \Lambda\Lambda^T + \Psi \vert^{1/2}}exp\left(-\frac{1}{2}(x^{(i)}-\mu)^T(\Lambda\Lambda^T + \Psi)^{-1}(x^{(i)}-\mu)\right) $$</center>

We shall use the [Expectation-Maximization Algorithm]({{site.baseurl}}{%post_url /blog/machine-learning/cs229-notes/2018-10-18-expectation-maximization-algorithm%}) to find optimal values for the parameters of this model.

For the E-step we need to set the value of $$Q_i(z^{(i)})$$ to $$p(z^{(i)} \vert x^{(i)})$$. Note that $$Q_i(z^{(i)}) \sim \mathcal{N}(\mu_{z^{(i)} \vert x^{(i)}},$$ $$\Sigma_{z^{(i)} \vert x^{(i)}})$$ where $$\mu_{z^{(i)} \vert x^{(i)}}$$ and $$\Sigma_{z^{(i)} \vert x^{(i)}}$$ are given by (see [this]({{site.baseurl}}{%post_url /blog/mathematics/2018-10-30-the-multivariate-distribution%}) for why this is so):

<center>$$ \begin{eqnarray} \mu_{z^{(i)} \vert x^{(i)}} &=& \mu_{z^{(i)}} + \Sigma_{z^{(i)}x^{(i)}}\Sigma_{x^{(i)}x^{(i)}}^{-1}(x^{(i)}-\mu_{x^{(i)}})\\ &=&\Lambda^T(\Lambda\Lambda^T+\Psi)^{-1}(x^{(i)}-\mu)\\ \Sigma_{z^{(i)} \vert x^{(i)}} &=& \Sigma_{z^{(i)}z^{(i)}}-\Sigma_{z^{(i)}x^{(i)}}\Sigma_{x^{(i)}x^{(i)}}^{-1}\Sigma_{x^{(i)}z^{(i)}}\\ &=&I - \Lambda^T(\Lambda\Lambda^T+\Psi)^{-1}\Lambda \end{eqnarray} $$</center>

For the M-step we need to maximize the following:

<center>$$ \begin{eqnarray} f(\mu,\Lambda,\Psi) &=& \sum_{i=1}^m\int_{z^{(i)}}Q_i(z^{(i)})log\left(\frac{p(x^{(i)},z^{(i)})}{Q_i(z^{(i)})}\right)\\ &=& \sum_{i=1}^m\int_{z^{(i)}}Q_i(z^{(i)})log\left(\frac{p(x^{(i)} \vert z^{(i)})p(z^{(i)})}{Q_i(z^{(i)})}\right)\\ &=& \sum_{i=1}^m\int_{z^{(i)}}Q_i(z^{(i)})\left(log\left(p(x^{(i)} \vert z^{(i)})\right)+log\left(p(z^{(i)})\right)-log\left({Q_i(z^{(i)})}\right)\right) \end{eqnarray} $$</center>

Dropping all terms that do not depend on $$\mu$$, $$\Lambda$$ and $$\Psi$$ gives (note that $$Q_i(z^{(i)})$$ is just some number because it was explicitly evaluated during the E-step):

<center>$$ \begin{eqnarray} f(\mu,\Lambda,\Psi)&=&\sum_{i=1}^m\int_{z^{(i)}}Q_i(z^{(i)})log\left(p(x^{(i)} \vert z^{(i)})\right)\\ &=& \sum_{i=1}^m \mathbb{E}_{z^{(i)} \sim Q_i}\left[log\left(p(x^{(i)}\vert z^{(i)})\right)\right]\\ &=& \sum_{i=1}^m \mathbb{E}_{z^{(i)}\sim Q_i}\left[log\left(\frac{1}{(2\pi)^{n/2}\vert\Psi\vert^{1/2}}exp\left(-\frac{1}{2}(x^{(i)}-\mu-\Lambda z^{(i)})^T\Psi^{-1}(x^{(i)}-\mu-\Lambda z^{(i)})\right)\right)\right]\\ &=& \sum_{i=1}^m \mathbb{E}_{z^{(i)}\sim Q_i}\left[log\left(\frac{1}{(2\pi)^{n/2}}\right) - \frac{1}{2}log\vert \Psi \vert -\frac{1}{2}(x^{(i)}-\mu-\Lambda z^{(i)})^T\Psi^{-1}(x^{(i)}-\mu-\Lambda z^{(i)})\right]\\ &=& \sum_{i=1}^m \mathbb{E}_{z^{(i)}\sim Q_i}\left[- \frac{1}{2}log\vert \Psi \vert - \frac{1}{2}(x^{(i)}-\mu-\Lambda z^{(i)})^T\Psi^{-1}(x^{(i)}-\mu-\Lambda z^{(i)})\right]\\ \end{eqnarray} $$</center>

In the last step above, we have dropped all terms that do not depend on $$\mu$$, $$\Lambda$$ and $$\Psi$$. Note that $$z^{(i)} \sim Q_i$$ means that $$z^{(i)}$$ is being sampled from the probability distribution $$Q_i$$, which in this case is just equal to the distribution of $$z^{(i)}$$ conditioned on $$x^{(i)}$$ (i.e. $$z^{(i)} \sim \mathcal{N}(\mu_{z^{(i)} \vert x^{(i)}},\Sigma_{z^{(i)} \vert x^{(i)}})$$).

We shall now maximize $$f$$ with respect to $$\Lambda$$ by only considering the terms that depend on $$\Lambda$$. We will be making use of the properties of a trace of a matrix (see [this]({{site.baseurl}}{% post_url blog/mathematics/2018-08-15-matrix-derivatives%})). We will also drop the subscript in $$\mathbb{E}_{z^{(i)}\sim Q_i}$$ for convenience.

<center>$$ \begin{eqnarray} \nabla_\Lambda f(\mu,\Lambda,\Psi) &=& -\frac{1}{2}\sum_{i=1}^m \mathbb{E}\left[ \nabla_\Lambda\left((z^{(i)})^T\Lambda^T\Psi^{-1}\Lambda z^{(i)}-(z^{(i)})^T\Lambda^T\Psi^{-1}(x^{(i)}-\mu)-(x^{(i)}-\mu)^T\Psi^{-1}\Lambda z^{(i)}\right)\right]\\ &=& -\frac{1}{2}\sum_{i=1}^m \mathbb{E}\left[\nabla_\Lambda tr\left((z^{(i)})^T\Lambda^T\Psi^{-1}\Lambda z^{(i)}\right)-2\nabla_\Lambda tr\left((z^{(i)})^T\Lambda^T\Psi^{-1}(x^{(i)}-\mu)\right)\right]\\ &=& -\frac{1}{2}\sum_{i=1}^m \mathbb{E}\left[\nabla_{\Lambda^T} tr\left(\Lambda\Psi^{-1}\Lambda^T z^{(i)}(z^{(i)})^T\right)-2\nabla_{\Lambda^T} tr\left(\Lambda\Psi^{-1}(x^{(i)}-\mu)(z^{(i)})^T\right)\right]\\ &=& -\frac{1}{2}\sum_{i=1}^m \mathbb{E}\left[\Psi^{-T}\Lambda z^{(i)}(z^{(i)})^T+\Psi^{-1}\Lambda z^{(i)}(z^{(i)})^T-2\Psi^{-1}(x^{(i)}-\mu)(z^{(i)})^T\right]\\ &=&\sum_{i=1}^m \mathbb{E}\left[-\Psi^{-1}\Lambda z^{(i)}(z^{(i)})^T+\Psi^{-1}(x^{(i)}-\mu)(z^{(i)})^T\right] \end{eqnarray} $$</center>

The first and second steps used the fact that $$a=tr(a)$$ when $$a$$ is scalar, the properties $$tr(AB)=$$ $$tr(BA)$$ and $$tr(A^T)=tr(A)$$, a simple change in variable and the fact that $$\Psi$$ is symmetric and so $$\Psi^T=\Psi$$. The third step used three properties: $$\nabla_{A}tr(ABA^TC)$$ $$=CAB+C^TAB^T$$, $$\nabla_{A^T}f(A)=(\nabla_{A}f(A))^T$$ and $$\nabla_A tr(AB)= B^T$$.

Setting $$\nabla_\Lambda f(\mu,\Lambda, \Psi)$$ to zero gives:

<center>$$ \begin{eqnarray} 0 &=& \sum_{i=1}^m \mathbb{E}\left[-\Psi^{-1}\Lambda z^{(i)}(z^{(i)})^T+\Psi^{-1}(x^{(i)}-\mu)(z^{(i)})^T\right]\\ \sum_{i=1}^m \Psi^{-1}\Lambda \mathbb{E}\left[z^{(i)}(z^{(i)})^T\right] &=& \sum_{i=1}^m \Psi^{-1}(x^{(i)}-\mu) \mathbb{E}\left[(z^{(i)})^T\right]\\ \Lambda &=& \sum_{i=1}^m (x^{(i)}-\mu) \mathbb{E}_{z^{(i)}\sim Q_i} \left[(z^{(i)})^T\right]\left(\sum_{i=1}^m\mathbb{E}_{z^{(i)}\sim Q_i}\left[z^{(i)}(z^{(i)})^T\right]\right)^{-1}\\ \end{eqnarray} $$</center>

We know that $$\mathbb{E}_{z^{(i)}\sim Q_i}[(z^{(i)})^T]=\mu_{z^{(i)} \vert x^{(i)}}^T$$. Also we know that $$Cov(z^{(i)})=\mathbb{E}[z^{(i)}(z^{(i)})^T]-$$ $$\mathbb{E}[z^{(i)}]\mathbb{E}[(z^{(i)})^T]$$ [see [this]({{site.baseurl}}{%post_url /blog/mathematics/2018-10-30-the-multivariate-distribution%})] and so $$\mathbb{E}_{z^{(i)}\sim Q_i}[z^{(i)}(z^{(i)})^T]=Cov(z^{(i)})$$ $$+\mathbb{E}_{z^{(i)}\sim Q_i}[z^{(i)}]\mathbb{E}[(z^{(i)})^T]$$ $$=\Sigma_{z^{(i)}\vert x^{(i)}}+\mu_{z^{(i)} \vert x^{(i)}}\mu_{z^{(i)} \vert x^{(i)}}^T$$. Therefore:

<center>$$ \Lambda = \sum_{i=1}^m (x^{(i)}-\mu) \mu_{z^{(i)} \vert x^{(i)}}^T\left(\sum_{i=1}^m\Sigma_{z^{(i)}\vert x^{(i)}}+\mu_{z^{(i)} \vert x^{(i)}}\mu_{z^{(i)} \vert x^{(i)}}^T\right)^{-1}\\ $$</center>

We shall now derive the update for $$\mu$$:

<center>$$ \begin{eqnarray} \nabla_\mu f(\mu,\Lambda,\Psi) &=& -\frac{1}{2}\nabla_\mu\sum_{i=1}^m\mathbb{E}\left[-\mu\Psi^{-1}(x^{(i)}-\Lambda z^{(i)})+\mu^T\Psi^{-1}\mu-(x^{(i)}-\Lambda z^{(i)})^T\Psi^{-1}\mu\right]\\ &=& -\frac{1}{2}\sum_{i=1}^m\mathbb{E}\left[-2\nabla_\mu tr\left(\mu\Psi^{-1}(x^{(i)}-\Lambda z^{(i)})\right)+\nabla_\mu tr\left(\mu^T\Psi^{-1}\mu\right)\right]\\ &=& -\frac{1}{2}\sum_{i=1}^m\mathbb{E}\left[-\Psi^{-1}(x^{(i)}-\Lambda z^{(i)})+\Psi^{-1}\mu \right]\\ \end{eqnarray} $$</center>

The reader is referred to [this post]({{site.baseurl}}{%post_url /blog/mathematics/2018-08-15-matrix-derivatives%}) for the properties of traces that have been used in the derivation above. Setting the above to zero gives:

<center>$$ \begin{eqnarray} 0 &=& -\frac{1}{2}\sum_{i=1}^m\mathbb{E}\left[-\Psi^{-1}(x^{(i)}-\Lambda z^{(i)})+\Psi^{-1}\mu \right]\\ \sum_{i=1}^m \mu &=& \sum_{i=1}^m\mathbb{E}\left[x^{(i)}-\Lambda z^{(i)}\right]\\ \mu &=& \frac{1}{m}\left(\sum_{i=1}^m\mathbb{E}\left[x^{(i)}\right]-\Lambda\sum_{i=1}^m\mathbb{E}\left[ z^{(i)}\right]\right)\\ &\approx& \frac{1}{m}\sum_{i=1}^m\mathbb{E}\left[x^{(i)}\right] \end{eqnarray} $$</center>

The last step assumes that $$\sum_{i=1}^m \mathbb{E}[z^{(i)}] \approx 0$$ if $$m$$ is very large because the prior distribution of $$z$$ was defined to be a Gaussian with zero mean.

Let us now derive the update for $$\Psi$$. We have (see the derivation of $$\Sigma$$ for the Gaussian Discriminant Analysis in [this]({{site.baseurl}}{%post_url /blog/machine-learning/cs229-notes/2018-08-29-generative-learning-algorithms%}) post):

<center>$$ \nabla_{\Psi}f(\mu,\Lambda,\Psi) = -\frac{1}{2}\sum_{i=1}^m\mathbb{E}\left[\Psi^{-1}-\Psi^{-1}(x^{(i)}-\mu-\Lambda z^{(i)})(x^{(i)}-\mu-\Lambda z^{(i)})^T\Psi^{-1}\right] $$</center>

Setting this to zero gives:

<center>$$ \begin{eqnarray} 0 &=& -\frac{1}{2}\sum_{i=1}^m\mathbb{E}\left[\Psi^{-1}-\Psi^{-1}(x^{(i)}-\mu-\Lambda z^{(i)})(x^{(i)}-\mu-\Lambda z^{(i)})^T\Psi^{-1}\right]\\ \Psi &=& \frac{1}{m}\sum_{i=1}^m\mathbb{E}\left[(x^{(i)}-\mu-\Lambda z^{(i)})(x^{(i)}-\mu-\Lambda z^{(i)})^T\right]\\ &=& \frac{1}{m}\sum_{i=1}^m\mathbb{E}\left[(x^{(i)}-\mu)(x^{(i)}-\mu)^T -\Lambda z^{(i)}(x^{(i)}-\mu)^T -(x^{(i)}-\mu)(z^{(i)})^T\Lambda^T+\Lambda z^{(i)}(z^{(i)})^T\Lambda^T\right]\\ &=& \frac{1}{m}\sum_{i=1}^m(x^{(i)}-\mu)(x^{(i)}-\mu)^T -\Lambda \mathbb{E}\left[z^{(i)}\right](x^{(i)}-\mu)^T -(x^{(i)}-\mu)\mathbb{E}\left[(z^{(i)})^T\right]\Lambda^T+\Lambda \mathbb{E}\left[z^{(i)}(z^{(i)})^T\right]\Lambda^T\\ &=& \frac{1}{m}\sum_{i=1}^m(x^{(i)}-\mu)(x^{(i)}-\mu)^T -\Lambda \mathbb{E}\left[z^{(i)}\right](x^{(i)}-\mu)^T \end{eqnarray} $$</center>

The last step used the equation we had reached upon earlier when deriving the update for $$\Lambda$$, i.e.:

<center>$$ \sum_{i=1}^m \Lambda \mathbb{E}\left[z^{(i)}(z^{(i)})^T\right] = \sum_{i=1}^m (x^{(i)}-\mu) \mathbb{E}\left[(z^{(i)})^T\right] $$</center>

Therefore, the update for $$\Psi$$ is given by:

<center>$$ \Psi = \frac{1}{m}diag\left\{\sum_{i=1}^m(x^{(i)}-\mu)(x^{(i)}-\mu)^T -\Lambda \mu_{z^{(i)} \vert x^{(i)}}(x^{(i)}-\mu)^T\right\} $$</center>

Note that we restrict $$\Psi$$ to be a diagonal matrix here.