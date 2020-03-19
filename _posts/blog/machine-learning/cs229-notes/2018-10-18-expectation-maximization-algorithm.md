---
layout: post
title: Expectation-Maximization Algorithm
permalink: blog/machine-learning/cs229-notes/expectation-maximization-algorithm
categories: [Machine Learning, CS229 Notes]
---

**Theorem (Jenson’s Inequality).** Let $$f$$ be a convex function (i.e $$f^{"} \geqslant 0$$, or for vector-valued inputs the hessian $$H$$ is positive semi-definite, i.e. $$H \geqslant 0$$) and $$X$$ be a random variable. Then $$\mathbb{E}[f(X)] \geqslant f(\mathbb{E}[X])$$. If $$f$$ is strictly convex (i.e. $$f^{"}> 0$$, or $$H$$ is positive definite, i.e. $$H>0$$), then $$\mathbb{E}[f(X)]$$$$= f(\mathbb{E}[X])$$ if and only if $$X=\mathbb{E}[X]$$ with probability $$1$$, i.e. $$X$$ is a constant.

Jenson’s inequality also holds for concave and strictly concave functions with the direction of inequality reversed. Note that $$f$$ is concave if $$-f$$ is convex and that $$f$$ is strictly concave if $$-f$$ is strictly convex.

Let $$S = \{x^{(1)}, x^{(2)}, ..., x^{(m)}\}$$ be a training set. The goal is to model the distribution $$p(x,z)$$ where $$z$$ are the latent random variables of the model. Let us define the log likelihood of the model as follows:

<center>$$ \begin{eqnarray} l(\theta) &=& log \prod_{i=1}^m p(x^{(i)}; \theta)\\ &=& \sum_{i=1}^m log \sum_{z^{(i)}} p(x^{(i)},z^{(i)}; \theta) \end{eqnarray} $$</center>

Note that the term $$\sum_{z^{(i)}}$$ is a summation over all possible values that $$z^{(i)}$$ can take.

Let us define $$Q_i$$ to be a distribution over $$z^{(i)}$$, i.e. $$\sum_{z^{(i)}} Q_i(z^{(i)}) = 1$$ and $$Q(z^{(i)})\geqslant0$$. Therefore:

<center>$$ l(\theta) = \sum_{i=1}^m log \sum_{z^{(i)}} Q_i(z^{(i)})\frac{p(x^{(i)},z^{(i)}; \theta)} {Q_i(z^{(i)})} $$</center>

Noting that the term $$Q_i(z^{(i)})\left(\frac{p(x^{(i)},z^{(i)}; \theta)}{Q_i(z^{(i)})}\right)$$ is essentially $$\mathbb{E}_{z \sim Q_i(z)} \left[ \frac{p(x^{(i)},z^{(i)}; \theta)}{Q_i(z^{(i)})}\right]$$ and using Jenson’s inequality (for concave functions) we have:

<center>$$ l(\theta) \geqslant \sum_{i=1}^m \sum_{z^{(i)}}Q_i(z^{(i)})log\left(\frac{p(x^{(i)},z^{(i)};\theta)}{Q_i(z^{(i)})}\right) $$</center>

Suppose that we have some guess of the parameters of the model, $$\theta$$. To transform the inequality above into an equality for this particular value of $$\theta$$ we require that:

<center>$$ \frac{p(x^{(i)},z^{(i)}; \theta)}{Q_i(z^{(i)})} = c $$</center>

where $$c$$ is a constant that does not depend on $$z$$. Let us choose $$c$$ to be $$\sum_{z^{(i)}} p(x^{(i)},z^{(i)})$$. Note that this makes the lower bound on $$l(\theta)$$ tight. Thus:

<center>$$ \begin{eqnarray} Q_i(z^{(i)}) &=& \frac{p(x^{(i)},z^{(i)}; \theta)}{\sum_{z^{(i)}}p(x^{(i)},z^{(i)})}\\ &=& \frac{p(x^{(i)},z^{(i)}; \theta)}{p(x^{(i)})}\\ &=& p(z^{(i)} \vert x^{(i)}) \end{eqnarray} $$</center>

The EM algorithm is then given as follows:

Repeat until convergence  
{

1.  (E-step) For each $$i$$ set:

    <center>$$ Q_i(z^{(i)}) := p(z^{(i)} \vert x^{(i)}) $$</center>

2.  (M-step) Update the parameters:  

<center>$$ \theta := \underset{\theta}{argmax} \sum_{i=1}^m \sum_{z^{(i)}} Q_i(z^{(i)})log\left(\frac{p(x^{(i)},z^{(i)};\theta)}{Q_i(z^{(i)})}\right) $$</center>

}

If we define $$J = \sum_{i=1}^m \sum_{z^{(i)}}Q_i(z^{(i)})log\left(\frac{p(x^{(i)},z^{(i)};\theta)}{Q_i(z^{(i)})}\right)$$, then the EM algorithm is essentially coordinate ascent on $$J$$. The E-step maximizes $$J$$ with respect to $$Q$$ (by making the bound tight) and the M-step maximizes $$J$$ with respect to $$\theta$$. Note that while we have not proved $$J$$ to be concave, it generally turns out to be so for the models we encounter.

Let us now prove the convergence of the EM algorithm. Let $$\theta^{(t)}$$ be the parameters that the model starts with at time $$t$$. Then:

<center>$$ l(\theta^{(t)}) = \sum_{i=1}^m \sum_{z^{(i)}}Q_i^{(t)}(z^{(i)})log\left(\frac{p(x^{(i)},z^{(i)};\theta^{(t)})}{Q_i^{(t)}(z^{(i)})}\right) $$</center>

We know that:

<center>$$ l(\theta^{(t+1)}) \geqslant \sum_{i=1}^m \sum_{z^{(i)}} Q_i(z^{(i)})log\left(\frac{p(x^{(i)},z^{(i)};\theta^{(t+1)})}{Q_i(z^{(i)})}\right) $$</center>

for any $$Q$$. Therefore:

<center>$$ l(\theta^{(t+1)}) \geqslant \sum_{i=1}^m \sum_{z^{(i)}} Q_i^{(t)}(z^{(i)})log\left(\frac{p(x^{(i)},z^{(i)};\theta^{(t+1)})}{Q_i^{(t)}(z^{(i)})}\right) $$</center>

Note that $$\theta^{(t+1)}$$ is explicitly chosen so that:

<center>$$ \sum_{i=1}^m \sum_{z^{(i)}}Q_i^{(t)}(z^{(i)})log\left(\frac{p(x^{(i)},z^{(i)};\theta^{(t+1)})}{Q_i^{(t)}(z^{(i)})}\right) \geqslant \sum_{i=1}^m \sum_{z^{(i)}}Q_i^{(t)}(z^{(i)})log\left(\frac{p(x^{(i)},z^{(i)};\theta^{(t)})}{Q_i^{(t)}(z^{(i)})}\right) $$</center>

Therefore:

<center>$$ \begin{eqnarray} l(\theta^{(t+1)}) &\geqslant& \sum_{i=1}^m \sum_{z^{(i)}}Q_i^{(t)}(z^{(i)})log\left(\frac{p(x^{(i)},z^{(i)};\theta^{(t)})}{Q_i^{(t)}(z^{(i)})}\right) \\ &=& l(\theta^{(t)}) \end{eqnarray} $$</center>

Hence, the likelihood of the model always increases.

## Mixture of Gaussians

Let us derive the EM equations for the Mixture of Gaussians model. In the E-step we do the following:

<center>$$ w^{(i)}_j = Q_i(z^{(i)}=j) = P(z^{(i)} =j \vert x^{(i)};\phi_j,\Sigma_j,\mu_j) $$</center>

In the M-step we update the parameters of the model by maximizing the following:

<center>$$ \begin{eqnarray} l(\phi,\Sigma,\mu) &=& \sum_{i=1}^m\sum_{z^{(i)}}Q_i(z^{(i)})log\left(\frac{p(x^{(i)}\vert z^{(i)};\Sigma,\mu)p(z^{(i)};\phi)}{Q_i(z^{(i)})}\right)\\ &=& \sum_{i=1}^m\sum_{j=1}^kQ_i(z^{(i)}=j)log\left(\frac{p(x^{(i)}\vert z^{(i)}=j;\Sigma,\mu)p(z^{(i)}=j;\phi)}{Q_i(z^{(i)}=j)}\right)\\ &=& \sum_{i=1}^m\sum_{j=1}^kw^{(i)}_jlog\left(\frac{\frac{1}{(2\pi)^{n/2}\vert \Sigma_j \vert^{1/2}}exp\left(-\frac{1}{2}(x^{(i)}-\mu_j)^T\Sigma_j^{-1}(x^{(i)}-\mu_j)\right)\phi_j}{w^{(i)}_j}\right)\\ \end{eqnarray} $$</center>

Therefore:

<center>$$ \begin{eqnarray} \nabla_{\mu_l}l(\phi,\Sigma,\mu) &=& \nabla_{\mu_l} \sum_{i=1}^m w^{(i)}_l \left(-\frac{1}{2}(x^{(i)}-\mu_l)^T\Sigma_l^{-1}(x^{(i)}-\mu_l)\right)\\ &=& -\frac{1}{2}\nabla_{\mu_l} \sum_{i=1}^m w^{(i)}_l \left((x^{(i)})^T\Sigma_l^{-1}(x^{(i)})-2\mu_l^T\Sigma_l^{-1}x^{(i)}-\mu_l^T\Sigma_l^{-1}\mu_l)\right)\\ &=& -\frac{1}{2}\sum_{i=1}^m w^{(i)}_l \Sigma_l^{-1}(x^{(i)}-\mu_j) \end{eqnarray} $$</center>

Setting this to zero gives:

<center>$$ \phi_l = \frac{\sum_{i=1}^m w^{(i)}_lx^{(i)}}{\sum_{i=1}^m w^{(i)}_l} $$</center>

Also:

<center>$$ \begin{eqnarray} \nabla_{\Sigma_{l}} l(\phi,\Sigma,\mu) &=& \sum_{i=1}^mw^{(i)}_l \nabla_{\Sigma_{l}} log\left(\frac{\frac{1}{(2\pi)^{n/2}\vert \Sigma_l \vert^{1/2}}exp\left(-\frac{1}{2}(x^{(i)}-\mu_l)^T\Sigma_l^{-1}(x^{(i)}-\mu_l)\right)\phi_l}{w^{(i)}_l}\right)\\ &=& \sum_{i=1}^mw^{(i)}_l \left(-\frac{1}{2}\Sigma^{-T}-\Sigma^{-T}(x^{(i)}-\mu_l)(x^{(i)}-\mu_l)^T\Sigma^{-T}\right) \end{eqnarray} $$</center>

Note that the derivation above is similar to that for [Gaussian Discriminant Analysis]({{site.baseurl}}{%post_url /blog/machine-learning/cs229-notes/2018-08-29-generative-learning-algorithms%}).

Setting this to zero gives:

<center>$$ \Sigma_l = \frac{\sum_{i=1}^mw^{(i)}_l(x^{(i)}-\mu_l)(x^{(i)}-\mu_l)^T}{\sum_{i=1}^mw^{(i)}_l} $$</center>

Lastly:

<center>$$ \nabla_{\phi_l}l(\phi,\Sigma,\mu) = \nabla_{\phi_l}\sum_{i=1}^m\sum_{j=1}^k w^{(i)}_jlog(\phi_j) $$</center>

However, note that we must ensure that $$\sum_j \phi_j = 1$$. We may thus construct a Lagrangian for the above equation subject to this condition as follows:

<center>$$ \mathcal{L}(\phi,\Sigma,\mu) = \sum_{i=1}^m\sum_{j=1}^k w^{(i)}_jlog(\phi_j) + \beta(\sum_{j=1}^k\phi_j-1) $$</center>

Therefore:

<center>$$ \nabla_{\phi_l} = \sum_{i=1}^m w^{(i)}_l\frac{1}{\phi_l} + \beta $$</center>

Setting this to zero gives:

<center>$$ \phi_l = \frac{\sum_{i=1}^m w^{(i)}_l}{-\beta} $$</center>

But we know that $$\sum_j \phi_j=1$$. Therefore $$-\beta=\sum_{i=1}^m\sum_{j=1}^k w^{(i)}_j$$. And as $$-\beta$$ $$=$$ $$\sum_{j=1}^k$$ $$w^{(i)}_j=$$ $$\sum_{j=1}^k$$ $$Q_i(z^{(i)}=j)=1$$, we have $$-\beta = m$$. So:

<center>$$ \phi_l = \frac{1}{m}\sum_{i=1}^m w^{(i)}_l $$</center>