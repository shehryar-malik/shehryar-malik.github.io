---
layout: post
title: Mixture of Gaussians and the EM Algorithm
permalink: blog/machine-learning/cs229-notes/mixture-of-gaussians-and-the-em-algorithm
categories: [Machine Learning, CS229 Notes]
---

Consider a training set $$\{x^{(1)},x^{(2)},...,x^{(m)}\}$$. We wish to specify a joint distribution $$p(x^{(i)},z^{(i)})$$, where $$z^{(i)} \sim$$ $$Multinomial$$($$\phi$$), such that $$\sum_{i=1}^k \phi_k=1$$ and $$\phi_k \geqslant 0$$ and $$x^{(i)} \vert z^{(i)}=j \sim \mathcal{N}(\mu_j,\Sigma_j)$$. Note that the $$z^{(i)}$$’s are not known in advance and hence are the latent variables of this model. We may write the following the maximum likelihood function:

<center>$$ \begin{eqnarray} l(\mu,\Sigma,\phi) &=& \sum_{i=1}^m log\left(p(x^{(i)};\mu,\Sigma,\phi)\right)\\ &=& \sum_{i=1}^m log\left(\sum_{j=1}^kp(x^{(i)} \vert z^{(i)}=j;\mu,\Sigma)p(z^{(i)}=j;\phi)\right) \end{eqnarray} $$</center>

It can be shown that it is not possible to find a closed-form solution for the parameters of this model by setting $$\nabla l(\mu,\Sigma,\phi)=0$$. Instead, suppose that the $$z^{(i)}$$’s were known in advance. Then:

<center>$$ \begin{eqnarray} l(\mu,\Sigma,\phi) &=& \sum_{i=1}^m log \left(p(x^{(i)},z^{(i)};\mu,\Sigma,\phi)\right)\\ &=& \sum_{i=1}^m log \left(p(x^{(i)} \vert z^{(i)};\mu,\Sigma)p(z^{(i)};\phi)\right) \end{eqnarray} $$</center>

Maximizing this with respect to the parameters gives:

<center>$$ \begin{eqnarray} \phi_j &=& \frac{1}{m}\sum_{i=1}^m 1\{z^{(i)}=j\}\\ \mu_j &=& \frac{\sum_{i=1}^m 1\{z^{(i)}=j\}x^{(i)}}{\sum_{i=1}^m 1\{z^{(i)}=j\}}\\ \Sigma_j &=& \frac{\sum_{i=1}^m 1\{z^{(i)}=j\}(x^{(i)}-\mu_j)(x^{(i)}-\mu_j)^T}{\sum_{i=1}^m 1\{z^{(i)}=j\}} \end{eqnarray} $$</center>

Note that the derivation for these are similar to that for [Gaussian Discriminant Analysis]({{site.baseurl}}{%post_url /blog/machine-learning/cs229-notes/2018-08-29-generative-learning-algorithms%}).

The Expectation-Maximization (EM) algorithm is an iterative algorithm that first “guesses” the value of $$z^{(i)}$$’s (the E-step) and then maximizes the likelihood function with respect to the parameters by using these (soft) guesses for $$z^{(i)}$$’s (the M-step):

Repeat until convergence  
{

1.  (E-step) For each $$i,j$$ set:

    <center>$$ w_j^{(i)} := p(z^{(i)}=j \vert x^{(i)}; \phi,\mu,\Sigma) = \frac{p(x^{(i)} \vert z^{(i)};\mu,\Sigma)p(z^{(i)};\phi)}{\sum_{i=1}^m p(x^{(i)} \vert z^{(i)};\mu,\Sigma)p(z^{(i)};\phi)} $$</center>

2.  (M-step) Update the parameters:  

<center>$$ \begin{eqnarray} \phi_j &=& \frac{1}{m}\sum_{i=1}^m w_j^{(i)}\\ \mu_j &=& \frac{\sum_{i=1}^m w_j^{(i)}x^{(i)}}{\sum_{i=1}^m w_j^{(i)}}\\ \Sigma_j &=& \frac{\sum_{i=1}^m w_j^{(i)}(x^{(i)}-\mu_j)(x^{(i)}-\mu_j)^T}{\sum_{i=1}^m w_j^{(i)}} \end{eqnarray} $$</center>

}
