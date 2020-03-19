---
layout: post
title: Principal Component Analysis
permalink: blog/machine-learning/cs229-notes/principal-component-analysis
categories: [Machine Learning, CS229 Notes]
---

Suppose that we have a training set $$S=\{x^{(1)},...,x^{(m)}\}$$ where $$x^{(i)} \in \mathbb{R}^n$$ for $$i=1,...,m$$. Let $$x^{(i)}_j$$ denote the $$j^{th}$$ attribute/feature of $$x^{(i)}$$. The Principal Component Analysis (PCA) algorithm projects this $$n$$ dimensional space onto a $$k$$ dimensional subspace where $$% <![CDATA[ k < n %]]>$$. Informally speaking, the feature vector (i.e. each $$x^{(i)}$$) may contain certain redundancies, i.e two attributes $$x^{(i)}_m$$ and $$x^{(i)}_n$$ - that we do not know of - may be capturing the same feature/attribute of $$x^{(i)}$$. The PCA algorithm essentially ‘deletes’ these redundancies by choosing the $$k$$ features that best differentiate the $$x^{(i)}$$’s axes/directions along which the data varies maximally (i.e. has maximum variance). This means that the PCA algorithm essentially choses the $$k$$ directions/axes along which the $$x^{(i)}$$’s have maximum variance.

## Preprocessing

The PCA algorithm requires that the training set must have zero mean and unit variance. This can be done in the following way:

1.  Calculate the mean of the training set: $$\mu = \frac{1}{m}\sum_{i=1}^m x^{(i)}$$
2.  Replace each $$x^{(i)}$$ with $$x^{(i)}-\mu$$
3.  Calculate the variance: $$\sigma_j = \frac{1}{m}\sum_{i=1}^m (x^{(i)}_j)^2$$
4.  Replace each $$x^{(i)}_j$$ with $$x^{(i)}_j/\sigma^2_j$$

## The PCA Algorithm

Let $$k=1$$ and let $$\vec{\mu}$$ denote the unit basis vector of the $$k$$ dimensional subspace. Let $$\vec{x^{(i)}}$$ denote the vector from the origin to the point $$x^{(i)}$$ and let $$\theta$$ denote the angle between $$\vec{x^{(i)}}$$ and $$\vec{\mu}$$. Let projection of $$\vec{x^{(i)}}$$ on $$\vec{\mu}$$ be denoted by $$\vec{(x^{(i)})^{'}}$$:

<center>$$ \begin{eqnarray} \vec{(x^{(i)})^{'}} &=& \vert \vec{x^{(i)}} \vert cos(\theta) \vec{u}\\ &=& \vert \vec{x^{(i)}} \vert \frac{\vec{x^{(i)}} \dot{} \vec{u}}{\vert \vec{x^{(i)}} \vert \vert \vec{u} \vert} \vec{u}\\ &=& \left(\vec{x^{(i)}} \dot{} \vec{u}\right) \vec{u} \end{eqnarray} $$</center>

The last step made use of the fact that $$\vec{\mu}$$ is a unit vector. As explained before, we would like $$\vec{\mu}$$ to be such that the $$x^{(i)}$$’s when projected onto it have the maximum variance, i.e. we need to solve the following optimization problem:

<center>$$ \begin{eqnarray} \underset{\mu}{argmax} \frac{1}{m} \sum_{i=1}^m (\vec{x^{(i)}} \dot{} \vec{\mu})^2 &&\\ s.t. \vert \mu \vert = 1 && \end{eqnarray} $$</center>

The above expression may be rearranged as follows (we drop $$\vec{}$$ for notational convenience):

<center>$$ \begin{eqnarray} \underset{\mu}{argmax}\mu^T\left(\frac{1}{m}\sum_{i=1}^m x^{(i)} (x^{(i)})^T\right)\mu &&\\ s.t. \vert \mu \vert - 1 = 0 && \end{eqnarray} $$</center>

Note that the term in brackets is just the covariance matrix of the training set (recall that the mean of the training set was set to zero). Denotes this covariance matrix as $$\Sigma$$, we may formulate a Lagrangian for the above problem as follows:

<center>$$ \mathcal{L}(\mu,\alpha) = \mu^T\Sigma\mu + \alpha(\vert\mu\vert-1) $$</center>

Maximizing with respect to $$\mu$$ gives:

<center>$$ \begin{eqnarray} \nabla_\mu\mathcal{L}(\mu,\alpha) &=& \nabla_\mu\mu^T\Sigma\mu + \alpha\nabla_\mu\vert\mu\vert\\ &=& \nabla_\mu tr(\mu^T\Sigma\mu) + \alpha\nabla_\mu tr(\mu^T\mu)^{1/2}\\ &=& 2\Sigma\mu + \alpha\frac{\mu}{\vert\mu\vert} \end{eqnarray} $$</center>

Here we have made use of the properties of traces (see [this]({{site.baseurl}}{%post_url /blog/mathematics/2018-08-15-matrix-derivatives%})). Setting this to zero gives:

<center>$$ \Sigma \mu = \left(-\frac{\alpha}{2\vert\mu\vert}\right)\mu $$</center>

Note that the term inside the brackets is just a constant. Denoting this with $$\lambda$$ we get:

<center>$$ \Sigma \mu = \lambda \mu $$</center>

Therefore, $$\mu$$ is an eigenvector of $$\Sigma$$. Note that the variance is given by:

<center>$$ \begin{eqnarray} \sigma^2 &=& \mu^T\left(\frac{1}{m}\sum_{i=1}^m (x^{(i)} (x^{(i)})^T\right)\mu\\ &=& \mu^T\lambda\mu\\ &=& \lambda \end{eqnarray} $$</center>

The last step made use of the fact that $$\vec{\mu}$$ is a unit vector. Hence, the variances are equal to the eigenvalues. Note that we were originally trying to maximize the variances. Consequently, $$\lambda$$ must correspond to the largest eigenvalue of $$\Sigma$$. Therefore, $$\mu$$ is the principal eigenvector of $$\Sigma$$.

More generally, if $$k$$ ($$% <![CDATA[ <n %]]>$$) is the dimension of the subspace we want to project $$S$$ on, then the PCA algorithm chooses $$\mu_1,...,\mu_k$$ to be the top $$k$$ eigenvectors of $$\Sigma$$. The $$\mu_i$$’s then form the basis of the $$k$$ dimensional subspace and the following vector is calculated for each $$x^{(i)}$$.

<center>$$ y^{(i)} = \begin{bmatrix} \mu_1^Tx^{(i)}_1 \\ \mu_2^Tx^{(i)}_2 \\ . \\ . \\ . \\ \mu_k^Tx^{(i)}_k \end{bmatrix} $$</center>

Note that because $$\Sigma$$ is symmetric it can be always be orthogonally diagonalized. Consequently, the $$\mu_i$$’s will always (or can made to) be orthogonal.

## Implementing PCA using Singular Value Decomposition

Let $$X \in \mathbb{R}^{m \times n}$$ be our design matrix of containing all of our training examples (see [this]({{site.baseurl}}{%post_url /blog/machine-learning/cs229-notes/2018-08-20-linear-regression%}) for how the design matrix is defined). Then, $$\Sigma = X^TX$$. We may decompose $$X$$ via Singular Value Decomposition (SVD), i.e.:

<center>$$ X = UDV^T $$</center>

The the top $$k$$ columns of $$V$$ then give the top $$k$$ eigenvectors for $$X^TX = \Sigma$$.