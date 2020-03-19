---
layout: post
title: Linear Regression
permalink: blog/machine-learning/cs229-notes/linear-regression
categories: [Machine Learning, CS229 Notes]
---

Let each $$x^{\left(i\right)}$$ in a training set be an $$N$$ dimensional vector. We can approximate the hypothesis $$h$$ using a linear function:

<center>$$h_{\theta}(x^{\left(i\right)}) = \theta_{0} + \theta_{1}x_1^{\left(i\right)} + \theta_{2}x_2^{\left(i\right)} + . . . + \theta_{n}x_n^{\left(i\right)}$$</center>

Or, compactly:

<center>$$h_{\theta}(x^{\left(i\right)}) = \sum_{i=0}^{n} \theta^Tx^{\left(i\right)}$$</center>

where $$x_0^{\left(i\right)} = 1$$ is known as the intercept term and $$\theta$$’s are the parameters or weights of the hypothesis parameterizing the space of linear functions mapping $$\mathcal{X}$$ to $$\mathcal{Y}$$.

Let $$J\left(\theta\right)$$ be a cost function that measures how close the $$h_{\theta}(x^{\left(i\right)})$$’s are to the $$y^{\left(i\right)}$$’s:

<center>$$J(\theta) = \frac{1}{2} \sum_{i=1}^{m} \left(h_{\theta}(x^{\left(i\right)})-y^{\left(i\right)}\right)^2$$</center>

### 1\. Gradient Descent Algorithm

The gradient descent algorithm performs an update rule on $$\theta$$ repeatedly until it converges to the minimum point of $$J\left(\theta\right)$$ and is given by:

<center>$$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)$$</center>

where $$\alpha$$ is a hyperparameter known as the learning rate.

Note that for a single training example:

<center>$$ \begin{eqnarray} \frac{\partial}{\partial \theta_j} J(\theta) &=& \frac{\partial}{\partial \theta_j} \frac{1}{2} (h_{\theta}(x^{(i)})-y^{(i)})^2 \nonumber \\ &=& (h_{\theta}(x^{(i)})-y^{(i)}) \frac{\partial}{\partial \theta_j}(h_{\theta}(x^{(i)})-y^{(i)}) \nonumber \\ &=& (h_{\theta}(x^{(i)})-y^{(i)}) \frac{\partial}{\partial \theta_j}(\sum_{i=0}^{n} \theta^Tx^{(i)}-y^{(i)}) \nonumber \\ &=& (h_{\theta}(x^{(i)})-y^{(i)}) x_j^{(i)} \nonumber \end{eqnarray} \nonumber $$</center>

Therefore:

<center>$$\theta_j := \theta_j + \alpha \left(y^{\left(i\right)} - h_{\theta}(x^{\left(i\right)})\right)x^{\left(i\right)}_j$$</center>

This is known as the Least Mean Squares (LMS) update rule or the Widrow-Hoff learning rule, and can be extended to an entire training set through two methods:

#### (i) Batch Gradient Descent

Repeat until convergence:  
      For every $$j$$:  
            $$\theta_j := \theta_j + \alpha \sum_{i=1}^{m}\left(y^{\left(i\right)} - h_{\theta}(x^{\left(i\right)})\right)x^{\left(i\right)}_j$$

#### (ii) Stochastic (or Incremental) Gradient Descent (SGD)

Repeat until convergence:  
      For $$i = 1$$ to $$m$$:  
            For every $$j$$:  
                  $$\theta_j := \theta_j + \alpha \left(y^{\left(i\right)} - h_{\theta}(x^{\left(i\right)})\right)x^{\left(i\right)}_j$$

Note that while for batch gradient descent we had to go through an entire training set to make one update, stochastic gradient descent allows us to make an update after processing each training example. Often, the SGD algorithm converges to the minimum point much faster than the batch gradient descent.

### 2\. The Normal Equations

Let us define a matrix-vectorial notation for our cost function $$J(\theta)$$. Let $$X \in \mathbb{R}^{m \times n}$$ be the design matrix that contains the input values in a training set, that is:

<center>$$ \begin{bmatrix} -- \left( x^{\left(1\right)}\right)^T -- \\ -- \left( x^{\left(2\right)}\right)^T -- \\ .\\ .\\ .\\ -- \left( x^{\left(m\right)}\right)^T -- \end{bmatrix} $$</center>

Let $$\vec{y} \in \mathbb{R}^m$$ be a vector that contains the outputs values of the training set, that is:

<center>$$ \begin{bmatrix} y^{\left(1\right)}\\ y^{\left(2\right)}\\ .\\ .\\ .\\ y^{\left(m\right)}\\ \end{bmatrix} $$</center>

Since, $$h_{\theta}(x^{\left(i\right)}) = \left(x^{\left(i\right)}\right)^T\theta$$, we have:

<center>$$ X\theta - \vec{y} = \begin{bmatrix} h_\theta(x^{\left(1\right)}) - y^{\left(1\right)}\\h_\theta(x^{\left(2\right)}) - y^{\left(2\right)}\\.\\.\\.\\h_\theta(x^{\left(m\right)}) - y^{\left(m\right)}\end{bmatrix} $$</center>

We know that for a vector $$z, z^Tz = \sum_{i}z_i^2$$. Therefore:

<center>$$ \begin{eqnarray} \frac{1}{2} \left(X\theta - y\right)^T\left(X\theta - y\right) &=& \frac{1}{2}\sum_{i=1}^m \left(h_\theta(x^{\left(i\right)}) - y^{\left(i\right)}\right)^2 \\ &=& J(\theta) \end{eqnarray} $$</center>

Let us now find the derivative of $$J(\theta)$$ with respect to $$\theta$$. For this we shall be making use of the properties of a trace of a matrix. The reader is referred to [this]({{site.baseurl}}{%post_url /blog/mathematics/2018-08-15-matrix-derivatives%}) article.

<center>$$ \begin{eqnarray} \nabla_\theta J(\theta) &=& \nabla_\theta \frac{1}{2}\left(X\theta - \vec{y}\right)^T\left(X\theta - \vec{y}\right) \\ &=& \nabla_\theta \frac{1}{2} \left(\theta^TX^TX\theta - \theta^TX^T\vec{y} - \vec{y}^TX\theta - \vec{y}^T\vec{y}\right) \\ &=& \nabla_\theta \frac{1}{2} tr\left(\theta^TX^TX\theta - \theta^TX^T\vec{y} - \vec{y}^TX\theta - \vec{y}^T\vec{y}\right) \\ &=& \nabla_\theta \frac{1}{2} \left(tr\left(\theta^TX^TX\theta\right) - 2tr\left(\vec{y}^TX\theta\right)\right) \\ &=& \frac{1}{2} \left(X^TX\theta + X^TX\theta - 2X^T\vec{y}\right) \\ &=& X^TX\theta - X^T\vec{y} \end{eqnarray} $$</center>

The minimum point of $$J(\theta)$$ can be found by setting $$\nabla_\theta J(\theta)=0$$ and obtaining the normal equations:

<center>$$ X^TX\theta - X^T\vec{y} = 0 $$</center>

Hence, the value of $$\theta$$ that minimizes $$J\left(\theta\right)$$ is given in closed form by:

<center>$$ \theta = \left(X^TX\right)^{-1}X^T\vec{y} $$</center>

### 3\. Probabilistic Interpretation

Suppose that the input and target values are related by the following equation:

<center>$$ y^{\left(i\right)} = \theta^Tx^{\left(i\right)} + \epsilon^{\left(i\right)} $$</center>

where $$\epsilon^{\left(i\right)}$$ either captures unmodeled effects or random noise. Suppose also that $$\epsilon^{\left(i\right)} \sim \mathcal{N}\left(0,\sigma^2\right)$$ and are independently and identically distributed (IID), that is:

<center>$$ p(e^{\left(i\right)}) = \frac{1}{\sqrt{2\pi\sigma^2}} exp\left(-\frac{\left(e^{\left(i\right)}\right)^2}{2\sigma^2}\right) $$</center>

Or:

<center>$$ p(e^{\left(i\right)}) = \frac{1}{\sqrt{2\pi\sigma^2}} exp\left(-\frac{\left(y^{\left(i\right)} - \theta^Tx^{\left(i\right)}\right)^2}{2\sigma^2}\right) $$</center>

Now the expression on the right hand side in the above equation can be interpreted as representing a probability distribution of some sort of relationship between $$x^{\left(i\right)}​$$ and $$y^{\left(i\right)}​$$. Let $$\theta^Tx^{\left(i\right)}​$$ represent the mean of this probability distribution, then because the mean is a constant value $$x^{\left(i\right)}​$$, in this relationship, must be “fixed”. Hence:

<center>$$ p(y^{\left(i\right)} \vert x^{\left(i\right)};\theta) = \frac{1}{\sqrt{2\pi\sigma^2}} exp\left(-\frac{\left(y^{\left(i\right)} - \theta^Tx^{\left(i\right)}\right)^2}{2\sigma^2}\right) $$</center>

or, simply, $$y^{\left(i\right)} \vert x^{\left(i\right)} \sim \mathcal{N}\left(\theta^Tx^{\left(i\right)},\sigma^2\right)$$. Note that the distribution is conditioned on $$x^{\left(i\right)}$$ rather than on $$y^{\left(i\right)}$$ because we are trying to predict the latter given the former.

As the $$e^{\left(i\right)}$$’s are IID, so are the $$y^{\left(i\right)} \vert x^{\left(i\right)};\theta$$’s. Thus, for the entire training set we may define a likelihood function as follows:

<center>$$ L(\theta) = \prod_{i=1}^{m}p\left(y^{\left(i\right)} \vert x^{\left(i\right)};\theta\right) $$</center>

Our goal, thus, is to choose a value of $$\theta$$ that maximizes $$L(\theta)$$. This is known as the principle of maximum likelihood. Alternatively, we may maximize the so-called log likelihood, $$\mathcal{l}\left(\theta\right)$$:

<center>$$ \begin{eqnarray} \mathcal{l}(\theta) &=& log \ L(\theta)\\ &=& log\prod_{i=1}^{m}p\left(y^{\left(i\right)} \vert x^{\left(i\right)};\theta\right)\\ &=& \sum_{i=1}^{m} log\left(\frac{1}{\sqrt{2\pi\sigma^2}} exp\left(-\frac{\left(y^{\left(i\right)} - \theta^Tx^{\left(i\right)}\right)^2}{2\sigma^2}\right)\right)\\ &=& mlog\left(\frac{1}{\sqrt{2\pi\sigma^2}}\right) - \sum_{i=1}^{m}\frac{\left(y^{\left(i\right)} - \theta^Tx^{\left(i\right)}\right)^2}{2\sigma^2} \end{eqnarray} $$</center>

Note that maximizing $$\mathcal{l}(\theta)$$ is the same as minimizing $$\sum_{i=1}^{m} \left(y^{\left(i\right)} - \theta^Tx^{\left(i\right)}\right)^2$$ which is just the cost function $$J(\theta)$$ we defined earlier.

### 4\. Locally-Weighted Linear Regression

Suppose that we have predict $$y$$ given $$x \in \mathcal{R}^n$$, given a training set. The locally-weighted linear regression algorithm does the following:

1.  Fit $$\theta$$ to minimize $$\sum_i{w^{\left(i\right)}\left(y^{\left(i\right)}-\theta^Tx^{\left(i\right)}\right)}$$
2.  Output $$\theta^Tx​$$

where $$w^{\left(i\right)}$$’s are non-negative weights. A fairly standard choice for the weights is:

<center>$$ w^{\left(i\right)} = exp\left(-\frac{\left(x^{\left(i\right)}-x\right)^T\left(x^{\left(i\right)}-x\right)}{2\tau^2}\right) $$</center>

where $$\tau$$ is called as the bandwidth.

Simply put, the weights ensure that training examples closer to the point $$x$$ (the point at which a prediction needs to be made) are given a higher priority. Note that this means that unlike linear regression, locally-weighted regression requires that the training set must always be available when a new prediction needs to be made. Algorithms with this property are often called as non-parametric algorithms.