---
layout: post
title: Assignment 1
permalink: blog/machine-learning/cs229-notes/assignment-1
categories: [Machine Learning, CS229 Notes]
---

**1\. Newton’s method for computing least squares**

(a) The Hessian is a matrix whose $$(i,j)$$ entry is given by:

<center>$$ \begin{eqnarray} H_{ij} &=& \frac{\partial J}{\partial \theta_j\partial \theta_k}\\ &=& \frac{\partial}{\partial \theta_k}\sum_{i=1}^m \left(\theta^Tx^{(i)} - y^{(i)}\right) x^{(i)}_j\\ &=& \sum_{i=1}^m x^{(i)}_jx^{(i)}_k \end{eqnarray} $$</center>

(b) Note that the Hessian is equal to $$X^TX$$ where $$X$$ is the design matrix. We showed in the lecture notes that $$\nabla_\theta J(\theta) = X^TX\theta - X^T\vec{y}$$. Therefore:

<center>$$ \begin{eqnarray} \theta &:=& \theta - H^{-1}\nabla_\theta f(\theta)\\ &=& \theta - (X^TX)^{-1}(X^TX\theta - X^T\vec{y})\\ &=& \theta - (X^TX)^{-1}(X^TX\theta) + (X^TX)^{-1}X^T\vec{y}\\ &=& (X^TX)^{-1}X^T\vec{y} \end{eqnarray} $$</center>

**3\. Multivariate least squares**

(a) Note that:

<center>$$ \begin{eqnarray} J(\theta) &=& \frac{1}{2} \sum_{i=1}^m \sum_{j=1}^p \left((\Theta^T x^{(i)})_j - y^{(i)}_j\right)^2 \\ &=& \frac{1}{2} \sum_{i=1}^m (\Theta^Tx^{(i)} - y^{(i)})^T(\Theta^Tx^{(i)} - y^{(i)})\\ &=& \frac{1}{2} (X\Theta - Y)^T(X\Theta - Y) \end{eqnarray} $$</center>

(b) To find the closed form equations, we need to set the derivative of $$J(\Theta)$$ to $$0$$:

<center>$$ \begin{eqnarray} \nabla_\Theta \frac{1}{2} (X\Theta - Y)^T(X\Theta - Y) &=& 0\\ \frac{1}{2} \nabla_\Theta tr\left[(X\Theta - Y)^T(X\Theta - Y)\right] &=& 0\\ \frac{1}{2} \nabla_\Theta tr\left[\Theta^TX^TX\Theta - \Theta^TX^TY-Y^TX\Theta + Y^TY \right] &=& 0\\ \frac{1}{2} \left(2X^TX\Theta - 2X^TY\right) &=& 0\\ \Theta &=& (X^TX)^{-1}X^TYY \end{eqnarray} $$</center>

(c) $$\theta_j$$ will just equal $$j^{th}$$ column of $$\Theta$$ (and thus the solution will be the same).

**4\. Naïve Bayes**

(a) and (b) These have already been done in [this]({{site.baseurl}}{%post_url /blog/machine-learning/cs229-notes/2018-08-29-generative-learning-algorithms%}) post.

(c) This can be shown in the following way:

<center>$$ \begin{eqnarray} p(y=1 \vert x) &\geqslant& p(y=0 \vert x)\\ \frac{p(x \vert y=1)p(y=1)}{p(x)} &\geqslant& \frac{p(x \vert y=0)p(y=0)}{p(x)}\\ \log\left(p(x \vert y=1)p(y=1)\right) &\geqslant& \log\left(p(x \vert y=0)p(y=0)\right)\\ \sum_{j=1}^n\log\left((\phi_{j\vert y=1})^{x_j}(1-\phi_{j\vert y=1})^{1-x_j}\right) + \log(\phi_y) &\geqslant& \sum_{j=1}^n\log\left((\phi_{j\vert y=0})^{x_j}(1-\phi_{j\vert y=0})^{1-x_j}\right) + \log(1-\phi_y)\\ \sum_{j=1}^n \left(x^{(j)}\log\left(\frac{\phi_{j\vert y=1}}{\phi_{j\vert y=0}}\right) + (1-x^{(j)}) \log\left(\frac{1-\phi_{j\vert y=1}}{1-\phi_{j\vert y=0}}\right)\right) &\geqslant& -\log\left(\frac{\phi_y}{1-\phi_y}\right)\\ \sum_{j=1}^n x^{(j)}\log\left(\frac{\phi_{j\vert y=1}(1-\phi_{j\vert y=0})}{\phi_{j\vert y=0}(1-\phi_{j\vert y=1})}\right) &\geqslant& -\sum_{j=1}^n\log\left(\frac{1-\phi_{j\vert y=1}}{1-\phi_{j\vert y=0}}\right)-\log\left(\frac{\phi_y}{1-\phi_y}\right)\\ \theta^T \begin{bmatrix}1\\x\end{bmatrix} &\geqslant& 0 \end{eqnarray} $$</center>

where:

<center>$$ \theta_j = \begin{cases} -\sum_{j=1}^n\log\left(\frac{1-\phi_{j\vert y=1}}{1-\phi_{j\vert y=0}}\right)-\log\left(\frac{\phi_y}{1-\phi_y}\right) & \text{if} & j =0\\ \log\left(\frac{\phi_{j\vert y=1}(1-\phi_{j\vert y=0})}{\phi_{j\vert y=0}(1-\phi_{j\vert y=1})}\right) & \text{if} & j = 1,...,n\end{cases} $$</center>

**5\. Exponential family and the geometric distribution**

(a) Note that:

<center>$$ \begin{eqnarray} p(y; \phi) &=& (1-\phi)^{y-1}\phi\\ &=& \exp\log((1-\phi)^{y-1}\phi)\\ &=& \exp\left(y\log(1-\phi)-\log\frac{1-\phi}{\phi}\right) \end{eqnarray} $$</center>

Therefore:

<center>$$ \begin{eqnarray} b(y) &=& 1\\ \eta &=& \log(1-\phi)\\ \phi &=& 1-\exp(\eta)\\ T(y) &=& y\\ \alpha(\eta) &=& \log\frac{1-\phi}{\phi} &=& \log\frac{\exp(\eta)}{1-\exp(\eta)} \end{eqnarray} $$</center>

(b) The canonical response function is $$\mathbb{E}[T(y);\eta] = \mathbb{E}[y;\eta] = \frac{1}{\phi} = \frac{1}{1-\exp(\eta)}$$.

(c) The log likelihood is given by:

<center>$$ \begin{eqnarray} l^{(i)}(\theta) &=&\log p(y^{(i)} \vert x^{(i)};\theta)\\ &=& \log\exp \left(y^{(i)}\log(1-\phi)-\log\frac{1-\phi}{\phi}\right)\\ &=& \eta y^{(i)}-\log\frac{\exp(\eta)}{1-\exp(\eta)}\\ &=& \theta^Tx^{(i)} y^{(i)}-\log\frac{\exp(\theta^Tx^{(i)})}{1-\exp(\theta^Tx^{(i)})}\\ \end{eqnarray} $$</center>

Therefore:

<center>$$ \begin{eqnarray} \frac{\partial l^{(i)}(\theta)}{\partial \theta_j} &=& x^{(i)}_jy^{(i)} - \frac{1-\exp(\theta^Tx^{(i)})}{\exp(\theta^Tx^{(i)})} \left(\frac{(1-\exp(\theta^Tx^{(i)}))\exp(\theta^Tx^{(i)})x_j^{(i)} - \exp(\theta^Tx^{(i)})(-\exp(\theta^Tx^{(i)})x^{(i)}_j)}{(1-\exp(\theta^Tx^{(i)}))^2}\right)\\ &=& \left(y^{(i)}-\frac{1}{1-\exp(\theta^Tx^{(i)})}\right)x^{(i)}_j \end{eqnarray} $$</center>

Hence, the stochastic gradient descent rule is given by:

<center>$$ \theta_j := \theta_j - \alpha \left(y^{(i)}-\frac{1}{1-\exp(\theta^Tx^{(i)})}\right) x^{(i)}_j $$</center>