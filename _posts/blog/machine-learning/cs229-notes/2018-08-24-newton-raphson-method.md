---
layout: post
title: Netwon-Raphson Method
permalink: blog/machine-learning/cs229-notes/newton-raphson-method
categories: [Machine Learning, CS229 Notes]
---

Suppose that we have a convex function $$f(\theta)$$ and we wish to find $$\theta^*$$ such that $$f(\theta^*) = 0$$. Note that a function $$f(\theta)$$ is convex in the interval $$[a,b]$$ if for any two points $$x_1$$, $$x_2$$ in $$[a,b]$$ and any $$\lambda$$ where $$0 \leqslant \lambda \leqslant 1$$:

<center>$$ f(\lambda x_1 + \lambda x_2) \leqslant \lambda f(x_1) + \lambda f(x_2) $$</center>

Alternatively (or consequently), a function $$f(\theta)$$ is convex if a tangent line $$g(\theta)$$ drawn at any point $$\theta$$ satisfies the following :

<center>$$ f(\theta) >= g(\theta) \ \ \ \ \ \ \ \forall \ \ \theta $$</center>

Suppose that we choose a random point $$\theta$$ on $$f(\theta)$$, such that $$\theta > \theta^*$$ and draw a tangent line $$g(\theta)$$ at that point. Suppose also that $$g(\theta') = 0$$. As a consequent of convexity, we would have $$\theta ' \geqslant \theta^*$$. Also, note that $$\theta ' \leqslant \theta$$. We may then repeat this process by setting $$\theta = \theta '$$ and drawing a new tangent line. Eventually, $$\theta$$ will converge to $$\theta^*$$.

The equation of a straight line is $$g(\theta) = g'(\theta) \theta + g(0)$$. At the end of each iteration, we want to set $$\theta = \theta '$$ such that $$g(\theta ') = 0$$. Therefore, we have:

<center>$$ \begin{eqnarray} 0 &=& g'(\theta) \theta + g(0)\\ \theta &=& \frac{-g(0)}{g'(\theta)}\\ \end{eqnarray} $$</center>

Using the equation of a straight line and the point [at which the tangent was drawn] $$(\theta, f(\theta))$$ and the fact that $$f'(\theta) = g(\theta)$$ we have at each iteration:

<center>$$ \theta := \theta - \frac{f(\theta)}{f'(\theta)} $$</center>

Note that the same argument may be made if we initially chose $$\theta$$ such that $$\theta \leq \theta^*$$.

Suppose, instead, we wanted to minimize (or maximize) a function $$f(\theta)$$. We know that the solution to the equation $$f'(\theta) = 0$$ gives us the required minima (or maxima). Consequently, at each iteration we have:

<center>$$ \theta := \theta - \frac{f'(\theta)}{f''(\theta)} $$</center>

This is known as Newton’s law. This may be further generalized to cases where $$\theta$$ is a vector :

<center>$$ \theta := \theta - H^{-1} \nabla_\theta f(\theta) $$</center>

where $$H$$ is a matrix, called as the Hessian, whose entries are given by:

<center>$$ H_{ij} = \frac{\partial^2 f(\theta)}{\partial \theta_i \partial \theta_j} $$</center>

This generalization is known as the Newton-Raphson method. Note the following points:

1.  While the Newton-Raphson method usually requires fewer iterations than the stochastic gradient descent algorithm to converge, in practice each iteration of the Newton-Raphson method is much more computationally expensive (because of the computation of the Hessian and its inverse).

2.  When the Newton-Raphson method is used to maximize the logistic regression log likelihood function, the resulting method is also called Fisher scoring.