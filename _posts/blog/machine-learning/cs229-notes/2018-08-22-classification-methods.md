---
layout: post
title: Classification Methods
permalink: blog/machine-learning/cs229-notes/classification-methods
categories: [Machine Learning, CS229 Notes]
---

## 1\. Logistic Regression

Suppose that we have a training set, $$\{(x^{(i)}, y^{(i)}); i=1,..,m\}$$, such that $$y^{(i)}$$ $$\in$$ $${\{0,1\}}$$. Let us also define the logistic (or sigmoid) function as follows:

<center>$$ g(z) = \frac{1}{1+e^{-z}} $$</center>

It can be shown that:

<center>$$ g'(z) = g(z)(1-g(z)) $$</center>

Let us define our hypothesis $$h_\theta(x^{(i)})$$ to be:

<center>$$ h_\theta(x^{(i)}) = g(\theta^Tx^{(i)}) = \frac{1}{1+e^{-\theta^Tx^{(i)}}} $$</center>

As $$g(z)$$, and consequently $$h_\theta(x^{(i)})$$, $$\in (0,1]$$, we may define:

<center>$$ \begin{eqnarray} P(y^{(i)}=1 \vert x^{(i)}; \theta) &=& h_\theta(x^{(i)}) \\ P(y^{(i)}=0 \vert x^{(i)}; \theta) &=& 1 - h_\theta(x^{(i)}) \end{eqnarray} $$</center>

which may also be written as:

<center>$$ P(y^{(i)} \vert x^{(i)}; \theta) = (h_\theta(x^{(i)}))^{y^{(i)}}(1-h_\theta(x^{(i)}))^{1-y^{(i)}} $$</center>

For $$m$$ training examples, we have (because of independence):

<center>$$ P(y|x; \theta) = \prod_{i=1}^{m} (h_\theta(x^{(i)}))^{y^{(i)}}(1-h_\theta(x^{(i)}))^{1-y^{(i)}} $$</center>

We thus need to maximize the likelihood $$L(\theta) = P(y \vert x; \theta)$$. Let us maximize the log likelihood $$\mathcal{l}(\theta)$$ for a single training example for now:

<center>\begin{eqnarray} \mathcal{l}(\theta) &=& log\ L(\theta)\\ &=& log (h_\theta(x^{(i)}))^{y^{(i)}}(1-h_\theta(x^{(i)}))^{1-y^{(i)}}\\ &=& y^{(i)}log(h_\theta(x^{(i)}))+(1-y^{(i)})log(1-h_\theta(x^{(i)}))\\ &=& y^{(i)}log(g(\theta^Tx^{(i)}))+(1-y^{(i)})log(1-g(\theta^Tx^{(i)})) \end{eqnarray}</center>

Taking the derivative of $$\mathcal{l}(\theta)$$ with respect to $$\theta_{j}$$ gives:

<center>\begin{eqnarray} \frac{\partial \mathcal{l(\theta)}}{\partial \theta_j} &=& \left(\frac{y^{(i)}}{g(\theta^Tx^{(i)})} - \frac{1-y^{(i)}}{1-g(\theta^Tx^{(i)})}\right)g'(\theta^Tx^{(i)})\\ &=& \left(\frac{y^{(i)}}{g(\theta^Tx^{(i)})} - \frac{1-y^{(i)}}{1-g(\theta^Tx^{(i)})}\right)g(\theta^Tx^{(i)})(1-g(\theta^Tx^{(i)}))x^{(i)}_j\\ &=& (y^{(i)}(1-g(\theta^Tx^{(i)})) - (1-y^{(i)})g(\theta^Tx^{(i)}))x^{(i)}_j\\ &=& (y^{(i)} - g(\theta^Tx^{(i)}))x^{(i)}_j\\ &=& (y^{(i)} - h_\theta(x^{(i)}))x^{(i)}_j \end{eqnarray}</center>

This gives us the update for one training example. We may then use the batch or stochastic gradient descent to train over the entire training set.

## 2\. Perceptron Learning Algorithm

The perceptron learning algorithm uses the same update rule as in linear and logistic regression but with a different hypothesis function $$h_\theta(x^{(i)})$$ $$=$$ $$g(\theta^Tx)$$ $$=$$ $$g(z)$$ where $$g(z)​$$ is given as follows:

<center>$$ g(z) = \begin{cases} 1, & \text{if}\ z \geqslant 0 \\ 0, & \text{if}\ z<0 \end{cases} $$</center>
