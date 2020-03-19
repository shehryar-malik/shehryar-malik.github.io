---
layout: post
title: Learning Theory
permalink: blog/machine-learning/cs229-notes/learning-theory
categories: [Machine Learning, CS229 Notes]
---

Let us start with the following two lemmas:

**Lemma (The Union Bound).** Let $$A_1,A_2,...,A_k$$ be $$k$$ (not necessarily independent) events. Then:

<center>$$ P(A_1 \cup ... \cup A_k) \leqslant P(A_1) + ... + P(A_k) $$</center>

**Lemma (The Hoeffding Inequality/Chernoff Bound).** Let $$Z_1,...,Z_m$$ be $$m$$ independent and identically distributed (iid) random variables drawn from a Bernoulli($$\phi$$) distribution. Let $$\hat{\phi}=\frac{1}{m}\sum_{i=1}^{m}Z_i$$ be the mean of these random variables and let any $$\gamma > 0$$ be fixed. Then:

<center>$$ P(\vert \phi-\hat{\phi} \vert > \gamma) \leqslant 2\exp(-2\gamma^2m) $$</center>

Suppose that we are given a training set $$S = \{(x^{(i)},y^{(i)}); i=1,...,m\}$$ where $$y^{(i)} \in \{0,1\}$$ and the training examples $$(x^{(i)},y^{(i)})$$ are drawn iid from some random distribution $$\mathcal{D}$$. Let us define the training error or the empirical risk of a hypothesis $$h$$ to be:

<center>$$ \hat{\mathcal{E}}(h) = \frac{1}{m}\sum_{i=1}^m Z_i $$</center>

where $$Z_i$$ is the event where a training example $$x^{(i)},y^{(i)}$$ is drawn and $$h(x^{(i)}) \neq y^{(i)}$$ Let us also define the generalization error to be:

<center>$$ \mathcal{E}(h) = P_{(x,y)\sim\mathcal{D}}(h(x) \neq y) $$</center>

Let $$\mathcal{H}$$ be the set of all hypothesis considered by the learning algorithm. The goal of empirical risk minimization is to find the hypothesis $$\hat{h}$$ such that

<center>$$ \hat{h} = \underset{h}{argmin}\ \hat{\mathcal{E}}(h) $$</center>

## The Case of Finite \\(\mathcal{H}\\)

Suppose that $$\vert \mathcal{H} \vert =k$$. Let $$h_i$$ be a hypothesis from the set of $$\mathcal{H}$$. From the Hoeffding inequality we have:

<center>$$ P(\vert \mathcal{E}(h_i)-\hat{\mathcal{E}}(h_i) \vert > \gamma) \leqslant 2\exp(-2\gamma^2m) $$</center>

Let $$A_i$$ be the event that $$\vert \mathcal{E}(h_i)-\hat{\mathcal{E}}(h_i) \vert > \gamma$$. Then from the Union Bound we have:

<center>$$ \begin{eqnarray} P(\exists h_i \in \mathcal{H}.\ \vert \mathcal{E}(h_i)-\hat{\mathcal{E}}(h_i) \vert > \gamma) &\leqslant& \sum_{i=1}^kP(A_i)\\ &\leqslant& \sum_{i=1}^k2\exp(-2\gamma^2m)\\ &=& 2k\exp(-2\gamma^2m) \end{eqnarray} $$</center>

Subtracting both sides from $$1$$ gives:

<center>$$ P(\forall h_i \in \mathcal{H}.\ \vert \mathcal{E}(h_i)-\hat{\mathcal{E}}(h_i) \vert \leqslant\gamma)\geqslant 1-2k\exp(-2\gamma^2m) $$</center>

Suppose that we require that the probability that $$\vert \mathcal{E}(h_i)-\hat{\mathcal{E}}(h_i) \vert \leqslant \gamma$$ for all $$h_i$$ in $$\mathcal{H}$$ should at least be $$1-\delta$$, for some fixed $$\delta, \gamma > 0$$. The minimum value of $$m$$ required is thus:

<center>$$ \begin{eqnarray} 1 - \delta &=& 1-2k\exp(-2\gamma^2m)\\ m &=& \frac{1}{2\gamma^2}log(\frac{2k}{\delta}) \end{eqnarray} $$</center>

Note that this is the minimum value of $$m$$ required because increasing $$m$$ will only increase the probability $$1-2k\exp(-2\gamma^2m)$$$$(=1-\delta)$$. This minimum value of $$m$$ required to achieve a certain performance is known as the sample complexity.

Also note that:

<center>$$ \begin{eqnarray} \gamma &=& \sqrt{\frac{1}{2m}log(\frac{2k}{\delta})}\\ \vert \mathcal{E(h)}-\hat{\mathcal{E}}(h)\vert &\leqslant& \sqrt{\frac{1}{2m}log(\frac{2k} {\delta})} \end{eqnarray} $$</center>

Let $$\hat{h}=\underset{h}{argmin}$$ $$\hat{\mathcal{E}}(h)$$ and $$h^*=\underset{h}{argmin}$$ $$\mathcal{E}(h)$$. Then with probability $$1-\delta$$ we have:

<center>$$ \begin{eqnarray} \mathcal{E}(\hat{h}) &\leqslant& \hat{\mathcal{E}}(\hat{h})+\gamma\\ &\leqslant& \hat{\mathcal{E}}(h^*)+\gamma\\ &\leqslant& \mathcal{E}(h^*)+2\gamma\\ \end{eqnarray} $$</center>

We may sum the above discussion in the following theorem:

**Theorem.** Let $$\vert \mathcal{H} \vert = k$$, and let any $$m$$, $$\delta$$ be fixed. Then with probability at least $$1-\delta$$ we have that:

<center>$$ \underset{h\in\mathcal{H}}{argmin}\ \hat{\mathcal{E}}(h) \leqslant \underset{h\in\mathcal{H}}{argmin}\ \mathcal{E}(h)+2\sqrt{\frac{1} {2m}log(\frac{2k}{\delta})}\\ $$</center>

Note that switching to a larger hypothesis space $$\mathcal{H}' \supseteq \mathcal{H}$$ will only decrease the first term (the “bias”) but increase the second term (the “variance”). This is known as the bias-variance tradeoff.

**Corollary.** Let $$\vert \mathcal{H} \vert = k$$, and let any $$\gamma$$, $$\delta$$ be fixed. Then for $$\underset{h\in\mathcal{H}}{argmin}\ \hat{\mathcal{E}}(h) \leqslant \underset{h\in\mathcal{H}}{argmin}\ \mathcal{E}(h)+2\gamma$$ to hold it suffices that:

<center>$$ \begin{eqnarray} m &\geqslant& \frac{1}{2\gamma^2}log(\frac{2k}{\delta})\\ &=& O(\frac{1}{\gamma^2}log(\frac{k}{\delta}) ) \end{eqnarray} $$</center>

## The Case of Infinite \(\mathcal{H}\)

Given a set $$S = \{x^{(1)},x^{(2)},...,x^{(d)}\}$$ and a hypothesis class $$\mathcal{H}$$, we say that $$\mathcal{H}$$ shatters $$S$$ if $$\mathcal{H}$$ can realize any labelling on $$S$$, i.e. for some $$h \in \mathcal{H}$$, $$h(x^{(i)})=y^{(i)}$$ $$\forall$$ $$i$$. We define the Vapnik-Chervonenkis (VC) dimension of $$\mathcal{H}$$, $$VC(\mathcal{H})$$, to be the largest set that can be shattered by $$\mathcal{H}$$. Note that to show that $$d=VC(\mathcal{H})$$ we only need to prove that there is at least one set of size $$d$$ that can be shattered by $$\mathcal{H}$$, i.e. there might exist other sets of size $$d$$ that $$\mathcal{H}$$ cannot shatter.

The following theorem, due to Vapnik, can then be shown:

**Theorem.** Let $$\mathcal{H}$$ be given and let $$d = VC(\mathcal{H})$$. Then with probability $$1-\delta$$ we have that for all $$h \in \mathcal{H}$$:

<center>$$ \vert \mathcal{E}(h)-\hat{\mathcal{E}}(h)\vert \leqslant O\left(\sqrt{\frac{d}{m}log\frac{m}{d}+\frac{1}{m}log\frac{1}{\delta}}\right) $$</center>

Thus, with probability $$1-\delta$$ we also have:

<center>$$ \mathcal{E}(\hat{h}) \leqslant \mathcal{E}(h^*) + O\left(\sqrt{\frac{d}{m}log\frac{m}{d}+\frac{1}{m}log\frac{1}{\delta}}\right) $$</center>

Note that as $$m$$ becomes large, uniform convergence occurs.

**Corollary.** For $$\vert \mathcal{E}(h)-\hat{\mathcal{E}}(h)\vert \leqslant \gamma$$ (and hence $$\mathcal{E}(\hat{h}) \leqslant \mathcal{E}(h^*) +2\gamma$$) to hold for all $$h \in \mathcal{H}$$ with probability $$1-\delta$$, it suffices that $$m=O_{\gamma,\delta}(d)$$ (where the subscripts $$\gamma$$, $$\delta$$ indicate that $$O$$ is hiding some constants that may depend on $$\gamma$$ and $$\delta$$).

Note that all results proved above are for algorithms that use empirical risk minimization.

## Feature Selection

Consider the following problem: suppose that we are given a training set $$S=\{x^{(i)},y^{(i)}; i=1,...m\}$$ where each $$x^{(i)}$$is a high-dimensional vector representing the features of the $$i$$th training example. In order to reduce this high-dimensional vector (say, for reducing the amount of computation required to train our model), we would need to select a subset of, say, $$k$$ most “relevant” features. One way this can be done is to calculate the mutual information $$MI(x_{i},y)$$ between each feature $$x_{i}$$ and $$y$$:

<center>$$ MI(x_i,y) = \sum_{x_i \in \{0,1\}}\sum_{y \in \{0,1\}}p(x_i,y)log\frac{p(x_i,y)}{p(x_i)p(y)} $$</center>

where we have assumed that both $$x_i$$ and $$y$$ are binary-valued. In general, the summation would be over the domains of the variable. Note that this may also be expressed as a Kullback-Leibler (KL) divergence:

<center>$$ MI(x_i,y) = KL((p(x_i,y)\vert\vert p(x_i)p(y)) $$</center>

which, informally put, gives a measure of how different the probability distributions $$p(x_i,y)$$ and $$p(x_i)p(y)$$ are.

## Bayesian Statistics

Till now, we have considered $$\theta$$ to be an unknown but fixed variable. This is the view taken in frequentist statistics. In contrast, the Bayesian world of statistics considers $$\theta$$ to be a random variable with some prior probability distribution $$p(\theta)$$. Note that for a training set $$S=\{x^{(i)},y^{(i)}; i=1,...m\}$$:

<center>\begin{eqnarray} p(\theta \vert S) &=& \frac{P(S \vert \theta)P(\theta)}{P(S)}\\ &=& \frac{\prod_{i=1}^m P(y^{(i)}\vert x^{(i)}, \theta)P(\theta)}{\int_\theta \prod_{i=1}^m P(y^{(i)}\vert x^{(i)},\theta)p(\theta)d\theta} \end{eqnarray}</center>

Thus, for a test example $$T = (x,y)$$ we have:

<center>$$ \begin{eqnarray} p(y \vert x, S) &=& \int_\theta P(T \vert,S,\theta)d\theta\\ &=& \int_\theta \frac{P(T, S \vert \theta)}{P(S)}d\theta\\ &=& \int_\theta \frac{P(T \vert \theta)P(S \vert \theta)}{P(S)}d\theta\\ &=& \int_\theta \frac{P(T \vert \theta)P(\theta \vert S)P(S)}{P(S)P(\theta)}d\theta\\ &=& \int_\theta P(y \vert x,\theta)P(\theta \vert S)d\theta\\ \end{eqnarray} $$</center>

Note that the third step makes use of the assumption that $$S$$ and $$T$$ are iid i.e. they both are independent of each other. Also note that the second-to-last step uses the fact that $$\int_\theta P(\theta) = 1$$. Note that:

<center>$$ \mathbb{E}(y \vert x, S) = \int_y yp(y \vert x, S) $$</center>

However, as $$\theta$$ is very high dimensional computing an integral over all possible $$\theta$$’s is very computationally expensive. So instead, we replace $$\theta$$ by a single-point estimate - the maximum a posteriori (MAP) estimate, given by:

<center>$$ \theta_{MAP} = \underset{\theta}{argmax} \prod_{i=1}^m p(y^{(i)}\vert x^{(i)},\theta)p(\theta) $$</center>

A common choice for the prior $$p(\theta)$$ is $$\theta \sim \mathcal{N}(0,\tau^2I)$$. The Bayesian approach usually leads to less overfitting.

## The Perceptron and Large Margin Classifiers

Consider the problem of online learning. Suppose that we have a sequence of examples $$(x^{(1)},y^{(1)}), (x^{(2)},$$ $$y^{(2)}),$$$$...,(x^{(m)},y^{(m)})$$, where $$y^{(i)} \in \{-1,+1\}$$. Instead of feeding the entire set into our model at once, in online learning we will feed in these examples one-by-one and make a prediction at each step as follows:

<center>$$ h_\theta(x) = g(\theta^Tx) $$</center>

where:

<center>$$ g(z) = \begin{cases} +1, & \text{if}\ z \geqslant 0 \\ -1, & \text{if}\ z<0 \end{cases} $$</center>

At each step, if $$h_\theta(x^{(i)})=y^{(i)}$$ we will make the following update:

<center>$$ \theta :=\theta + y^{(i)}x^{(i)} $$</center>

The following theorem gives a bound on the number of mistakes this (perceptron algorithm) makes.

**Theorem (Block, 1962 and Novikoff, 1962).** Let a sequence of examples $$(x^{(1)},y^{(1)}),$$ $$(x^{(2)},$$ $$y^{(2)}),$$ $$...,(x^{(m)},y^{(m)})$$ be given. Suppose that $$\vert\vert x^{(i)} \vert\vert \leqslant D$$ for all $$i$$, and further that there exists a unit-length vector $$u$$ such that $$y^{(i)}u^T$$$$x^{(i)} \geq \gamma$$ for all examples in the sequence. Then the total number of mistakes the perceptron makes on this sequence is at most $$(\frac{D}{\gamma})^2$$.

**Proof.** Let $$\theta^{(k)}$$ be the weights of the perceptron when it makes the $$k$$th mistake and let $$\theta^{(0)} = \vec{0}$$. Let the $$k$$th mistake be on the example $$(x^{(i)},y^{(i)})$$. Therefore:

<center>$$ (x^{(i)})^T\theta^{(k)}y^{(i)} \leqslant 0 $$</center>

Also as $$\theta^{(k+1)} = \theta{(k)} + y^{(i)}x^{(i)}$$ we have:

<center>$$ \begin{eqnarray} (\theta^{(k+1)})^Tu &=& (\theta^{(k)})^Tu + y^{(i)}(x^{(i)})^Tu\\ &\geqslant& (\theta^{(k)})^Tu + \gamma\\ &\geqslant& (\theta^{(k-1)})^Tu + \gamma + \gamma\\ &\geqslant& k\gamma \end{eqnarray} $$</center>

Also:

<center>$$ \begin{eqnarray} \vert\vert \theta^{(k+1)} \vert\vert^2 &=& \vert\vert \theta^{(k)} \vert\vert^2 + \vert\vert x^{(i)}\vert\vert^2 + 2\vert\vert y^{(i)}(x^{(i)})^T\theta^{(k)} \vert\vert\\ &\leqslant& \vert\vert \theta^{(k)} \vert\vert^2 + \vert\vert x^{(i)}\vert\vert^2\\ &\leqslant& \vert\vert \theta^{(k)} \vert\vert^2 + D^2\\ &\leqslant& \vert\vert \theta^{(k-1)} \vert\vert^2 + D^2 + D^2\\ &\leqslant& kD^2 \end{eqnarray} $$</center>

Note that the first inequality used the fact that $$\vert\vert y^{(i)}(x^{(i)})^T\theta^{(k)} \vert\vert \leqslant 0$$.

Therefore:

<center>$$ \begin{eqnarray} \sqrt{k}D &\geqslant& \vert\vert \theta^{(k+1)} \vert\vert\\ &\geqslant& (\theta^{(k+1)})^Tu\\ &\geqslant& k\gamma\\ \end{eqnarray} $$</center>

Note that the second inequality above used the fact that $$z^Tu = \vert\vert z \vert\vert \ \vert\vert$$$$u \vert\vert cos\phi \leqslant$$$$\vert\vert z \vert\vert \ \vert\vert u \vert\vert$$ and that $$u$$ is a unit vector i.e. $$\vert\vert u \vert\vert = 1$$.

So:

<center>$$ k \leqslant (\frac{D}{\gamma})^2 $$</center>
