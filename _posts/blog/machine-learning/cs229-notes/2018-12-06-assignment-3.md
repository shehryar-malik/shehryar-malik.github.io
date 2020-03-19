---
layout: post
title: Assignment 3
permalink: blog/machine-learning/cs229-notes/assignment-3
categories: [Machine Learning, CS229 Notes]
---

**1\. Uniform convergence and Model selection**

(a) Define a set of hypothesis $$\mathcal{H}_{cv}$$ to contain the hypothesis $$\{\hat{h}_1,...,\hat{h}_k\}$$. The training error for each hypothesis in $$\mathcal{H}_{cv}$$ is found by evaluating it on $$S_{cv}$$. Then, from the lecture notes it follows that with probability $$1-\frac{\delta}{2}$$ we have:

<center>$$ \begin{eqnarray} \vert\mathcal{E}(\hat{h}_i)-\mathcal{E}(\hat{h}_i)\vert &\leqslant& \sqrt{\frac{1}{2\vert S_{cv}\vert}\log\left(\frac{2k}{\frac{\delta}{2}}\right)}\\ &=& \sqrt{\frac{1}{2\beta m}\log\left(\frac{4k}{\delta}\right)}\\ \end{eqnarray} $$</center>

(b) From the lecture notes we have:

<center>$$ \mathcal{E}(\hat{h}) \leqslant \mathcal{E}(h^*) + 2\gamma $$</center>

where $$h^* = \underset{h}{\text{argmin}}[ \mathcal{E}(h)]$$ and $$\hat{h}$$ is as defined in the question. Therefore, substituting $$\gamma$$ with the value found in part (a) we have with probability $$1-\frac{\delta}{2}$$:

<center>$$ \begin{eqnarray} \mathcal{E}(\hat{h}) &\leqslant& \min_{i=1,...,k} \mathcal{E}(\hat{h}_i) + 2\sqrt{\frac{1}{2\beta m}\log\left(\frac{4k}{\delta}\right)}\\ &=& \min_{i=1,...,k} \mathcal{E}(\hat{h}_i) + \sqrt{\frac{2}{\beta m}\log\left(\frac{4k}{\delta}\right)} \end{eqnarray} $$</center>

(c) Note that the $$j^{th}$$ hypothesis in $$\mathcal{H}_{cv}$$ gives the lowest generalization error where as $$\hat{h}$$ gives the lowest training error. From (b) we have with probability $$1-\frac{\delta}{2}$$:

<center>$$ \mathcal{E}(\hat{h}) \leqslant \mathcal{E}(\hat{h}_j) + \sqrt{\frac{2}{\beta m}\log\left(\frac{4k}{\delta}\right)} $$</center>

Now, from the inequality given in part (c) and the lecture notes we have with probability $$1-\frac{\delta}{2}$$:

<center>$$ \mathcal{E}(\hat{h}_j) \leqslant \mathcal{E}(h_j^*) + \sqrt{\frac{2}{(1-\beta) m}\log\left(\frac{4\vert \mathcal{H_j} \vert}{\delta}\right)} $$</center>

where $$(1-\beta)m$$ follows from the fact that we use the set $$S_{train}$$ to evaluate each hypothesis in the set to which the $$j^{th}$$ hypothesis of $$\mathcal{H}_{cv}$$ belongs to and that the size of $$S_{train}$$ is $$(1-\beta)m$$. Also note that we have replaced the total number of hypothesis to $$\vert \mathcal{H_j}\vert$$ as we are restricting ourselves to the set $$\mathcal{H_j}$$ only.

From the above two inequalities we have with probability $$1-\frac{\delta}{2}$$:

<center>$$ \begin{eqnarray} \mathcal{E}(\hat{h}) &\leqslant& \mathcal{E}(h_j^*) + \sqrt{\frac{2}{(1-\beta) m}\log\left(\frac{4\vert \mathcal{H_j} \vert}{\delta}\right)} + \sqrt{\frac{2}{\beta m}\log\left(\frac{4k}{\delta}\right)}\\ &=&\min_{i=1,...,k}\left( \mathcal{E}(h_i^*) + \sqrt{\frac{2}{(1-\beta) m}\log\left(\frac{4\vert \mathcal{H_i} \vert}{\delta}\right)}\right) + \sqrt{\frac{2}{\beta m}\log\left(\frac{4k}{\delta}\right)}\\ \end{eqnarray} $$</center>

**2\. VC Dimension**

$$[ h(x) = 1(a<x)]$$: VC dimension = 1.

1.  $$h(x)$$ can shatter a $$d=1$$ points. Consider the set $$\{x_1\}$$. Then $$h(x)$$ can shatter this set for $$x_1=0$$ and $$x_1=1$$ by choosing $$a>x_1$$ and $$% <![CDATA[ a < x_1 %]]>$$ respectively.
2.  $$h(x)$$ cannot shatter $$d=2$$ points. Consider the set $$\{x_1,x_2\}$$ where $$% <![CDATA[ x_1<x_2 %]]>$$. Then the labelling $$x_1=1$$ and $$x_2=1$$ cannot be realized.

$$h(x) = 1(a<x<b)]$$: VC dimension = 2.

1.  $$h(x)$$ can shatter $$d=2$$ points. Consider the set $$\{x_1,x_2\}$$ where $$% <![CDATA[ x_1<x_2 %]]>$$. Then all four labellings can be realized:
    1.  $$x_1=x_2=0: x_1=x_2>a>b$$.
    2.  $$% <![CDATA[ x_1=x_2=1: a<x_1=x_2<b %]]>$$.
    3.  $$% <![CDATA[ x_1=0, x_2=1: x_1<a<x_2<b %]]>$$.
    4.  $$% <![CDATA[ x_1=1, x_2=0: a<x_1<b<x_2 %]]>$$.
2.  $$h(x)$$ cannot shatter $$d=3$$ points. Consider the set $$\{x_1,x_2,x_3\}$$ where $$% <![CDATA[ x_1<x_2<x_3 %]]>$$. Then the labelling $$x_1=x_3=1, x_2=0$$ cannot be realized.

$$[ h(x) = 1(asin(x)<0)]$$: VC dimension = 1.

1.  $$h(x)$$ can shatter a $$d=1$$ points. Consider the set $$\{x_1\}$$ where $$% <![CDATA[ 0< x_1<\pi %]]>$$ Then $$h(x)$$ can shatter this set for $$x_1=0$$ and $$x_1=1$$ by choosing $$% <![CDATA[ a<0 %]]>$$ and $$a > 0$$ respectively. Similarly, if $$\pi\leqslant x\leqslant 2\pi$$, then $$h(x)$$ can shatter this set for $$x_1=0$$ and $$x_1=1$$ by choosing $$a>0$$ and $$% <![CDATA[ a < 0 %]]>$$ respectively.
2.  $$h(x)$$ cannot shatter $$d=2$$ points. Consider the set $$\{x_1,x_2\}$$ where $$% <![CDATA[ x_1<x_2 %]]>$$. Then if the labelling $$x_1=0$$ and $$x_2=1$$ is realized for some $$a$$ then the labelling $$x_1=x_2=1$$ cannot be realized.

$$[h(x) = 1(sin(x+a)<0)]$$: VC dimension = 1.

1.  $$h(x)$$ can shatter $$d=2$$ points. Consider the set $$\{0,\pi/4\}$$. Then $$h(x)$$ can shatter this set for all four labellings $$(0,0)$$, $$(0,1)$$, $$(1,0)$$ and $$(1,1)$$ by choosing $$a=-\frac{\pi}{4}$$, $$0$$, $$\frac{3\pi}{4}$$ and $$\frac{\pi}{2}$$ respectively.
2.  $$h(x)$$ cannot shatter $$d=3$$ points. Consider the set $$\{x_1,x_2,x_3\}$$ where $$% <![CDATA[ 0<x_1<x_2<x_3<2\pi %]]>$$. Note that the constraint from $$0$$ to $$2\pi$$ can be satisfied by any set by simply adding or subtracting $$2\pi$$ from all $$x$$ repeatedly. Then in this case if the labelling $$x_1=x_3=1,x_2=0$$ can be realized then either the labelling $$x_1=x_2=x_3=1$$ or the labelling $$x_1=x_2=x_3=0$$ cannot be realized.

**3\. $$\mathcal{l}_1$$ regularization for least squares**

(a) Note that:

<center>$$ \begin{eqnarray} \frac{\partial J(\theta)}{\partial\theta_i} &=& \frac{\partial J(\theta)}{\partial\theta_i}\left(\frac{1}{2}tr\vert\vert X\bar{\theta} + X_i\theta_i - \bar{y} \vert\vert_2^2 + \lambda\vert\vert\bar{\theta}\vert\vert_1 + \lambda\vert\theta_i\vert\right)\\ &=& \frac{\partial J(\theta)}{\partial\theta_i}\left(\frac{1}{2}tr\left((X\bar{\theta} + X_i\theta_i - \bar{y} )^T(X\bar{\theta} + X_i\theta_i - \bar{y})\right) + \lambda\vert\vert\bar{\theta}\vert\vert_1 + \lambda\vert\theta_i\vert\right)\\ &=& \frac{\partial J(\theta)}{\partial\theta_i}\left(\frac{1}{2}tr\left(\bar{\theta}^TX^TX_i\theta_i +\theta_i^TX_i^TX\bar{\theta} + \theta_i^TX_i^TX_i\theta_i-\theta_i^TX_i^T\bar{y}-\bar{y}^TX_i\theta_i \right) + \lambda s_i\theta_i\right)\\ &=&\left(\frac{1}{2}\left((\bar{\theta}^TX^TX_i)^T + X_i^TX\bar{\theta} + \left(X_i^TX_i\theta_i + (\theta_i^TX_i^TX_i)^T\right) - X_i^T\bar{y}-(\bar{y}^TX_i)^T\right)\right) + \lambda s_i\\ &=& X_i^TX\bar{\theta} + X_i^TX_i\theta_i - X_i^T\bar{y} + \lambda s_i \end{eqnarray} $$</center>

In the third equation above we have only retained terms that are dependent on $$\theta_i$$. Note that $$\bar{\theta}$$ by definition does not depend upon $$\theta_i$$. Setting this to zero gives:

<center>$$ \theta_i = \frac{X_i^T(\bar{y}-X\bar{\theta})+\lambda s_i}{X_i^TX_i} $$</center>

Note that if $$s_i=+1$$ and if the update above results in $$% <![CDATA[ \theta_i<0 %]]>$$, then we must set its value to $$0$$. Similarly, if $$s_i=-1$$ and the update results in $$\theta_i>0$$, then we must set its value to $$0$$.

**4\. The Generalized EM algorithm**

(a) From the lecture notes we have:

<center>$$ \begin{eqnarray} l(\theta^{(t+1)}) &\geqslant& \sum_{i=1}^m \sum_{z^{(i)}} Q_i^{(t)}(z^{(i)})\log\left(\frac{p(x^{(i)},z^{(i)};\theta^{(t+1)})}{Q_i^{(t)}(z^{(i)})} \right)\\ &\geqslant& \sum_{i=1}^m \sum_{z^{(i)}}Q_i^{(t)}(z^{(i)})\log\left(\frac{p(x^{(i)},z^{(i)};\theta^{(t)})}{Q_i^{(t)}(z^{(i)})}\right)\\ &=& l(\theta^T) \end{eqnarray} $$</center>

where the second inequality, in this case, follows from the fact that we explicitly choose $$\alpha$$ such that the update on $$\theta$$ does not result in a decrease in the log likelihood function, $$l(\theta)$$.

(b) Note that for the GEM algorithm we have:

<center>$$ \begin{eqnarray} \nabla_\theta \sum_{i=1}^m \sum_{z^{(i)}}Q_i(z^{(i)})\log\left(\frac{p(x^{(i)},z^{(i)};\theta)}{Q_i(z^{(i)})}\right) &=& \sum_{i=1}^m\sum_{z^{(i)}}Q_i(z^{(i)})\nabla_\theta \log\left(\frac{p(x^{(i)},z^{(i)};\theta)}{Q_i(z^{(i)})}\right)\\ &=& \sum_{i=1}^m\sum_{z^{(i)}}\frac{Q_i(z^{(i)})}{p(x^{(i)},z^{(i)};\theta)}\nabla_\theta p(x^{(i)},z^{(i)};\theta)\\ &=& \sum_{i=1}^m\sum_{z^{(i)}}\frac{1}{\sum_{z^{(j)}}p(x^{(i)},z^{(j)};\theta)}\nabla_\theta p(x^{(i)},z^{(i)};\theta) \end{eqnarray} $$</center>

where the last step follows from the fact that we choose $$\frac{Q_i(z^{(i)})}{p(x^{(i)},z^{(i)};\theta)}$$ to be equal to $$\frac{1}{\sum_{z^{(j)}}p(x^{(i)},z^{(j)};\theta)}$$ in the E-step. Also for the update step given in the question we have:

<center>$$ \begin{eqnarray} \nabla_\theta\sum_{i=1}^m\log\sum_{z^{(i)}}p(x^{(i)},z^{(i)};\theta) &=& \sum_{i=1}^m\frac{1}{\sum_{z^{(j)}}p(x^{(i)},z^{(j)};\theta)}\sum_{z^{(i)}}\nabla_\theta p(x^{(i)},z^{(i)};\theta)\\ &=& \sum_{i=1}^m\sum_{z^{(i)}}\frac{1}{\sum_{z^{(j)}}p(x^{(i)},z^{(j)};\theta)}\nabla_\theta p(x^{(i)},z^{(i)};\theta) \end{eqnarray} $$</center>
