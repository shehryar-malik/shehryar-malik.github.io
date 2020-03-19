---
layout: post
title: Assignment 2
permalink: blog/machine-learning/cs229-notes/assignment-2
categories: [Machine Learning, CS229 Notes]
---

**1\. Kernel ridge regression**

(a) We showed in the lecture notes that $$\sum_{i=1}^m(\theta^Tx^{(i)}-y^{(i)})^2 = (X\theta-\vec{y})^T(X\theta-\vec{y})$$. Also, we showed that $$\nabla_\theta (X\theta-\vec{y})^T(X\theta-\vec{y}) = X^TX\theta-X^T\vec{y}$$:

<center>$$ \begin{eqnarray} \nabla_\theta J(\theta) &=& 0\\ \nabla_\theta \left(\frac{1}{2}(X\theta-\vec{y})^T(X\theta-\vec{y}) + \frac{\lambda}{2} \theta^T\theta\right) &=& 0\\ X^TX\theta - X^T\vec{y} + \nabla_\theta\frac{\lambda}{2} tr(\theta^T\theta) &=& 0\\ X^TX\theta - X^T\vec{y} + \lambda\theta &=& 0\\ \theta &=& (X^TX + \lambda I)^{-1}X^T\vec{y} \end{eqnarray} $$</center>

(b) Note that:

<center>$$ \begin{eqnarray} \theta^T\phi(x_{\text{new}}) &=& \left[(\phi(X)^T\phi(X)+\lambda I)^{-1}\phi(X)^T\vec{y}\right]^T\phi(x_{\text{new}})\\ &=& \vec{y}^T\phi(X)(\phi(X)^T\phi(X)+\lambda I)^{-1}\phi(x_{\text{new}})\\ &=& \vec{y}^T(\phi(X)\phi(X)^T+\lambda I)^{-1}\phi(X)\phi(x_{\text{new}}) \end{eqnarray} $$</center>

Here, $$\phi(X)\phi(X^T)$$ and $$\phi(X)\phi(x_{\text{new}})$$ are just matrices whose each entry is a dot product between $$\phi(x^{(i)})$$ and $$\phi(x^{(j)})$$ for some appropriate $$i$$ and $$j$$. Each of these dot products can efficiently be calculated using the kernel trick.

**2\. $$\mathcal{l}_2$$ norm soft margin SVMs**

(a) Allowing for negative $$\xi_i$$ will have no effect on the minimum value of the objective function because if the constraint $$y^{(i)}(w^Tx^{(i)}+b) \geqslant 1+ \vert\xi_i\vert$$ is satisfied for some negative $$\xi_i$$ then it is also satisfied for $$\xi_i = 0$$. However, $$\xi_i=0$$ will always result in a lower value of the objective function and hence the model will end up choosing it instead.

(b) The Lagrangian is given by:

<center>$$ \mathcal{L}(w,b,\xi_i,\alpha_i) = \frac{1}{2}\vert\vert w \vert\vert^2 + \frac{C}{2}\sum_{i=1}^m \xi_i^2 + \sum_{\substack{i=1\\\alpha_i>0}}^m\alpha_i(-y^{(i)}(w^Tx^{(i)}+b)+(1-\xi_i)) $$</center>

(c) Minimizing the Lagrangian with respect to the parameters gives:

<center>$$ \begin{eqnarray} \nabla_w\mathcal{L}(w,b,\xi_i,\alpha_i) &=& 0\\ w - \sum_{\substack{i=1\\\alpha_i>0}}^m\alpha_iy^{(i)}x^{(i)} &=& 0\\ w &=& \sum_{\substack{i=1\\\alpha_i>0}}^m\alpha_iy^{(i)}x^{(i)} \end{eqnarray} $$</center>

And:

<center>$$ \begin{eqnarray} \frac{\partial\mathcal{L}(w,b,\xi_i,\alpha_i)}{\partial b} &=& 0\\ \sum_{\substack{i=1\\\alpha_i>0}}^m \alpha_iy^{(i)}&=& 0 \end{eqnarray} $$</center>

Finally:

<center>$$ \begin{eqnarray} \frac{\partial\mathcal{L}(w,b,\xi_i,\alpha_i)}{\partial \xi_j} &=& 0\\ C\xi_j-\alpha_j &=& 0\\ \xi_j &=& \frac{\alpha_j}{C} \end{eqnarray} $$</center>

(d) Plugging the values of $$w$$ and $$\xi$$ found above and using the fact that $$\sum_{\substack{i=1\\\alpha_i>0}}^m \alpha_iy^{(i)}= 0$$ gives:

<center>$$ \begin{eqnarray} W(\alpha)&=& \min_{w,b,\xi_i}\mathcal{L}(w,b,\xi_i,\alpha_i)\\ &=& \sum_{i=1}^m \alpha_i- \frac{1}{2}\sum_{\substack{i=1\\\alpha_i>0}}^m \sum_{\substack{j=1\\\alpha_j>0}}^m \alpha_i\alpha_jy^{(i)}y^{(j)}{x^{(i)}}^Tx^{(j)} - \frac{1}{2}\sum_{i=1}^m\frac{\alpha_i^2}{C} \end{eqnarray} $$</center>

The dual problem is thus given by:

<center>$$ \max_{\alpha} W(\alpha)\\ \begin{eqnarray} s.t.\ \ \ \ \ \alpha_i &\geqslant& 0\\ \sum_{i=1}^m\alpha_iy^{(i)}&=& 0 \end{eqnarray} $$</center>

**3\. SVM with Gaussian Kernel**

(a) Note that to make a correct prediction on $$x^{(j)}$$ we require:

<center>$$ \begin{cases}f(x^{(j)}) > 0 & \text{if} & y^{(j)} = +1\\ f(x^{(j)}) < 0 & \text{if} & y^{(j)} = -1\end{cases} $$</center>

Or:

<center>$$ \begin{cases}f(x^{(j)}) - y^{(j)}> -1 & \text{if} & y^{(j)} = +1\\ f(x^{(j)}) - y^{(j)}< +1 & \text{if} & y^{(j)} = -1\end{cases} $$</center>

This implies that if $$% <![CDATA[ \vert f(x^{(j)})-y^{(j)} \vert < 1 %]]>$$ then we must have made the right decision. We need to find a value of $$\tau$$ for which this inequality holds for all $$j=1,...,m$$. Note that for all $$\alpha_i=1$$ and $$b=0$$ we have:

<center>$$ \begin{eqnarray} \vert f(x^{(j)})-y^{(j)} \vert &=&\vert \sum_{i=1}^m y^{(i)}K(x^{(i)},x^{(j)})-y^{(j)}\vert\\ &=& \sum_{i=1}^m y^{(i)}\exp(-\frac{\vert\vert x^{(i)}-x^{(j)}\vert\vert^2}{\tau^2})-y^{(j)}\vert\\ &=&\vert \sum_{\substack{i=1\\i\neq j}}^m y^{(i)}\exp(-\frac{\vert\vert x^{(i)}-x^{(j)}\vert\vert^2}{\tau^2})\vert\\ &\leqslant& \vert \sum_{\substack{i=1\\i\neq j}}^m y^{(i)}\exp(-\frac{\epsilon^2}{\tau^2})\vert\\ &=& \exp(-\frac{\epsilon^2}{\tau^2})\vert \sum_{\substack{i=1\\i\neq j}}^m y^{(i)}\vert\\ &\leqslant& \exp(-\frac{\epsilon^2}{\tau^2})(m-1) \end{eqnarray} $$</center>

where the last line follows from the fact that $$\vert\sum_{\substack{i=1\\i\neq j}}^m y^{(i)}\vert$$ is at most $$m-1$$ which corresponds to the case when all $$y^{(i)}$$ (except possibly for $$i=j$$ as $$y^{(j)}$$ is not included in this sum) are equal to $$+1$$ or $$-1$$. If there exists some $$\tau$$ for which $$% <![CDATA[ \exp(-\frac{\epsilon^2}{\tau^2})(m-1)<1 %]]>$$ then we can be certain that $$\vert f(x^{(j)})-$$ $$y^{(j)} \vert$$ $$% <![CDATA[ <1 %]]>$$ for all $$j$$:

<center>$$ \begin{eqnarray} \exp(-\frac{\epsilon^2}{\tau^2})(m-1) &<& 1\\ 2\log\exp(-\frac{\epsilon}{\tau}) &<& \log\frac{1}{m-1}\\ \log\exp(-\frac{\epsilon}{\tau}) &<& \log\frac{1}{m-1}\\ -\frac{\epsilon^2}{\tau^2} &<& \log\frac{1}{m-1}\\ \frac{\epsilon}{\tau} &>& \log(m-1)\\ \tau &<& \frac{\epsilon}{\log(m-1)} \end{eqnarray} $$</center>

(b) Yes, the model will always achieve zero training error. This is because the objective function with slack variables $$\xi_i \geqslant 0$$ contains the term $$C\sum_{i=1}^m \xi_i$$, where $$C$$ is some constant. One can, therefore, always choose the term $$C$$ to be so large (i.e $$C \rightarrow \infty$$) so that the objective function is minimized only when all $$\xi_i=0$$.

(c) No, the model will not necessarily achieve zero training error. Suppose that $$C=0$$. Then the objective function will end up being just $$\min_w \frac{1}{2}\vert\vert w \vert\vert^2$$. The model could then just choose $$w=0$$ (note that the objective function is minimized for this value of $$w$$) and pick $$\xi_i$$ that will satisfy the constraint $$y^{(i)}(w^Tx^{(i)}+b)\geqslant1-\xi_i$$ for all $$i$$.

**5\. Uniform Convergence**

(a) Let $$A_i$$ be the event that $$\mathcal{\hat{E}}(h_i)=0 \vert \mathcal{E}(h_i) > \gamma$$. Note that $$\mathcal{E}(h_i) > \gamma$$ implies that the probability of the hypothesis $$h_i$$ correctly classifying an example is equal to $$1-\gamma$$. Hence the probability of $$h_i$$ classifying all $$m$$ training examples correctly is $$(1-\gamma)^m$$. This, by definition, is equal to $$P(A_i)$$. So:

<center>$$ \begin{eqnarray} P(\exists h_i \in \mathcal{H}: \mathcal{\hat{E}}(h_i) = 0 \vert \mathcal{E}(h_i) > \gamma) &\leqslant& \sum_{i=1}^k P(A_i)\\ &=& \sum_{i=1}^k (1-\gamma)^m\\ &\leqslant& k\exp(-\gamma m) \end{eqnarray} $$</center>

Note that:

<center>$$ 1- P(\exists h_i \in \mathcal{H}: \mathcal{\hat{E}}(h_i) = 0 \vert \mathcal{E}(h_i) > \gamma) = P(\forall h_i \in \mathcal{H}: \mathcal{\hat{E}}(h_i) = 0 \vert \mathcal{E}(h_i) \leqslant \gamma) \geqslant 1-k\exp(-\gamma m) $$</center>

We want this to hold with probability at least $$1-\delta$$. Therefore:

<center>$$ \begin{eqnarray} 1-\delta &\geqslant& 1 -k\exp(-\gamma m)\\ \exp(-\gamma m) &\geqslant& \frac{\delta}{k}\\ -\gamma m &\geqslant& \log\frac{\delta}{k}\\ \gamma &\leqslant& \frac{1}{m}\log\frac{k}{\delta} \end{eqnarray} $$</center>

Hence, with probability $$1-\delta$$:

<center>$$ \mathcal{E}(\hat{h}) \leqslant \frac{1}{m}\log\frac{k}{\delta} $$</center>

where $$\hat{h}$$ is a hypothesis that achieves zero training error.

(b) Clearly:

<center>$$ m = \frac{1}{\gamma}\log\frac{k}{\delta} $$</center>

satisfies $$\mathcal{E}(\hat{h})\leqslant \gamma$$ with probability $$1-\delta$$, so for this to always hold it suffices that:

<center>$$ m \geqslant \frac{1}{\gamma}\log\frac{k}{\delta} $$</center>