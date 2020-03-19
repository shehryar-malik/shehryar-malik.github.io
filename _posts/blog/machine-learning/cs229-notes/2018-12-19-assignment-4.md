---
layout: post
title: Assignment 4
permalink: blog/machine-learning/cs229-notes/assignment-4
categories: [Machine Learning, CS229 Notes]
---

**1\. EM for supervised learning**

(a) If $$z^{(i)}$$ are known then the log likelihood is given by:

<center>$$ \begin{eqnarray} l(\theta_0,\theta_1,\phi) &=& \log \prod_{i=1}^m p(y^{(i)}\vert x^{(i)},z^{(i)};\theta_0,\theta_1)p(z^{(i)}\vert x^{(i)};\phi)\\ &=& \sum_{i=1}^m \log p(y^{(i)}\vert x^{(i)},z^{(i)};\theta_0,\theta_1) + \log p(z^{(i)}\vert x^{(i)};\phi) \end{eqnarray} $$</center>

Taking the derivative with respect to $$\theta_0$$ (by only considering the terms that depend on $$\theta_0$$) gives:

<center>$$ \begin{eqnarray} \nabla_{\theta_0}l(\theta_0,\theta_1,\phi) &=& \nabla_{\theta_0} \sum_{\substack{i=1 \\ z^{(i)}=0}}^m -\frac{(y^{(i)}-\theta_0^Tx^{(i)})^2}{2\sigma^2}\\ &=& \sum_{\substack{i=1 \\ z^{(i)}=0}}^m \frac{(y^{(i)}-\theta_0^Tx^{(i)})}{\sigma^2}x^{(i)}\\ &=& \frac{1}{\sigma^2}X^T_0(\vec{y}_0-X_0\theta_0)\\ \end{eqnarray} $$</center>

where the design matrix $$X_0$$ and the corresponding labels $$\vec{y}_0$$ only contain the examples for which $$z$$ is $$0$$. Setting this to zero gives:

<center>$$ \theta_0 = (X^T_0X_0)^{-1}X_0^T\vec{y}_0 $$</center>

Similarly:

<center>$$ \theta_1 = (X^T_1X_1)^{-1}X_1^T\vec{y}_1 $$</center>

We have already shown in the lecture notes that for logistic regression:

<center>$$ \frac{\partial l}{\partial \phi_j} = \sum_{i=1}^m(z^{(i)}-g(\phi^Tx^{(i)}))x^{(i)}_j $$</center>

Therefore:

<center>$$ \frac{\partial^2 l}{\partial\phi_j\partial\phi_k} = -\sum_{i=1}^mg(\phi^Tx^{(i)})(1-g(\phi^Tx^{(i)}))x^{(i)}_jx^{(i)}_k $$</center>

The Hessian $$H$$ for $$l$$ with respect to $$\phi$$ is a matrix whose entry $$(i,j)$$ is given by $$\partial^2 l/\partial\phi_j\partial\phi_k$$.

(b) In the E-step we set $$w^{(i)}_j =Q_i(z^{(i)}) := p(z^{(i)}\vert x^{(i)})$$. In the M-step we maximize the following log likelihood:

<center>$$ l(\theta_1,\theta_2,\phi) =\sum_{i=1}^m\sum_{j=0}^{k=1}Q_i(z^{(i)}=j)\log\frac{p(y^{(i)}\vert x^{(i)},z^{(i)};\theta_0,\theta_1)p(z^{(i)}=j)\vert x^{(i)}}{Q_i(z^{(i)}=j)} $$</center>

Therefore:

<center>$$ \begin{eqnarray} \nabla_{\theta_0} l(\theta_0,\theta_1,\phi) &=& \nabla_{\theta_0} \sum_{\substack{i=1\\z^{(i)}=0} }^m -w_0^{(i)} \frac{(y^{(i)}-\theta_0^Tx^{(i)})^2}{2\sigma^2}\\ &=& \frac{1}{\sigma^2}X^T_0W_0(\vec{y}_0-X_0\theta_0) \end{eqnarray} $$</center>

where $$W_0 = diag(w_0^{(1)},...,w_0^{(m)})$$. Setting this to zero gives:

<center>$$ \theta_0 = (X_0^TW_0X_0)^{-1}X_0^TW_0\vec{y}_0 $$</center>

Similarly:

<center>$$ \theta_1 = (X_1^TW_1X_1)^{-1}X_1^TW_1\vec{y}_1 $$</center>

Let us now take the derivative with respect to $$\phi$$. Note that:

<center>$$ \begin{eqnarray} \frac{\partial l(\theta_1,\theta_2,\phi)}{\partial\phi_m} &=& \frac{\partial}{\partial\phi_m}\sum_{i=1}^m\sum_{j=0}^{k=1}w_j^{(i)}\log p(z^{(i)}\vert x^{(i)};\phi)\\ &=& \frac{\partial}{\partial\phi_m}\sum_{i=1}^mw_0^{(i)}\log g(\phi^Tx^{(i)}) + w_1^{(i)}\log (1-g(\phi^Tx^{(i)}))\\ &=& \frac{\partial}{\partial\phi_m}\sum_{i=1}^mw_0^{(i)}\log g(\phi^Tx^{(i)}) + (1-w_0^{(i)})\log (1-g(\phi^Tx^{(i)}))\\ &=& \sum_{i=1}(w^{(i)}_0-g(\phi^Tx^{(i)}))x_m^{(i)} \end{eqnarray} $$</center>

Hence:

<center>$$ \frac{\partial l(\theta_1,\theta_2,\phi)}{\partial\phi_m\partial\phi_n} = -\sum_{i=1}g(\phi^Tx^{(i)})(1-g(\phi^Tx^{(i)}))x_m^{(i)}x_n^{(i)} $$</center>

The Hessian $$H$$ for $$l$$ with respect to $$\phi$$ is a matrix whose entry $$(m,n)$$ is given by $$\partial^2 l/\partial\phi_m\partial\phi_n$$.

**2\. Factor Analysis and PCA**

(a) In the lecture notes we showed that:

<center>$$ \begin{bmatrix} z \\ x \end{bmatrix} \sim \mathcal{N}\left(\begin{bmatrix} \vec{0} \\ \mu \end{bmatrix}, \begin{bmatrix} I & \Lambda^T \\ \Lambda & \Lambda\Lambda^T + \Psi \end{bmatrix}\right) $$</center>

In this case $$\mu = \vec{0}$$, $$\Lambda=U$$ and $$\Psi=\sigma^2I$$. Therefore:

<center>$$ \begin{bmatrix} z \\ x \end{bmatrix} \sim \mathcal{N}\left(\vec{0}, \begin{bmatrix} I & U^T \\ U & UU^T + \sigma^2I\end{bmatrix}\right) $$</center>

Also $$z \vert x \sim \mathcal{N}(\mu_{z\vert x},\Sigma_{z\vert x})$$, where:

<center>$$ \begin{eqnarray} \mu_{z \vert x} &=& U^T(UU^T+\sigma^2I)^{-1}x\\ &=& (U^TU+\sigma^2I)^{-1}U^Tx\\ \Sigma_{z \vert x} &=& I - U^T(UU^T+\sigma^2I)^{-1}U\\ &=& I - (U^TU+\sigma^2I)^{-1}U^TU \end{eqnarray} $$</center>

where the second and fourth line follow from the identity given in the question.

(b) For the E-step we set $$Q_i(z^{(i)})=p(z^{(i)}\vert x^{(i)})$$. For the M-step we maximize the term:

<center>$$ f(\mu,\Lambda,\Psi) = \sum_{i=1}^m\int_{z^{(i)}}Q_i(z^{(i)})log\left(\frac{p(x^{(i)},z^{(i)})}{Q_i(z^{(i)})}\right)\\ $$</center>

Using the result for $$\Lambda$$ in the lecture notes, we have:

<center>$$ U = \sum_{i=1}^m x^{(i)}\mu_{z^{(i)} \vert x^{(i)}}^T\left(\sum_{i=1}^m\Sigma_{z^{(i)}\vert x^{(i)}}+\mu_{z^{(i)} \vert x^{(i)}}\mu_{z^{(i)} \vert x^{(i)}}^T\right)^{-1}\\ $$</center>

(c) As $$\sigma^2 \rightarrow 0$$, $$\Sigma_{z\vert x} = I - (U^TU+\sigma^2I)^{-1}U^TU \approx I-(U^TU)^{-1}U^TU=0$$. Also, $$\mu_{z \vert x} = (U^TU)^{-1}U^Tx$$. Define:

<center>$$ \begin{eqnarray} w &=& \begin{bmatrix}(U^TU)^{-1}U^Tx^{(1)} & . &. &. & (U^TU)^{-1}U^Tx^{(m)}\end{bmatrix}\\ &=& (U^TU)^{-1}XU \end{eqnarray} $$</center>

where $$X$$ is the design matrix.

Therefore, the update in the M-step is given by:

<center>$$ \begin{eqnarray} U &=& \sum_{i=1}^m x^{(i)}\mu_{z^{(i)} \vert x^{(i)}}^T\left(\sum_{i=1}^m\Sigma_{z^{(i)}\vert x^{(i)}}+\mu_{z^{(i)} \vert x^{(i)}}\mu_{z^{(i)} \vert x^{(i)}}^T\right)^{-1}\\ &=& \sum_{i=1}^m x^{(i)}\mu_{z^{(i)} \vert x^{(i)}}^T\left(\sum_{i=1}^m\mu_{z^{(i)} \vert x^{(i)}}\mu_{z^{(i)} \vert x^{(i)}}^T\right)^{-1}\\ &=& X^Tw(w^Tw)^{-1} \end{eqnarray} $$</center>

If the algorithm converges to $$U^*$$, then the value of $$U$$ will remain $$U^*$$ even after an update. So:

<center>$$ \begin{eqnarray} U^* &=& X^Tw(w^Tw)^{-1}\\ &=& X^T(U^*{^T}U^*)^{-1}XU^*(w^Tw)^{-1}\\ &=& (U^*{^T}U^*)^{-1}(w^Tw)^{-1}X^TXU^*\\ \lambda U^* &=& \Sigma U^* \end{eqnarray} $$</center>

where $$\lambda=(U^*{^T}U^*)(w^Tw)$$ and $$\Sigma=X^TX$$ is the covariance matrix.

**4\. Convergence of Policy Iteration**

(a) This can be proved in the following way:

<center>$$ \begin{eqnarray} V_1(s') &\leqslant& V_2(s') && \forall s'\in S\\ P_{s\pi(s)}(s')V_1(s') &\leqslant& P_{s\pi(s)}(s')V_2(s') && \forall s'\in S\\ \sum_{s'\in S}P_{s\pi(s)}(s')V_1(s') &\leqslant& \sum_{s'\in S}P_{s\pi(s)}(s')V_2(s')\\ R(s) + \gamma\sum_{s'\in S}P_{s\pi(s)}(s')V_1(s') &\leqslant& R(s) + \gamma\sum_{s'\in S}P_{s\pi(s)}(s')V_2(s')\\ B^\pi(V_1)(s) &\leqslant& B^\pi(V_2)(s) \end{eqnarray} $$</center>

where the second line follows from the fact that $$P_{s\pi(s)}(s')\geqslant 0$$ for all $$s'$$ and the fourth line from the fact that $$\gamma >0$$.

(b) This can be proved in the following way:

<center>$$ \begin{eqnarray} \vert\vert B^\pi(V)-V^\pi\vert\vert_\infty &=& \max_{s \in S}\left[R(s) + \gamma\sum_{s' \in S}P_{s\pi(s)}(s')V(s')-V^{\pi}(s)\right]\\ &\leqslant& \max_{s \in S}\left[R(s) + \gamma V(s)-V^{\pi}(s)\right]\\ &=& \max_{s \in S}\left[R(s) + \gamma V(s)-R(s)-\gamma\sum_{s' \in S}P_{s\pi(s)}(s')V^\pi(s')\right]\\ &\leqslant& \gamma \max_{s \in S}\left[V(s)-V^{\pi}(s)\right] \end{eqnarray} $$</center>

where the second and fourth line follow from the fact that for any $$\alpha,x \in \mathbb{R}^n$$ where $$\sum_{i}\alpha_i=1$$ and $$\alpha_i\geqslant 0$$:

<center>$$ \begin{eqnarray} \sum_{i}\alpha_ix_i &\leqslant& \sum_i \alpha_i\left[\max_ix_i\right]\\ &=& \left[\max_ix_i\right]\sum_i \alpha_i\\ &=& \max_ix_i \end{eqnarray} $$</center>

(c) Note that:

<center>$$ \begin{eqnarray} V^\pi(s) &=& R(s) + \gamma\sum_{s'\in S}P_{s\pi(s)}(s')V^{\pi}(s')\\ &\leqslant& R(s) + \max_{a}\gamma\sum_{s\in S}P_{sa}V^\pi(s')\\ &=& B^{\pi'}(V^\pi)(s) \end{eqnarray} $$</center>

From (a) it follows that:

<center>$$ B^{\pi'}(V^\pi)(s) \leqslant B^{\pi'}(B^{\pi'}(V^\pi))(s) $$</center>

Applying this repeatedly and using the answer to part (b) we get:

<center>$$ \begin{eqnarray} B^{\pi'}(B^{\pi'}(V^\pi))(s) &\leqslant& &.&.&.& &\leqslant& B^{\pi'}(B^{\pi'}(...B^{\pi'}(V^\pi)...))(s) = V^{\pi'}(s) \end{eqnarray} $$</center>

(d) We know that $$V^{\pi}(s)$$ is finite for any policy $$\pi$$. We also know that $$V(s)$$ improves monotonically after every iteration of the policy iteration algorithm. Therefore, $$V(s)$$ must converge to some finite value. Denote the policy at convergence by $$\pi'$$. As the algorithm has converged, another iteration of the policy iteration algorithm will again yield $$\pi'$$. Therefore:

<center>$$ V^{\pi'}(s) = R(s) + \max_{a \in A}\gamma\sum_{s'\in S}P_{sa}(s')V^{\pi'}(s') $$</center>

Hence, using the property of the optimal value function given in the question we have $$\pi'=\pi^*$$.
