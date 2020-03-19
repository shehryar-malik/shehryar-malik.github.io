---
layout: post
title: Support Vector Machines
permalink: blog/machine-learning/cs229-notes/support-vector-machines
categories: [Machine Learning, CS229 Notes]
---

Suppose that we have a training set $$\{(x^{(i)},y^{(i)}); i=1,...,m\}$$, where $$y^{(i)} \in \{-1,+1\}$$. Let us assume that the $$x^{(i)}$$’s are linearly separable, i.e. $$\exists$$ some hyperplane $$w^Tx+b=0$$ for appropriate choices of $$w$$ and $$b$$ such that $$% <![CDATA[ w^Tx^{(i)}+b < 0 %]]>$$ for $$y^{(i)}=-1$$ and $$w^Tx^{(i)}+b>0$$ for $$y^{(i)}=+1$$. Our goal is to find this hyperplane.

## 1\. Functional and Geometric Margins

We shall define the functional margin of this training set as follows:

<center>$$ \hat{\gamma}^{(i)} = y^{(i)}(w^Tx^{(i)}+b) $$</center>

Note that:

1.  A large functional margin indicates more confidence in a certain prediction.

2.  A positive functional margin indicates a correct prediction on a particular example.

3.  The functional margin of an entire training set is given by:

    <center>$$ \hat{\gamma} = \min_i \hat{\gamma}^{(i)} $$</center>

4.  The functional margin can be made arbitrarily large by multiplying it with a scalar. However, note that this does not change the predictions on the training examples (the predictions only depend on the sign of $$w^Tx^{(i)}+b$$).

To solve the problem discussed in last point above we shall introduce the concept of the geometric margin.

Suppose that we have a (hyper)plane in 3d space. Let $$P_0=(x_0, y_0, z_0)$$ be a known point on this plane. Therefore, the vector from the origin $$(0,0,0)$$ to this point is just $$% <![CDATA[ <x_0,y_0,z_0> %]]>$$. Suppose that we have an arbitrary point $$P=(x,y,z)$$ on the plane. The vector joining $$P$$ and $$P_0$$ is then given by:

<center>$$ \vec{P} - \vec{P_0} = <x-x_0,y-y_0,z-z_0> $$</center>

Note that this vector lies in the plane. Now let $$\hat{n}$$ be the normal (orthogonal) vector to the plane. Therefore:

<center>$$ \hat{n} \bullet (\vec{P}-\vec{P_0}) = 0 $$</center>

Or:

<center>$$ \hat{n} \bullet \vec{P}- \hat{n} \bullet \vec{P_0} = 0 $$</center>

Note that $$-\hat{n} \bullet \vec{P_0}$$ is just a number. Therefore, for the hyperplane $$w^Tx+b$$, $$w$$ represents the normal vector $$\hat{n}$$ while $$b$$ is $$-\hat{n}\bullet \vec{P_0}$$ and the point $$x$$ is $$P$$.

Suppose that we draw a vector perpendicular to the hyperplane to a point $$x^{(i)}$$ in the training set. The magnitude $$\gamma^{(i)}$$ of this vector is known as the geometric margin. Note that this vector is in the direction of the unit vector $$y^{(i)}\frac{w}{\vert w \vert}$$. The point at which this vector cuts the hyperplane is thus given by $$x^{(i)}-y^{(i)}\gamma^{(i)}\frac{w}{\vert w \vert}$$. Thus:

<center>$$ \begin{eqnarray} w^T(x^{(i)}-y^{(i)}\gamma^{(i)}\frac{w}{\vert w \vert})+b &=& 0\\ \gamma^{(i)} &=& y^{(i)}\frac{w^Tx+b}{\vert \vert w \vert \vert} \end{eqnarray} $$</center>

Hence, the geometric margin $$\gamma^{(i)}$$ is related to the functional margin $$\hat{\gamma}^{(i)}$$ by the following:

<center>$$ \gamma^{(i)} = \frac{\hat{\gamma}^{(i)}}{\vert\vert w \vert\vert} $$</center>

Note that the geometric margin is invariant to rescaling of the parameters.

We define the geometric margin of the entire training set to be:

<center>$$ \gamma = \min_i \gamma^{(i)} $$</center>

## 2\. The Optimal Margin Classifier

Our goal is to maximize the geometric margin. In that respect, consider the following optimization problem:

<center>$$ \begin{eqnarray} && max_{\gamma, w,b} \ \ \gamma \\ && \ \ \ \ s.t.\ \ \ \ y^{(i)}(w^Tx^{(i)}+b) \geqslant \gamma \ \ \ i=1,...,m\\ && \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \vert \vert w \vert \vert= 1 \end{eqnarray} $$</center>

Note that the condition $$\vert\vert w \vert\vert = 1$$ ensures that the functional margin is equal to the geometric margin. We may rewrite the above problem as follows:

<center>$$ \begin{eqnarray} && max_{\gamma,w,b} \ \ \frac{\hat{\gamma}}{\vert\vert w \vert\vert} \\ && \ \ \ \ s.t.\ \ \ \ y^{(i)}(w^Tx^{(i)}+b) \geqslant \hat{\gamma} \ \ \ i=1,...,m \end{eqnarray} $$</center>

Note that we can make $$\hat{\gamma}$$ equal to any arbitrary value by simply rescaling the parameters (without changing anything meaningful). Let us choose $$\hat{\gamma}=1$$. We may then pose the following optimization problem:

<center>$$ \begin{eqnarray} && min_{\gamma,w,b} \ \ \frac{1}{2}\vert\vert w \vert\vert^2 \\ && \ \ \ \ s.t.\ \ \ \ y^{(i)}(w^Tx^{(i)}+b) \geqslant 1 \ \ \ i=1,...,m \end{eqnarray} $$</center>

Note that we have essentially transformed a non-convex problem into a convex one. We shall now introduce the Lagrange duality that will help us solve this problem.

## 3\. Lagrange Duality

Suppose that we have the following optimization problem:

<center>$$ \begin{eqnarray} && min_{w} \ \ f(w) \\ && \ \ \ \ s.t.\ \ \ \ h_i(w)=0 \ \ \ i=1,...,l \end{eqnarray} $$</center>

We may solve this problem by defining the Lagrangian:

<center>$$ \mathcal{L}(w, \beta) = f(w) + \sum_{i=1}^l \beta_ih_i(w) $$</center>

and solving for:

<center>$$ \begin{eqnarray} \frac{\partial \mathcal{L}(w,\beta)}{\partial w} &=& 0\\ \frac{\partial \mathcal{L}(w,\beta)}{\partial \beta_i} &=& 0 \end{eqnarray} $$</center>

For problems of the following form:

<center>$$ \begin{eqnarray} && min_{w} \ \ f(w) \\ && \ \ \ \ s.t.\ \ \ \ h_i(w)=0 \ \ \ i=1,...,l\\ && \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ g_i(w) \leqslant 0 \ \ \ j=1,...,k \end{eqnarray} $$</center>

we may define the generalized Lagrangian:

<center>$$ \mathcal{L}(w, \alpha, \beta) = f(w) + \sum_{\substack{i=1\\ \alpha_i \geqslant 0}}^k \alpha_i g_i(w) + \sum_{i=1}^l \beta_ih_i(w) $$</center>

Consider the following:

<center>$$ \theta_P(w) = \max_{\alpha, \beta} \mathcal{L}(w, \alpha, \beta) = \max_{\alpha, \beta} \left( f(w) + \sum_{\substack{i=1\\ \alpha_i \geqslant 0}}^k \alpha_i g_i(w) + \sum_{i=1}^l \beta_ih_i(w) \right) $$</center>

Note that if the constraints $$\alpha_i\geqslant 0$$, $$g_i(w) \leqslant 0$$ and $$h_i(w) = 0$$ are satisfied then $$\theta_p$$ is just equal to $$f(w)$$. Otherwise, it equals $$+\infty$$.

Let us define the primal problem to be $$p^*=\min_w\theta_P(w)$$. Note that this is the same problem we were trying to solve initially (i.e. $$\min_w f(w)$$ subject to some constraints).

Alternatively, consider the following problem called as the dual problem:

<center>$$ d^* = \max_{\alpha,\beta} \theta_D(\alpha,\beta) = \max_{\alpha,\beta} \min_w \left( f(w) + \sum_{\substack{i=1\\ \alpha_i \geqslant 0}}^k \alpha_i g_i(w) + \sum_{i=1}^l \beta_ih_i(w) \right) $$</center>

It follows from the fact that the $$\min\max$$ of a function is always greater than or equal to the $$\max\min$$ that:

<center>$$ d^* \leqslant p^* $$</center>

It can be shown that if the $$h_i$$’s are affine, the $$\alpha_i$$’s and $$g_i$$’s convex and the constraints (strictly) feasible i.e. $$% <![CDATA[ g_i(w) <0 %]]>$$ $$\forall$$ $$i$$ then there must exist $$w^*$$, $$\alpha^*$$, $$\beta^*$$ such that $$w^*$$ is the solution to the primal problem and $$\alpha^*$$, $$\beta^*$$ are the solutions to the dual problem and that:

<center>$$ d^* = p^* $$</center>

Moreover the following Karush-Kuhn-Tucker (KKT) conditions are also satisfied:

<center>$$ \begin{eqnarray} \frac{\partial\mathcal{L}(w^*,\alpha^*,\beta^*)}{\partial w_i} &=& 0 \ \ \ i=1,..,n\\ \frac{\partial\mathcal{L}(w^*,\alpha^*,\beta^*)}{\partial \beta_i} &=& 0 \ \ \ i=1,..,l\\ \alpha_i^*g_i(w^*) &=& 0 \ \ \ i=1,..,k\\ g_i(w^*) &\leqslant& 0 \ \ \ i=1,..,k\\ \alpha_i^* &\geqslant& 0 \ \ \ i=1,..,k \end{eqnarray} $$</center>

The third equation above is known as the KKT dual complementarity equation, and essentially constrains at least one of $$\alpha_i$$ and $$g_i$$ to be zero.

## 4\. Optimal Margin Classifiers

Let us return to the following optimization problem:

<center>$$ \begin{eqnarray} && min_{\gamma,w,b} \ \ \frac{1}{2}\vert\vert w \vert\vert^2 \\ && \ \ \ \ s.t.\ \ \ \ -y^{(i)}(w^Tx^{(i)}+b) + 1 \leqslant 0 \ \ \ i=1,...,m \end{eqnarray} $$</center>

Note that the constraint $$g(x) = -y^{(i)}(w^Tx^{(i)}+b) + 1$$ is only $$0$$ when the functional margin of $$x^{(i)}$$ is $$1$$. Hence, because of the KKT dual complementarity equation $$\alpha_i > 0$$ only for points that have a functional margin equal to $$1$$. These are known as the support vectors. Note that these points must be the ones that are closest to the hyperplane.

We may formulate a Lagrangian for the above problem:

<center>$$ \mathcal{L}(w, b, \alpha) = \frac{1}{2}\vert\vert w \vert\vert^2 +\sum_{\substack{i=1\\ \alpha_i \geqslant 0}}^{m}\alpha_i\left(-y^{(i)}(w^Tx^{(i)}+b) + 1\right) $$</center>

Therefore:

<center>$$ \begin{eqnarray} \frac{\partial\mathcal{L}(w, b, \alpha)}{\partial w} &=& 0\\ w &=& \sum_{\substack{i=1\\ \alpha_i \geqslant 0}}^{m} \alpha_iy^{(i)}x^{(i)} \end{eqnarray} $$</center>

And:

<center>\begin{eqnarray} \frac{\partial\mathcal{L}(w, b, \alpha)}{\partial b} &=& 0\\ \sum_{\substack{i=1\\ \alpha_i \geqslant 0}}^{m} \alpha_iy^{(i)}&=& 0 \end{eqnarray}</center>

Substituting $$w$$ into the $$\mathcal{L}(\gamma,w,b)$$ gives:

<center>\begin{eqnarray} \mathcal{L}(w, b, \alpha) &=& \frac{1}{2}\vert\vert \sum_{\substack{i=1\\ \alpha_i \geqslant 0}}^{m} \alpha_iy^{(i)}x^{(i)} \vert\vert^2 +\sum_{\substack{i=1\\ \alpha_i \geqslant 0}}^{m}\alpha_i\left(-y^{(i)}\left(\left(\sum_{\substack{j=1\\ \alpha_j \geqslant 0}}^{m} \alpha_jy^{(j)}x^{(j)}\right)^Tx^{(i)}+b\right) + 1\right)\\ &=& \frac{1}{2} \left(\sum_{\substack{i=1\\ \alpha_i \geqslant 0}}^{m} \alpha_iy^{(i)}x^{(i)}\right)^{T}\left(\sum_{\substack{i=1\\ \alpha_i \geqslant 0}}^{m} \alpha_iy^{(i)}(x^{(i)})\right) +\sum_{\substack{i=1\\ \alpha_i \geqslant 0}}^{m}\alpha_i\left(-y^{(i)}\left(\left(\sum_{\substack{j=1\\ \alpha_j\geqslant 0}}^{m} \alpha_jy^{(j)}(x^{(j)})^Tx^{(i)}\right)+b\right) + 1\right)\\ &=&\frac{1}{2} \sum_{\substack{i=1\\ \alpha_i \geqslant 0}}^{m}\sum_{\substack{j=1\\ \alpha_j \geqslant 0}}^{m} \alpha_i\alpha_jy^{(i)}y^{(j)}(x^{(j)})^Tx^{(i)} -\sum_{\substack{i=1\\ \alpha_i \geqslant 0}}^{m}\sum_{\substack{j=1\\ \alpha_j \geqslant 0}}^{m} \alpha_i\alpha_jy^{(i)}y^{(j)}(x^{(j)})^Tx^{(i)}+\sum_{\substack{i=1\\ \alpha_i \geqslant 0}}^{m}\alpha_iy^{(i)}b + \sum_{\substack{i=1\\ \alpha_i \geqslant 0}}^{m}\alpha_i\\ &=& \sum_{\substack{i=1\\ \alpha_i \geqslant 0}}^{m}\alpha_i - \frac{1}{2} \sum_{\substack{i=1\\ \alpha_i \geqslant 0}}^{m}\sum_{\substack{j=1\\ \alpha_j \geqslant 0}}^{m} \alpha_i\alpha_jy^{(i)}y^{(j)}(x^{(j)})^Tx^{(i)} \end{eqnarray}</center>

Note that in the last step we used the fact that $$\sum_{\substack{i=1\\ \alpha_i \geqslant 0}}^{m}\alpha_iy^{(i)} = 0$$. Hence, we may state the following dual optimization problem:

<center>$$ \max_{\alpha} W(\alpha) = \sum_{\substack{i=1\\ \alpha_i \geqslant 0}}^{m}\alpha_i - \frac{1}{2} \sum_{\substack{i=1\\ \alpha_i \geqslant 0}}^{m}\sum_{\substack{j=1\\ \alpha_j \geqslant 0}}^{m} \alpha_i\alpha_jy^{(i)}y^{(j)}(x^{(j)})^Tx^{(i)}\\ \begin{eqnarray} s.t. \ \ \ \ \sum_{i=1}^{m}\alpha_iy^{(i)} &=& 0 \ \ \ \ i=1,...,m\\ \alpha_i &\geqslant& 0\ \ \ \ i=1,...,m \end{eqnarray} $$</center>

Once we have solved for $$\alpha$$ we can find $$w$$ using the equation we derived earlier:

<center>$$ w = \sum_{\substack{i=1\\ \alpha_i \geqslant 0}}^{m} \alpha_iy^{(i)}x^{(i)} $$</center>

To solve for $$b$$ consider two points $$x_1$$ and $$x_2$$ whose functional margin is equal to $$1$$ such that $$y_1 = 1$$ and $$y_2=-1$$. Therefore $$w^Tx_1 + b=1$$ and $$w^Tx_1 + b=-1$$. Hence:

<center>$$ \begin{eqnarray} 0 &=& (w^Tx_1 + b) + (w^Tx_2 + b)\\ b &=& -\frac{w^Tx_1+w^Tx_2}{2}\\ b &=& -\frac{\min_{i:y^{(i)}=1}w^Tx^{(i)}+\max_{i:y^{(i)}=-1}w^Tx^{(i)}}{2} \end{eqnarray} $$</center>

Note that in the last step we used the fact that all points have a functional margin of at least $$1$$, i.e. $$1$$ is the minimum functional margin a point can have.

Also, note that the equation (that we will use to make a prediction on an input $$x$$):

<center>$$ \begin{eqnarray} w^Tx+b &=& \left(\sum_{\substack{i=1\\ \alpha_i \geqslant 0}}^{m} \alpha_iy^{(i)}x^{(i)}\right)^T x + b\\ &=& \sum_{\substack{i=1\\ \alpha_i \geqslant 0}}^{m} \alpha_iy^{(i)}(x^{(i)})^Tx + b \end{eqnarray} $$</center>

only depends on the inner product between the training examples $$x^{(i)}$$ and the input $$x$$. Moreover most of the $$\alpha_i$$’s, as discussed previously, will be $$0$$.

## 5\. Kernels

Suppose that we map the input vector $$x$$ to a higher (possibly infinite) dimensional vector space and input the resulting vector $$\phi(x)$$ to an SVM. Consequently we would replace $$x$$ in the SVM formulation with $$\phi(x)$$. Let us define $$K(x,z)$$$$=$$$$\phi(x)^T\phi(z)$$, where $$K(x,z)$$ is called as the kernel. Note that this implies that inner product $$\phi(x)^T\phi(z)$$ can be calculated using only $$x$$ and $$z$$, i.e. without explicitly finding $$\phi(x)$$ or $$\phi(z)$$.

Let us define the Kernel matrix $$K$$ such that $$K_{ij} = K(x^{(i)},x^{(j)})$$. Then for $$K$$ to be a valid kernel $$K_{ij}=K_{ji}$$ because $$\phi(x^{(i)})\phi(x^{(j)})$$ $$=$$ $$\phi(x^{(j)})\phi(x^{(i)})$$. Also, suppose that $$z$$ is any vector. Then:

<center>$$ \begin{eqnarray} z^TKz &=& \sum_i\sum_j z_iK_{ij}z_j\\ &=& \sum_i\sum_j z_i\phi(x^{(i)})^T\phi(x^{(j)})z_j\\ &=& \sum_i\sum_j z_i\left(\sum_k\phi_k(x^{(i)})\phi_k(x^{(j)})\right)z_j\\ &=& \sum_k\sum_i\sum_j z_i\phi_k(x^{(i)})\phi_k(x^{(j)})z_j\\ &=& \sum_k\left(\sum_i z_i\phi_k(x^{(i)})\right)\left(\sum_j\phi_k(x^{(j)})z_j\right)\\ &=& \sum_k\left(\sum_i z_i\phi_k(x^{(i)})\right)^2\\ &\geqslant& 0 \end{eqnarray} $$</center>

which shows that $$K$$ is positive semidefinite. It turns out that for $$K$$ to be a valid (Mercer) kernel it is sufficient that it is symmetric and positive semidefinite.

## 6\. Regularization and the Non-Separable Case

Consider the following problem:

<center>\begin{eqnarray} && min_{\gamma,w,b} \ \ \frac{1}{2}\vert\vert w \vert\vert^2 + C\sum_{i=1}^m\xi_i\\ && \ \ \ s.t. \ \ \ \ \ y^{(i)}(w^Tx^{(i)}+b) \geqslant 1-\xi_i \ \ \ i=1,...,m\\ && \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \xi_i \geqslant 0 \ \ \ i=1,...,m \end{eqnarray}</center>

Note that the regularization term allows, but penalizes, points to have a functional margin of less than $$1$$. The Lagrangian is thus given by:

<center>$$ \mathcal{L}(w, b, \xi, \alpha, r) = \frac{1}{2}\vert\vert w \vert\vert^2 + C\sum_{i=1}^m\xi_i +\sum_{\substack{i=1\\ \alpha_i \geqslant 0\\ \xi_i \geqslant 0}}^{m}\alpha_i\left(-y^{(i)}(w^Tx^{(i)}+b) + 1 - \xi_i\right)-\sum_{\substack{i=1\\ \xi_i \geqslant 0}}^{m}r_i\xi_i $$</center>

Therefore:

<center>\begin{eqnarray} \frac{\partial\mathcal{L}(w, b, \xi, \alpha, r)}{\partial w} &=& 0\\ w &=& \sum_{\substack{i=1\\ \alpha_i \geqslant 0}}^{m} \alpha_iy^{(i)}x^{(i)} \end{eqnarray}</center>

And:

<center>$$ \begin{eqnarray} \frac{\partial\mathcal{L}(w, b, \xi, \alpha, r)}{\partial b} &=& 0\\ \sum_{\substack{i=1\\ \alpha_i \geqslant 0}}^{m} \alpha_iy^{(i)}&=& 0 \end{eqnarray} $$</center>

Also:

<center>$$ \begin{eqnarray} \frac{\partial\mathcal{L}(w, b, \xi, \alpha, r)}{\partial \xi_i} &=& 0\\ r_i &=& C - \alpha_i \end{eqnarray} $$</center>

Note that the last equation implies that $$0\leqslant \alpha_i\leqslant C$$ because $$r_i\geqslant 0$$.

Substituting $$w$$ in $$\mathcal{L}(w, b, \xi, \alpha, r)$$ and simplifying (see Section 4 above) gives:

<center>\begin{eqnarray} L(w,b,\xi,\alpha,r) &=& \sum_{\substack{i=1\\ \alpha_i \geqslant 0}}^{m}(1-\xi_i)\alpha_i +C\sum_{i=1}^m\xi_i -\frac{1}{2} \sum_{\substack{i=1\\ \alpha_i \geqslant 0}}^{m}\sum_{\substack{j=1\\ \alpha_j \geqslant 0}}^{m} \alpha_i\alpha_jy^{(i)}y^{(j)}(x^{(j)})^Tx^{(i)}-\sum_{\substack{i=1\\ r_i \geqslant 0}}^{m}r_i\xi_i\\ &=& \sum_{\substack{i=1\\ \alpha_i \geqslant 0}}^{m}\alpha_i -\frac{1}{2} \sum_{\substack{i=1\\ \alpha_i \geqslant 0}}^{m}\sum_{\substack{j=1\\ \alpha_j \geqslant 0}}^{m} \alpha_i\alpha_jy^{(i)}y^{(j)}(x^{(j)})^Tx^{(i)}+\sum_{\substack{i=1\\ \alpha_i \geqslant 0}}^{m}(C-\alpha_i)\xi_i -\sum_{\substack{i=1\\ r_i \geqslant 0}}^{m}r_i\xi_i\\ &=& \sum_{\substack{i=1\\ \alpha_i \geqslant 0}}^{m}\alpha_i -\frac{1}{2} \sum_{\substack{i=1\\ \alpha_i \geqslant 0}}^{m}\sum_{\substack{j=1\\ \alpha_j \geqslant 0}}^{m} \alpha_i\alpha_jy^{(i)}y^{(j)}(x^{(j)})^Tx^{(i)}+\sum_{\substack{i=1\\ \alpha_i \geqslant 0}}^{m}(C-\alpha_i)\xi_i -\sum_{\substack{i=1\\ \alpha_i \geqslant 0}}^{m}(C-\alpha_i)\xi_i\\ &=& \sum_{\substack{i=1\\ \alpha_i \geqslant 0}}^{m}\alpha_i -\frac{1}{2} \sum_{\substack{i=1\\ \alpha_i \geqslant 0}}^{m}\sum_{\substack{j=1\\ \alpha_j \geqslant 0}}^{m} \alpha_i\alpha_jy^{(i)}y^{(j)}(x^{(j)})^Tx^{(i)} \end{eqnarray}</center>

We formulate the dual problem as follows:

<center>$$ \max_{\alpha} W(\alpha) = \sum_{\substack{i=1\\ \alpha_i \geqslant 0}}^{m}\alpha_i - \frac{1}{2} \sum_{\substack{i=1\\ \alpha_i \geqslant 0}}^{m}\sum_{\substack{j=1\\ \alpha_j \geqslant 0}}^{m} \alpha_i\alpha_jy^{(i)}y^{(j)}(x^{(j)})^Tx^{(i)}\\ \begin{eqnarray} s.t. \ \ \ \ \ \ \ 0 \leqslant \alpha_i &\leqslant& C \ \ \ \ i=1,...,m\\ \sum_{i=1}^{m}\alpha_iy^{(i)} &=& 0 \ \ \ \ \ i=1,...,m \end{eqnarray} $$</center>

Also, the KKT dual complementarity equations are now given as:

<center>$$ \begin{eqnarray} \alpha_i &=& 0 &\Rightarrow& y^{(i)}(w^Tx^{(i)}+b) &\geqslant& 1 \\ \alpha_i &=& C &\Rightarrow& y^{(i)}(w^Tx^{(i)}+b) &\leqslant& 1 \\ 0 < \alpha_i &<& C &\Rightarrow& y^{(i)}(w^Tx^{(i)}+b) &=& 1 \\ \end{eqnarray} $$</center>

## 7\. Coordinate Ascent

Consider the following optimization problem:

<center>$$ \max_{\alpha_1,\alpha_2,...,\alpha_n} W(\alpha_1,\alpha_2,...,\alpha_n) $$</center>

The coordinate ascent algorithm does the following:

Loop until convergence  
{  
	$$\;\;$$For $$i =1,...,m$$ do:  
    $$\;\;\;\;$$$$\max_{\hat{\alpha}_i} W(\alpha_1,\alpha_2,...,\hat{\alpha}_i,...,\alpha_m)$$  
}

i.e. at each iteration, $$W(\alpha_1,\alpha_2,...,\alpha_n)$$ is optimized with respect to only one parameter while all others are held constant.

## 8\. Sequential Minimal Optimization

We have the following problem:

<center>$$ \max_{\alpha} W(\alpha) = \sum_{\substack{i=1\\ \alpha_i \geqslant 0}}^{m}\alpha_i - \frac{1}{2} \sum_{\substack{i=1\\ \alpha_i \geqslant 0}}^{m}\sum_{\substack{j=1\\ \alpha_j \geqslant 0}}^{m} \alpha_i\alpha_jy^{(i)}y^{(j)}(x^{(j)})^Tx^{(i)}\\ \begin{eqnarray} s.t. \ \ \ \ \ \ \ 0 \leqslant \alpha_i &\leqslant& C \ \ \ \ i=1,...,m\\ \sum_{i=1}^{m}\alpha_iy^{(i)} &=& 0 \ \ \ \ \ i=1,...,m \end{eqnarray} $$</center>

Note that we cannot use the coordinate ascent algorithm, as presented above, here because of the second constraint which essentially states that any one $$\alpha_i$$ is exactly determined by the other $$\alpha_i$$’s (for $$i=1,...,m$$). We, thus, modify the coordinate ascent algorithm as follows:

Loop until convergence  
{  
	$$\;\;$$For some $$i,j=1,...,m$$ do:  
    $$\;\;\;\;$$$$\max_{\hat{\alpha}_i\hat{\alpha}_j} W(\alpha_1,...,\hat{\alpha}_i,...,\hat{\alpha}_j,...,\alpha_m)$$  
}

i.e. we optimize $$W(\alpha_1,...,\alpha_m)$$ with respect to two parameters at each iteration. Therefore, at each iteration we have:

<center>$$ \begin{eqnarray} y^{(i)}\alpha_i+y^{(j)}\alpha_j &=& \sum_{\substack{k=1\\ k \neq i,j}}^m y^{(k)}\alpha_k\\ y^{(i)}\alpha_i+y^{(j)}\alpha_j &=& \zeta\\ \alpha_i &=& y^{(i)}(\zeta-y^{(j)}\alpha_j) \end{eqnarray} $$</center>

where $$\zeta$$ is a constant. Note that at each iteration all $$\alpha_k$$’s where $$k=1,...,m$$ and $$k \neq i,j$$ are constant. Note also that $$\alpha_i$$ and $$\alpha_j$$ must be between $$0$$ and $$C$$. Suppose that $$L$$ and $$H$$ are the lower and upper bounds on $$\alpha_j$$ respectively. We may rewrite $$W(\alpha_1,...,\hat{\alpha}_i,...,\hat{\alpha}_j,...,\alpha_m)$$ as $$W(\alpha_1,...,y^{(i)}$$$$(\zeta-y^{(j)}\alpha_j),...,\hat{\alpha}_j,...,\alpha_m)$$. Note that this is just a quadratic equation in $$\hat{\alpha_j}$$. Setting its derivative to zero will yield the optimal value for $$\alpha_j$$ which we shall represent by $$\alpha_j^{unclipped}$$. As $$\alpha_j$$ is constrained to be between $$L$$ and $$H$$, we shall clip its value as follows:

<center>$$ \alpha_j^{clipped} = \begin{cases} H, & \text{if} & &\alpha_j^{unclipped} &>& H\\ \alpha_j^{unclipped} & \text{if} & L\leqslant &\alpha_j^{unclipped}&\leqslant& H\\ L, & \text{if} & & \alpha_j^{unclipped} &<& L\end{cases} $$</center>

The convergence of this algorithm can be tested using the KKT dual complementarity conditions.
