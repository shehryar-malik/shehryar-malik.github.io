---
layout: post
title: Matrix Derivatives
permalink: blog/mathematics/matrix-derivatives
categories: [Mathematics]
---

Let $$A \in \mathbb{R}^{m \times n}$$ and let $$f$$ be a mapping from $$\mathbb{R}^{m \times n}$$ to $$\mathbb{R}$$. Then we define the Jacobian matrix as follows:

<center>$$ J = \nabla_A f(A) = \begin{bmatrix} \frac{\partial f(A)}{\partial A_{11}} & \frac{\partial f(A)}{\partial A_{12}} &.&.&.& \frac{\partial f(A)}{\partial A_{1n}} \\ \frac{\partial f(A)}{\partial A_{21}} & \frac{\partial f(A)}{\partial A_{22}} &.&.&.& \frac{\partial f(A)}{\partial A_{2n}} \\ .&.&.&.&.&. \\ .&.&.&.&.&. \\ \frac{\partial f(A)}{\partial A_{m1}} & \frac{\partial f(A)}{\partial A_{m2}} &.&.&.& \frac{\partial f(A)}{\partial A_{mn}}\end{bmatrix} $$</center>
## 1\. The Trace Operator

**Definition.** Let $$A$$ be an $$n \times n$$ matrix. The trace of $$A$$ is given by:

<center>$$ \text{tr}(A) = \sum_{i=1}^n A_{ii} $$</center>
**1.A** Let $$A \in \mathbb{R}^{n \times n}$$:

$$\text{tr}(A^T) = \sum_{i=1}^n (A^T)_{ii} = \sum_{i=1}^n A_{ii} = \text{tr}(A)$$

**1.B** Let $$A \in \mathbb{R}^{n \times n}$$and $$B\in \mathbb{R}^{n \times n}$$

$$\text{tr}(A+B) = \sum_{i=1}^n (A+B)_{ii} = \sum_{i=1}^n (A_{ii}+B_{ii}) = \text{tr}(A)+\text{tr}(B)$$

**1.C** Let $$A \in \mathbb{R}^{n \times n}$$ and $$a \in \mathbb{R}$$:

$$\text{tr}(aA) = \sum_{i=1}^n (aA)_{ii}= a\sum_{i=1}^n A_{ii} = a\text{tr}(A)$$

**1.D** Let $$A \in \mathbb{R}^{n \times m}$$ and $$B\in \mathbb{R}^{m \times n}$$:

$$\text{tr}(AB) = \sum_{i=1}^n (AB)_{ii} = \sum_{i=1}^n \sum_{j=1}^m A_{ij}B_{ji} = \sum_{j=1}^m \sum_{i=1}^n B_{ji}A_{ij} = \sum_{j=1}^m(BA)_{jj}=\text{tr}(BA)$$

## 2\. Derivative of a Scalar With Respect to a Vector

**2.A** Let $$\mu \in \mathbb{R}^n$$ and let $$A \in \mathbb{R}^{n \times n}$$. Then $$\nabla_\mu \mu^TA\mu =$$.$$A\mu + A^T\mu$$

<center>$$ \begin{eqnarray} \nabla_\mu \mu^TA\mu &=& \nabla_\mu \text{tr}(\mu^TA\mu) \\ &=& \nabla_x \text{tr}(x^TA\mu) + \nabla_x \text{tr}(\mu^TAx)\\ &=& A\mu + A^T\mu \end{eqnarray} $$</center>
The first step takes advantage of the fact that the trace of a real number is just that real number. Note that $$\mu^TA\mu$$ is just a scalar. The rest of the derivation is similar to that in **3.B**.

## 3\. Derivative of a Scalar With Respect To A Matrix

**3.A** Let $$A \in \mathbb{R}^{n \times m}$$ and $$B\in \mathbb{R}^{m \times n}$$. Then $$\nabla_A \text{tr}(AB)= B^T$$.

<center>$$ \begin{eqnarray} \frac{\partial}{\partial{A}_{lm}} \text{tr}(AB) &=& \frac{\partial}{\partial{A}_{lm}}\sum_{i=1}^n\sum_{j=1}^m A_{ij}B_{ji}\\ &=& B_{ml} \end{eqnarray} $$</center>
**3.B** Let $$ABA^TC$$ be a square matrix. Then $$\nabla_A \text{tr}(ABA^TC) = C^TAB^T + CAB$$.

<center>$$ \begin{eqnarray} \nabla_A \text{tr}(ABA^TC) &=& \nabla_{X} \text{tr}(XBA^TC) + \nabla_{X} \text{tr}(ABX^TC)\\ &=& \nabla_{X} \text{tr}(XBA^TC) + \left(\nabla_{X} \text{tr}(ABXC)\right)^T\\ &=& \nabla_{X} \text{tr}(XBA^TC) + \left(\nabla_{X} \text{tr}(XBAB)\right)^T\\ &=& C^TAB^T + CAB \end{eqnarray} $$</center>
In the first step, we have applied the product rule. Note that we essentially take the derivative with respect to each $$A$$ separately (think of the two $$A$$’s as different variables). For coherence, we have replaced the $$A$$ we are taking the derivative with respect to with $$X$$. In the second step we have made use of **3.C**. The third step uses **1.D** and the final step makes use of **3.A**.

**3.C** Let $$A \in \mathbb{R}^{m \times n}$$ and let $$f$$ be a mapping from $$\mathbb{R}^{m \times n}$$ to $$\mathbb{R}$$. Then $$\nabla_{A^T} f(A) = \left(\nabla_A f(A)\right)^T$$.

<center>$$ \begin{eqnarray} \nabla_{A^T} f(A) &=& \begin{bmatrix} \frac{\partial f(A)}{\partial (A^T)_{11}} &.& \frac{\partial f(A)}{\partial (A^T)_{1m}} \\ .&.&. \\ \frac{\partial f(A)}{\partial (A^T)_{n1}} &.&\frac{\partial f(A)}{\partial (A^T)_{nm}}\end{bmatrix} &=&\begin{bmatrix} \frac{\partial f(A)}{\partial A_{11}} &.& \frac{\partial f(A)}{\partial A_{m1}} \\ .&.&. \\ \frac{\partial f(A)}{\partial A_{1n}} &.&\frac{\partial f(A)}{\partial A_{mn}}\end{bmatrix}\\ &=& \begin{bmatrix} \frac{\partial f(A)}{\partial A_{11}} &.& \frac{\partial f(A)}{\partial A_{1n}} \\ .&.&. \\ \frac{\partial f(A)}{\partial A_{m1}} &.&\frac{\partial f(A)}{\partial A_{mn}}\end{bmatrix}^T &=& \nabla_A f(A) \end{eqnarray} $$</center>
Note that we have used the Jacobian matrix as defined earlier.

**3.D** Let $$A$$ be some square, non-singular matrix. Then $$\nabla_A \vert A \vert = \vert A \vert (A^{-1})^T$$. This follows from the fact that $$A^{-1}=(A')^T/\vert A \vert$$ where $$A'$$ is the matrix whose $$(i,j)$$ element is $$(-1)^{i+j}$$ times the determinant of the square matrix resulting from deleting the $$i^{th}$$ row and $$j^{th}$$ column of $$A$$. Also, the standard definition for the determinant is: $$\vert A \vert =\sum_{j}A_{ij}A'_{ij}$$ for any $$i$$. Since $$A'_{ij}$$ does not depend on the $$(i,j)$$ element of $$A$$ because of the way it was defined above, we have $$\nabla_A\vert A \vert = A' = \vert A \vert (A^{-1})^T$$.
