---
layout: post
title: K-Means Clustering Algorithm
permalink: blog/machine-learning/cs229-notes/k-means-clustering-algorithm
categories: [Machine Learning, CS229 Notes]
---

Given a training set $$S = \{x^{(1)}, x^{(2)}, ... , x^{(m)}\}$$, where $$x^{(i)} \in \mathbb{R}^n$$, the goal of the k-means clustering algorithm is to group this data into $$k$$ cohesive clusters. Note that there are no labels $$y^{(i)}$$, and consequently this is said to be an unsupervised learning problem.

The k-Means Clustering Algorithm does the following:

1.  Initialize $$k$$ cluster centroids $$\mu_1, \mu_2,...,\mu_k \in \mathbb{R}^n$$ randomly.

2.  Repeat until convergence  
    {  

    1.  For every $$i$$, set:  

        <center>$$c^{(i)} = \underset{j}{argmin} \vert\vert x^{(i)}-\mu_{j}\vert\vert$$</center>

    2.  For every $$j$$, set:  

        <center>$$\mu_j = \frac{\sum_{i=1}^m 1\{j=c^{(i)}\}x^{(i)}}{\sum_{i=1}^m1\{j=x^{(i)}\}}$$</center>

    }

Note that we are essentially minimizing the following distortion function:

<center>$$ J(\mu,c) = \sum_{i=1}^m \vert\vert x^{(i)}-\mu_{c^{(i)}} \vert\vert $$</center>

first with respect to $$c$$ while holding $$\mu$$ constant and then with respect to $$\mu$$ while holding $$c$$ constant. Note that is just coordinate descent on $$J$$. Note, that while this implies that $$J$$ will always monotonically decrease and so must converge to an optimum, the distortion function is non-convex and hence this might be a local optimum rather than the global optimum.