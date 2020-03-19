---
layout: post
title: Reinforcement Learning - III
permalink: blog/machine-learning/cs229-notes/reinforcement-learning-III
categories: [Machine Learning, CS229 Notes]
---

In this post we shall talk about (direct) policy search algorithms.

Let $$\Pi$$ be a set of policies. Our goal it to search for the best policy $$\pi \in \Pi$$. Note that this is analogous to the supervised learning setting where we define a set of hypothesis $$\mathcal{H}$$ and searched for the best hypothesis $$\mathcal{h} \in \mathcal{H}$$ (see [this]({{site.baseurl}}{%post_url /blog/machine-learning/cs229-notes/2018-09-22-learning-theory%})).

Let us define a stochastic policy as a function $$\Pi : \mathcal{S} \times \mathcal{A} \mapsto \mathbb{R}​$$ such that $$\pi(s,a)​$$ gives the probability of taking an action $$a​$$ in state $$s​$$. Consequently, note that $$\sum_{a\in\mathcal{A}} \pi(s,a)=1​$$ and $$\pi(s,a)>=0 \ \forall a \in \mathcal{A}​$$. As an example suppose that we have a two action MDP. We may then define:

<center>$$ \begin{eqnarray} \pi_{\theta}(s,a_1) &=& \frac{1}{1+\exp(-\theta^Ts)}\\ \pi_{\theta}(s,a_2) &=& 1-\frac{1}{1+\exp(-\theta^Ts)} \end{eqnarray} $$</center>

where $$\theta$$ is some parameter to be learned. Or, if we have more than two possible actions, we may choose:

<center>$$ \pi_\theta(s,a_i) = \frac{\exp(\theta_i^Ts)}{\sum_j \exp(\theta_j^Ts)} $$</center>

Our goal is to maximize the expected payoff, i.e. find:

<center>$$ \max_{\theta} \left[\mathbb{E}[R(s_0,a_0)+...+R(s_T,a_T)]\right] $$</center>

We shall refer to the term $$R(s_0,a_0)+...+R(s_T,a_T)$$ as payoff. Note that:

<center>$$ \begin{eqnarray} \max_\theta \left[\mathbb{E}[\text{payoff}]\right] &=& \max_\theta \left[ \sum_{(s_0a_0),...,(s_Ta_T)} P\left((s_0,a_0),...,(s_T,a_T)\right)[R(s_0,a_0)+...+R(s_T,a_T)]\right]\\ &=& \max_\theta \left[\sum_{(s_0a_0),...,(s_Ta_T)}P(s_0)\pi_\theta(s_0,a_0)P_{s_0a_0}(s_1)\pi_\theta(s_1,a_1)...P_{s_{T-1}a_{T-1}}(s_T)\pi_\theta(s_T,a_T) [R(s_0,a_0)+...+R(s_T,a_T)]\right] \end{eqnarray} $$</center>

where the first line follows from the definition of expectation i.e. $$\mathbb{E}[A] = \sum_{a\in A}P(a)*a$$ (note that $$P(R(s_0,a_0)+...+R(s_T,a_T))$$ is just equal to the probability of the events $$(s_0,a_0)$$ through $$(s_T,a_T)$$ happening). So, the summation is just over all the possible sequences $$(s_0,a_0),...,(s_T,a_T)$$.

One possible algorithm to find the best policy is as follows:

Loop  
{  
$$\ \ \ \ \$$ 1\. Sample $$(s_0,a_0),...,(s_T,a_T)$$  
$$\ \ \ \ \$$ 2\. Compute $$\text{payoff} = R(s_0,a_0)+...+R(s_T,a_T)$$  
$$\ \ \ \ \$$ 3\. Update:

<center>$$ \theta := \theta + \alpha\left[\frac{\nabla_\theta \pi_\theta(s_0,a_0)}{\pi_\theta(s_0,a_0)} + ... + \frac{\nabla_\theta \pi_\theta(s_T,a_T)}{\pi_\theta(s_T,a_T)} \right] \times \text{payoff} $$</center>

}

The update step is just stochastic gradient descent on $$\theta$$ because:

<center>$$ \begin{eqnarray} \nabla_\theta\mathbb{E}[\text{payoff}] &=& \sum_{(s_0a_0),...,(s_Ta_T)}P(s_0)\nabla_\theta(\pi_\theta(s_0,a_0))P_{s_0a_0}(s_1)\pi_\theta(s_1,a_1)...P_{s_{T-1}a_{T-1}}(s_T)\pi_\theta(s_T,a_T) [R(s_0,a_0)+...+R(s_T,a_T)]\\ && +P(s_0)\pi_\theta(s_0,a_0)P_{s_0a_0}(s_1)\nabla_\theta(\pi_\theta(s_1,a_1))...P_{s_{T-1}a_{T-1}}(s_T)\pi_\theta(s_T,a_T) [R(s_0,a_0)+...+R(s_T,a_T)]\\ && + \ .\ .\ . \ . \ . \ . \ .\ .\ .\\ && + P(s_0)\pi_\theta(s_0,a_0)P_{s_0a_0}(s_1)\pi_\theta(s_1,a_1)...P_{s_{T-1}a_{T-1}}(s_T)\nabla_\theta(\pi_\theta(s_T,a_T)) [R(s_0,a_0)+...+R(s_T,a_T)]\\ &=& \sum_{(s_0a_0),...,(s_Ta_T)} P(s_0)\pi_\theta(s_0,a_0)...P_{s_{T-1}a_{T-1}}(s_T)\pi_\theta(s_T,a_T)\left[\frac{\nabla_\theta \pi_\theta(s_0,a_0)}{\pi_\theta(s_0,a_0)} + ... + \frac{\nabla_\theta \pi_\theta(s_T,a_T)}{\pi_\theta(s_T,a_T)}\right][R(s_0,a_0)+...+R(s_T,a_T)]\\ &=& \sum_{(s_0a_0),...,(s_Ta_T)} P\left((s_0,a_0),...,(s_T,a_T)\right)\left[\frac{\nabla_\theta \pi_\theta(s_0,a_0)}{\pi_\theta(s_0,a_0)} + ... + \frac{\nabla_\theta \pi_\theta(s_T,a_T)}{\pi_\theta(s_T,a_T)}\right][R(s_0,a_0)+...+R(s_T,a_T)]\\ &=& \mathbb{E}\left[\left(\frac{\nabla_\theta \pi_\theta(s_0,a_0)}{\pi_\theta(s_0,a_0)} + ... + \frac{\nabla_\theta \pi_\theta(s_T,a_T)}{\pi_\theta(s_T,a_T)}\right)[R(s_0,a_0)+...+R(s_T,a_T)]\right]\\ &=& \mathbb{E}\left[\left(\frac{\nabla_\theta \pi_\theta(s_0,a_0)}{\pi_\theta(s_0,a_0)} + ... + \frac{\nabla_\theta \pi_\theta(s_T,a_T)}{\pi_\theta(s_T,a_T)}\right)\times \text{payoff}\right] \end{eqnarray} $$</center>

where the first step follows from the product rule in differentiation and the rest from simple algebraic manipulation.