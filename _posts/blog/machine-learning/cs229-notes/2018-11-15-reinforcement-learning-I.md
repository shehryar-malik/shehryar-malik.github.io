---
layout: post
title: Reinforcement Learning - I
permalink: blog/machine-learning/cs229-notes/reinforcement-learning-I
categories: [Machine Learning, CS229 Notes]
---

We define a Markov Decision Process (MDP) for a system to be a tuple $$\{\mathcal{S},\mathcal{A},\{P_{sa}\},\gamma,R\}$$ where:

1.  $$\mathcal{S}$$ is the set of all possible states that a system can be in
2.  $$\mathcal{A}$$ is the set of all possible actions that the system can take
3.  $$\{P_{sa}\}$$ is the set of state transition probabilities, i.e. the probability that the system will transition into some state if an action $$a \in \mathcal{A}$$ is taken in state $$s \in \mathcal{S}$$
4.  $$\gamma \in [0,1)$$ is the discount factor
5.  $$R: \mathcal{S} \times \mathcal{A} \mapsto \mathbb{R}$$ is the reward function, i.e. the “reward” that the system gets for choosing action $$\mathcal{A}$$ while being in state $$\mathcal{S}$$.

For now, we shall assume that the reward only depends on the state $$\mathcal{S}$$, i.e. $$R : \mathcal{S} \mapsto \mathbb{R}$$.

Suppose that the system transitions from $$s_0$$ to $$s_1$$ by taking action $$a_0$$ and from $$s_1$$ to $$s_2$$ by taking action $$a_2$$ and so on. The total payoff is then given by:

<center>$$ R(s_0) + \gamma R(s_1) + \gamma^2 R(s_2) + ... $$</center>

where the reward for the $$t^{th}$$ state is multiplied by $$\gamma^t$$. Note that $$\gamma$$ ensures that the cumulative reward sums up to a finite number. This can be shown by assuming that the rewards $$R(s)$$ are bounded by some finite number $$\bar{R}$$. Then:

<center>$$ \begin{eqnarray} \sum_{t=0}^\infty \gamma^t R(s) &\leqslant& \bar{R}\sum_{t=0}^\infty \gamma^t\\ &<& \infty \end{eqnarray} $$</center>

Note that the last step follows from the fact that $$\sum_{t=0}^\infty \gamma^t$$ is a geometric series that sums to a finite number because $$% <![CDATA[ \vert \gamma \vert < 1 %]]>$$.

The goal in reinforcement learning is to maximize the expected value of the total payoff.

<center>$$ \mathbb{E}\left[R(s_0) + \gamma R(s_1) + \gamma^2 R(s_2) + ...\right] $$</center>

We define the policy $$\pi: \mathcal{S} \mapsto \mathcal{A}$$ as a mapping from state $$\mathcal{S}$$ to action $$\mathcal{A}$$. We, thus, execute a policy $$\pi$$ if, whenever we are in state $$s$$, we choose action $$a = \pi (s)$$. We also define the value function for a policy $$\pi$$ as:

<center>$$ V^{\pi}(s) = \mathbb{E}\left[R(s_0) + \gamma R(s_1) + \gamma^2 R(s_2) + ... \vert s_0=s,\pi\right] $$</center>

Note that we condition on $$\pi$$ even though it is not a random variable. This is not technically correct but is customary in literature. Note that:

<center>$$ \begin{eqnarray} V^{\pi}(s) &=& \mathbb{E}\left[R(s_0) + \gamma R(s_1) + \gamma^2 R(s_2) + ... \vert s_0=s,\pi\right]\\ &=& \mathbb{E}\left[R(s_0)\vert s_0=s,\pi\right] + \mathbb{E}\left[\gamma R(s_1) + \gamma^2 R(s_2) + ... \vert s_0=s,\pi\right]\\ &=& R(s) + \gamma\mathbb{E}\left[R(s_1) + \gamma R(s_2) + ... \vert s_0=s,\pi\right]\\ &=& R(s) + \gamma \sum_{s' \in \mathcal{S}}P_{s\pi(s)}(s')\mathbb{E}\left[R(s_1) + \gamma R(s_2) + ... \vert s_1=s',\pi\right]\\ &=& R(s) + \gamma \sum_{s' \in \mathcal{S}}P_{s\pi(s)}(s')V^\pi(s') \end{eqnarray} $$</center>

Note that the second-to-last step follows from the fact that $$P_{s\pi(s)}(s')$$ gives the probability that we will transition into state $$s'$$ if we execute policy $$\pi$$ in state $$s$$. Note that we have dropped the condition $$s_0 = s$$ in this step for the second term because no term in the expectation depends on it.

The final equation is known as the Bellman’s equation. Note that one such equation can be written out for each state $$s$$. Thus, we have will have $$\vert \mathcal{S} \vert$$ equations in $$\vert \mathcal{S} \vert$$ variables which can then we solved for the $$V^{\pi}(s)$$’s.

We also define the optimal value function as follows:

<center>$$ V^*(s) = \max_{\pi}V^{\pi}(s) $$</center>

Equivalently:

<center>$$ V^*(s) = R(s) + \max_{a \in \mathcal{A}}\gamma \sum_{s' \in \mathcal{S}}P_{sa}(s')V^\pi(s') $$</center>

where we have replaced $$\pi(s)$$ with $$a$$ for notational convenience.

Note that, we may also define the optimal policy to be:

<center>$$ \pi^*(s) = \underset{a \in \mathcal{A}}{argmax}\sum_{s' \in \mathcal{S}}P_{sa}(s')V^\pi(s') $$</center>

Clearly:

<center>$$ V^*(s) = V^{\pi^*}(s) \geqslant V^{\pi}(s) $$</center>

## Solving Finite-State MDPs

For the case when $$% <![CDATA[ \vert \mathcal{S} \vert < \infty %]]>$$ and $$% <![CDATA[ \vert \mathcal{A} \vert < \infty %]]>$$, we may use either of the following two algorithms to solve the MDP:

### Value Iteration

1.  For each state $$s$$ initialize $$V(s) := 0$$
2.  Repeat until convergence:  
    {  
    $$\ \ \ \ \ \ \$$For every state $$s$$, update $$V(s) := R(s) + \max_{a \in \mathcal{A}}\sum_{s' \in \mathcal{S}}P_{sa}(s')V(s')$$  
    }

Note that the inner loop may be carried out by either (1) finding the new $$V(s)$$’s for all the states first and then updating them altogether, or by (2) updating $$V(s)$$’s to their new value immediately after the new value is found (i.e. updating states one at a time). These two update mechanisms are referred to as synchronous and asynchronous updates respectively.

It can be shown that $$V(s)$$ ultimately converges to $$V^*(s)$$. Once it does, we can then calculate the optimal policy using its equation defined above.

### Policy Iteration

1.  Initialize a policy $$\pi$$ randomly.

2.  Repeat until convergence:  
    {  
    $$\ \ \ \ \ \ \$$a) Let $$V := V^{\pi}$$

    $$\ \ \ \ \ \ \$$b) For each state $$s$$, let $$\pi(s) := \underset{a \in \mathcal{A}}{argmax}\sum_{s' \in \mathcal{S}}P_{sa}(s')V(s')$$  
    }

Note that in the first step of the inner loop, we set $$V$$ to be equal to our current value function. This can be done by solving the Bellman’s equations for $$V(s)$$’s as described earlier. Then in the second step, we compute our next policy using our current value function (This policy $$\pi$$ is also called as the policy that is greedy with respect to $$V$$). It can be shown that ultimately $$V$$ will converge to $$V^*$$ and $$\pi$$ to $$\pi^*$$.

## Learning a Model for an MDP

Suppose that for a particular MDP the state transition probabilities are unknown. We may then run different trials of that MDP to estimate these probabilities:

<center>$$ P_{sa}(s') = \frac{\text{No. of times we took action}\ a\ \text{in state}\ s\ \text{and got to}\ s'}{\text{No. of times we took action}\ a\ \text{in state}\ s} $$</center>

We may also do this to find the rewards if they are unknown.

Thus, for an MDP with unknown state transition probabilities we may use the following algorithm:

1.  Initialize $$\pi$$ randomly.

2.  Repeat until convergence:  
    {  
    $$\ \ \ \ \ \ \$$a) Execute $$\pi$$ in the MDP for some number of trials.

    $$\ \ \ \ \ \ \$$b) Using the results of these trials update the state transition probabilities and rewards.

    $$\ \ \ \ \ \ \$$c) Apply value iteration with the estimated state transition probabilities to find an estimate of $$V$$.

    $$\ \ \ \ \ \ \$$d) Update $$\pi$$ to be the greedy policy with respect to $$V$$.  
    }

Note that we may optimize the inner loop by initializing $$V$$ in the value iteration step to our previous estimate of $$V$$. This will help the algorithm to converge faster.

## Continuous State MDPs

Consider an MDP that has a continuous state i.e. $$\mathcal{S} \in \mathbb{R}^n$$ is an infinite set of states. For now we will consider $$% <![CDATA[ \vert \mathcal{A} \vert < \infty %]]>$$. We discuss two approaches to solve for this MDP.

### Discretization

In this approach, we discretize the state space $$\mathcal{S}$$ into $$k$$ discrete states. We refer to the set of these discrete states as $$\bar{\mathcal{S}}$$. We then solve for $$V^*(\bar{s})$$ and $$\pi^*(\bar{s})$$ in the discrete state MDP $$\{\bar{\mathcal{S}},\mathcal{A},\{P_{\bar{s}a}\},$$ $$\gamma,R\}$$. However, there are two major problems with this approach:

1.  Discretization assumes that $$V^*$$ is constant within a discretization interval. This may not be the case.
2.  Suppose $$\mathcal{S} \in \mathbb{R}^n$$. If we discretize each dimension into $$k$$ values, then we have a total number of $$k^n$$ discrete states. Note that this grows exponentially in the $$n$$ and hence will not scale well to higher dimensions. This is also known as the curse of dimensionality.

### Building a Simulator of an MDP

We digress to talk about how to build a simulator for an MDP. Suppose that we execute a MDP for $$m$$ trials, where each trial has $$T$$ timesteps. We may then learn a model of the following form:

<center>$$ s_{t+1} = A\phi_s(s_t) + B\phi_a(a_t) $$</center>

i.e. given the state at time $$t$$ and the action chosen at that time, the model outputs the next state. Here $$\phi_s$$ and $$\phi_a$$ are some mappings of the state and action at time $$t$$ respectively. Note that to $$A$$ and $$B$$ may be found by solving for the following optimization problem:

<center>$$ \underset{A,B}{argmin} \sum_{i=1}^m\sum_{t=0}^{T-1}\left\vert\left\vert s_{t+1}^{(i)}-\left(A\phi_s(s_t^{(i)}) + B\phi_a(a_t^{(i)}) \right)\right\vert\right\vert^2 $$</center>

Note that the superscript $$(i)$$ on $$s$$ indicates the trial number of the MDP. Also, note that this is a deterministic model. We may also build a stochastic model of the following form:

<center>$$ s_{t+1} = A\phi_s(s_t) + B\phi_a(a_t) + \epsilon_t $$</center>

where $$\epsilon_t$$ is a noise term and is usually modelled as $$\epsilon \sim \mathcal{N}(0,\Sigma)$$, where the covariance matrix $$\Sigma$$ may also be estimated from the data.

Alternatively, one may also use a physics simulator to simulate the states.

### Fitted Value Iteration

In value iteration, we repeatedly do the following update:

<center>$$ \begin{eqnarray} V(s) &:=& R(s) + \gamma \max_{a}\int_{s'\in \mathcal{S}} P_{sa}(s')V(s')ds'\\ &=& R(s) + \gamma \max_{a}\mathbb{E}_{s' \sim P_{sa}}[V(s')] \end{eqnarray} $$</center>

Note that we use the integral here (instead of the summation) because we have a continuous state space.

In fitted value iteration, we carry out this step for a finite sample of states $$s^{(1)},...,s^{(m)}$$ and try to approximate the value function as a function of the states, i.e:

<center>$$ V(s) = \theta^T \phi(s) $$</center>

where $$\phi(s)$$ is some appropriate function/feature mapping of the states.

The algorithm is as follows:

1.  Randomly sample $$m$$ states $$s^{(1)},...,s^{(m)} \in \mathcal{S}$$
2.  Initialize $$\theta := 0$$
3.  Repeat  
    {  
    $$\ \ \ \ \ \ \$$For $$i=1,...,m$$  
    $$\ \ \ \ \ \ \$${  
    $$\ \ \ \ \ \ \$$$$\ \ \ \ \ \ \$$For each action $$a \in \mathcal{A}$$  
    $$\ \ \ \ \ \ \$$$$\ \ \ \ \ \ \$${  
    $$\ \ \ \ \ \ \$$$$\ \ \ \ \ \ \$$$$\ \ \ \ \ \ \$$Sample $$s_1',...,s_k' \sim P_{s^{(i)}a}$$  
    $$\ \ \ \ \ \ \$$$$\ \ \ \ \ \ \$$$$\ \ \ \ \ \ \$$Set $$q(a):=\frac{1}{k}\sum_{j=1}^k R(s^{(i)})+\gamma V(s_j')$$  
    $$\ \ \ \ \ \ \$$$$\ \ \ \ \ \ \$$}  
    $$\ \ \ \ \ \ \$$$$\ \ \ \ \ \ \$$Set $$y^{(i)} = \max_a q(a)$$  
    $$\ \ \ \ \ \ \$$}  
    $$\ \ \ \ \ \ \$$Set $$\theta := \underset{\theta}{argmin}\frac{1}{2}\sum_{i=1}^m\left(\theta^T\phi(s^{(i)})-y^{(i)}\right)^2$$  
    }

In the innermost loop above, we first sample $$k$$ next states given state $$s^{(i)}$$ and action $$a$$. Note that if we are using a deterministic simulator then we may set $$k=1$$. This is because the simulator will return the same state no matter how many times we input $$s$$ and $$a$$. If, however, we are using a stochastic simulator then we may get a different output each time. Hence, in that case we can sample $$k$$ (distinct) states. After we are done sampling, we calculate $$q(a)$$. To do so, we use our previous estimate of $$\theta$$ to compute $$V(s_j')$$. Note that, $$q(a)$$ is an estimate of $$R(s) + \gamma \mathbb{E}_{s' \sim P_{sa}}[V(s')]$$. We then set $$y^{(i)}$$ to be equal to the maximum value of $$q(a)$$. Therefore, $$y^{(i)}$$ is an estimate of $$R(s) + \gamma \max_{a}\mathbb{E}_{s' \sim P_{sa}}[V(s')]$$. Finally, we try to choose $$\theta$$ such that $$V(s)=\theta^T\phi(s)$$ is as close to $$y^{(i)}$$ as possible. Note that this step is analogous to the step in value iteration where we set $$V(s) = R(s) + \max_{a \in \mathcal{A}}\sum_{s' \in \mathcal{S}}P_{sa}(s')V(s')$$.

While fitted value iteration has not been proven to always converge, it does so (approximately) in most cases.

Note that $$V$$ in the algorithm above is an approximation to $$V^*$$. Hence, when we are in some state $$s$$, we would chose the action:

<center>$$ \underset{a}{argmax}\mathbb{E}_{s' \sim P_{sa}}[V(s')] $$</center>

One way to compute the expectation is to do something similar to the innermost loop of the algorithm above, i.e. sample $$k$$ states (where we set $$k=1$$ if we have a discriminant model) and take their average. However, there are other ways too. Suppose that the simulator is of the form of $$s_{t+1}=f(s_t,a_t)+\epsilon$$, where $$f$$ is some deterministic function and $$\epsilon$$ is some zero-mean Gaussian noise. Then:

<center>$$ \begin{eqnarray} \mathbb{E}_{s'\sim P_{sa}}[V(s')] &\approx& V\left(\mathbb{E}_{s'\sim P_{sa}}[s']\right)\\ &=& V\left(\mathbb{E}_{s'\sim P_{sa}}[f(s,a)+\epsilon]\right)\\ &=& V\left(\mathbb{E}_{s'\sim P_{sa}}[f(s,a)]\right) \end{eqnarray} $$</center>

where the last step follows from the fact that the mean of $$\epsilon$$ is zero and the first step is a reasonable assumption in most cases. Equivalently, note that if we ignore $$\epsilon$$ (or if we have a deterministic simulator), i.e. $$s_{t+1}=f(s_t,a_t)$$ where $$f$$ is deterministic, then:

<center>$$ \begin{eqnarray} \mathbb{E}_{s'\sim P_{sa}}[V(s')] &=& \mathbb{E}_{s'\sim P_{sa}}[V(f(s,a))]\\ &=& V(f(s,a)) \end{eqnarray} $$</center>

where the second step follows from the fact that $$V(f(s,a))$$ will have the same value always (and so its expectation is just that value) because $$f$$ is deterministic (i.e $$f$$ will always return the same value for same $$s$$ and $$a$$).

Note, however, that for cases where $$\epsilon​$$ cannot be ignored, we would need to sample $$k\vert \mathcal{A} \vert​$$ states ($$k​$$ states for each action $$a​$$). This may turn out to be computationally very expensive.