---
layout: post
title: Reinforcement Learning - II
permalink: blog/machine-learning/cs229-notes/reinforcement-learning-II
categories: [Machine Learning, CS229 Notes]
---

## 1\. Finite-Horizon MDPs

We shall consider the following general setting:

1.  We shall allow the states and actions to be either continuous or discrete. In that regard, we shall write $$\mathbb{E}_{s \sim P_{sa}}[V^\pi(s')]$$ instead of $$\sum_{s' \in \mathcal{S}}P_{sa}(s')V^\pi(s')$$ or $$\int_{s' \in \mathcal{S}}P_{sa}(s')V^\pi(s')$$.

2.  We shall assume that the rewards depend on both the states and the actions, i.e. $$R:\mathcal{S}\times\mathcal{A}\mapsto \mathbb{R}$$.

3.  We shall assume a finite horizon MDP defined by the tuple $$(\mathcal{S},\mathcal{A},P_{sa},T,R)$$, where $$T>0$$ is the time horizon. We shall, consequently, modify the definition of the total payoff as:

    <center>$$ R(s_0,a_0) + R(s_1,a_1) + ... + R(s_T, a_T) $$</center>

    Note that since the time horizon is finite the total payoff will always be finite, and hence we do not need $$\gamma$$ any more (recall that $$\gamma$$ was used to ensure that the infinite sum summed up to a finite value).

4.  We shall allow the policy $$\pi^*$$ to be non-stationary (i.e. it can change over time). Hence, we have $$\pi^{(t)}:\mathcal{S} \mapsto \mathcal{A}$$, where the superscript $$(t)$$ denotes that the policy is at time step $$t$$. This also allows us to use time-dependent dynamics. Specifically, $$s_{t+1} \sim P_{s_ta_t}^{(t)}$$, meaning that the state transition probabilities vary over time. Similarly, the reward function will be denoted with $$R^{(t)}(s_t,a_t)$$, i.e. it too will vary with time. Consequently, our MDP will be defined by the tuple $$(\mathcal{S},\mathcal{A},P_{sa}^{(t)},T,R^{(t)})$$.

Our goal is to find the optimal value function $$V^*_t(s)=\max_{\pi}V_t(s)$$. In that regard, note that:

1.  At time $$T$$, the optimal value function is given by:

    <center>$$ V^*_T(s) := \max_{a \in \mathcal{A}} R^{(T)}(s,a)\ \ \ \ \forall s \in \mathcal{S} $$</center>

2.  Given the optimal value function at $$t+1$$, for $$% <![CDATA[ t < T %]]>$$, the optimal value function at $$t$$ is:

    <center>$$ V^*_t(s) := \max_{a \in \mathcal{A}}\left[R^{(t)}(s,a) + \mathbb{E}_{s' \sim P_{sa}^{(t)}}[V^*_{t+1}(s')]\right]\ \ \ \ \forall s \in \mathcal{S} $$</center>

Therefore, we can calculate the optimal value functions at all time steps by calculating its values for all $$s$$ at time $$T$$ using the equation in the first step above. Thereafter, using the equation in the second step we can calculate the optimal value functions at times $$T-1,T-2,...,1,0$$ in that order. This is an example of Dynamic Programming.

## 2\. Linear Quadratic Regulation (LQR)

Suppose that $$\mathcal{S} \in \mathbb{R}^n$$ and $$\mathcal{A} \in \mathbb{R}^d$$. Also, suppose that:

<center>$$ s_{t+1} = A_ts_t + B_ta_t + w_t $$</center>

where $$A_t \in \mathbb{R}^{n \times n}$$, $$B_t \in \mathbb{R}^{n \times d}$$ and $$w_t \sim \mathcal(0,\Sigma_t)$$. We shall also assume that the rewards are quadratic, i.e.:

<center>$$ R^{(t)}(s_t,a_t) = -s^T_tU_ts_t - a^T_tV_ta_t $$</center>

where $$U_t \in \mathbb{R}^{n \times n}$$ and $$V_t \in \mathbb{R}^{d \times d}$$ are positive semi-definite matrices. Note that this means that the rewards are always negative.

The LQR model has the following two steps:

1.  Find the matrices $$A$$ and $$B$$ using a simulator model as described in this [post]({{site.baseurl}}{%post_url /blog/machine-learning/cs229-notes/2018-11-15-reinforcement-learning-I%}). Specifically, execute the MDP for some $$m$$ trials and choose values for $$A$$ and $$B$$ that minimize:

    <center>$$ \sum_{i=1}^m\sum_{j=0}^T \left\vert\left\vert s_{t+1}^{(i)} - (As_t^{(i)}+Ba_t^{(i)}) \right\vert\right\vert $$</center>

    Finally, estimate $$w$$ using [Gaussian Discriminant Analysis]({{site.baseurl}}{%post_url /blog/machine-learning/cs229-notes/2018-08-29-generative-learning-algorithms%}).

2.  Using the dynamic programming algorithm in the previous section to find the optimal value policy.

    1.  Initialization Step:

        <center>$$ \begin{eqnarray} V^*_T(s_T) &=& \max_{a_T \in \mathcal{A}}R^{(T)}(s_T,a_T)\\ &=& \max_{a_T \in \mathcal{A}}\left[-s_T^TU_Ts_T - a_T^TV_Ta_T\right]\\ &=& -s_T^TU_Ts_T \end{eqnarray} $$</center>

        where the last step follows from the fact that setting $$a_T = \vec{0}$$ gives the maximum reward. This is because $$V_T$$ is semi-positive definite and hence $$a_T^TV_Ta_T$$ is always greater than or equal to zero.

    2.  Recurrence Step:

        Let $$% <![CDATA[ t<T %]]>$$. Suppose that we know $$V^*_{t+1}$$.

        It can be shown that if $$V_{t+1}^* = s_{t+1}^T\Phi_{t+1}s_{t+1}+\Psi$$ then $$V_{t}^* = s_{t}^T\Phi_{t}s_{t}+\Psi$$ for some symmetric matrix $$\Phi$$ and some scalar $$\Psi$$. Note that since $$V_T^*$$ can be expressed in this form (specifically, $$\Phi_T=U_T$$ and $$\Psi_T=0$$), $$V_t$$ for all $$% <![CDATA[ t<T %]]>$$ too can be expressed in this form.

        Therefore:

        <center>$$ \begin{eqnarray} V^*_t(s_t) &=& s_t\Phi_ts+\Psi_t\\ &=& \max_{a_t \in A}\left[R^{(t)}(s_t,a_t) + \mathbb{E}_{s_{t+1}\sim P_{s_ta_t}}[V^*_{t+1}(s_{t+1})] \right]\\ &=& \max_{a_t \in A}\left[-s_t^TU_ts_t - a_t^TV_ta + \mathbb{E}_{s_{t+1}\sim\mathcal{N}(As_t+Bs_t,\Sigma_t)}[s_{t+1}^T\Phi_{t+1}s_{t+1} + \Psi_{t+1}] \right] \end{eqnarray} $$</center>

        where the second line is just the definition of the optimal value function and the third line follows from our initial assumption about $$s_{t+1}$$. Note that to find $$a_t$$ we just need to set the derivative within the outermost square brackets above to zero, i.e.:

        <center>$$ \frac{\partial}{\partial a_t}\left(-s_t^TU_ts_t - a_t^TV_ta + \mathbb{E}_{s_{t+1}\sim\mathcal{N}(As_t+Bs_t,\Sigma_t)}[s_{t+1}^T\Phi_{t+1}s_{t+1} + \Psi_{t+1}] \right) = 0 $$</center>

        To solve this equation we will be making use of the properties and derivatives of trace operators discussed in this [post]({{site.baseurl}}{%post_url /blog/mathematics/2018-08-15-matrix-derivatives%}).

        Denoting $$\mathbb{E}_{s_{t+1}\sim\mathcal{N}(As_t+Bs_t,\Sigma_t)}[s_{t+1}^T\Phi_{t+1}s_{t+1} + \Psi_{t+1}]$$ by $$C$$ we note that:

        <center>$$ \begin{eqnarray} C &=& \mathbb{E}_{w_{t}\sim\mathcal{N}(0,\Sigma_{t})}\left[(A_ts_t+B_ta_t+w_t)^T \Phi_{t+1}(A_ts_t+B_ta_t+w_t)+\Psi_{t+1}\right]\\ &=&\mathbb{E}_{w_{t}\sim\mathcal{N}(0,\Sigma_{t})}\left[(A_ts_t+B_ta_t)^T\Phi_{t+1}(A_ts_t+B_ta_t) + (A_ts_t+B_ta_t)^T\Phi_{t+1}w_t + w_t^T\Phi_{t+1} (A_ts_t+B_ta_t) + w^T_t\Phi_{t+1}w_t + \Psi_{t+1}\right]\\ &=& (A_ts_t+B_ta_t)^T\Phi_{t+1}(A_ts_t+B_ta_t) + \mathbb{E}_{w_{t}\sim\mathcal{N}(0,\Sigma_{t})}[w^T_t\Phi_{t+1}w_t] + \Psi_{t+1} \end{eqnarray} $$</center>

        where the last step follows from the fact that:

        <center>$$ \begin{eqnarray} \mathbb{E}_{w_{t}\sim\mathcal{N}(0,\Sigma_{t})} \left[(A_ts_t+B_ta_t)^T\Phi_{t+1}w_t\right] &=& (A_ts_t+B_ta_t)^T\Phi_{t+1} \mathbb{E}_{w_{t}\sim\mathcal{N}(0,\Sigma_{t})} \left[w_t\right]\\ &=& 0 \end{eqnarray} $$</center>

        And similarly $$\mathbb{E}_{w_{t}\sim\mathcal{N}(0,\Sigma_{t})}\left[w_t^T\Phi_{t+1} (A_ts_t+B_ta_t)\right] =0$$.

        Note that only the first term in $$C$$ depends on $$a_t$$. Therefore:

        <center>$$ \begin{eqnarray} \frac{\partial C}{\partial a_t} &=& \frac{\partial}{\partial a_t} \left( s_t^TA_t^T\Phi_{t+1}A_ts_t + s_t^TA_t^T\Phi_{t+1}B_ta_t +a_t^TB_t^T\Phi_{t+1}A_ts_t + a_t^TB_t^T\Phi_{t+1}B_ta_t\right)\\ &=& (s_t^TA_t^T\Phi_{t+1}B_t)^T + B_t^T\Phi_{t+1}A_ts_t + B_t^T\Phi_{t+1}B_ta_t + (B_t^T\Phi_{t+1}B_t)^Ta_t\\ &=& B_t^T\Phi_{t+1}^TA_ts_t + B_t^T\Phi_{t+1}A_ts_t + B_t^T\Phi_{t+1}B_ta_t + B_t^T\Phi_{t+1}^TB_ta_t\\ &=& 2B_t^T\Phi_{t+1}A_ts_t+2B_t^T\Phi_{t+1}B_ta_t \end{eqnarray} $$</center>

        where the last step follows from the fact that $$\Phi_{t+1}$$ is symmetric. Therefore:

        <center>$$ \begin{eqnarray} \frac{\partial}{\partial a_t}\left(-s_t^TU_ts_t - a_t^TV_ta + \mathbb{E}_{s_{t+1}\sim\mathcal{N}(A_ts_t+B_ts_t,\Sigma_t)}[s_{t+1}^T\Phi_{t+1}s_{t+1} + \Psi_{t+1}] \right) &=& -(V^T_ta_t + V_ta_t) + 2B_t^T\Phi_{t+1}A_ts_t+2B_t^T\Phi_{t+1}B_ta_t\\ &=& -2V_ta_t + 2B_t^T\Phi_{t+1}A_ts_t+2B_t^T\Phi_{t+1}B_ta_t \end{eqnarray} $$</center>

        where the last step follows from the fact that $$V_{t}$$ is positive semi-definite and hence symmetric.

        Setting this to zero gives:

        <center>$$ \begin{eqnarray} 0 &=& -2V_ta_t + 2B_t^T\Phi_{t+1}A_ts_t + 2B_t^T\Phi_{t+1}B_ta_t\\ (V_t-B_t^T\Phi_{t+1}B_t)a_t &=& B_t^T\Phi_{t+1}A_ts_t\\ a_t &=& (V_t-B_t^T\Phi_{t+1}B_t)^{-1}B_t^T\Phi_{t+1}A_ts_t \end{eqnarray} $$</center>

        We shall refer to this value of $$a_t$$ as $$a_t^*$$. Note that:

        <center>$$ a^* = L_ts_{t} $$</center>

        where $$L_t=(V_t-B^T\Phi_{t+1}^TB)^{-1}B^T\Phi_{t+1}A$$. In other words, the policy is a linear function of the states.

        Let us now plug $$a^*$$ into the equation for optimal value function:

        <center>$$ \begin{eqnarray} V_t^*(s_t) &=& -s_t^TU_ts_t - {a_t^*}^TV_ta^* + (A_ts_t+B_ta_t)^T\Phi_{t+1}(A_ts_t+B_ta_t) + \mathbb{E}_{w_{t}\sim\mathcal{N}(0,\Sigma_{t})}[w^T_t\Phi_{t+1}w_t] + \Psi_{t+1}\\ &=& \left(-s_t^TU_ts_t - {a_t^*}^TV_ta^* + (A_ts_t+B_ta_t)^T\Phi_{t+1}(A_ts_t+B_ta_t)\right) + \left(tr(\Sigma_t\Phi_{t+1}) + \Psi_{t+1}\right)\\ &=&\left(-s_t^TU_ts_t - s^T_tL_t^TV_tL_ts_t + (A_ts_t+B_tL_ts_t)^T\Phi_{t+1}(A_ts_t+B_tL_ts_t)\right) + \left(tr(\Sigma_t\Phi_{t+1}) + \Psi_{t+1}\right)\\ &=&\left(-s_t^TU_ts_t - s^T_tL_t^TV_tL_ts_t + s_t^T(A_t+B_tL_t)^T\Phi_{t+1}(A_t+B_tL_t)s_t \right) + \left(tr(\Sigma_t\Phi_{t+1}) + \Psi_{t+1}\right)\\ &=&s_t^T\left(-U_ts_t - L_t^TV_tL_t + (A_t+B_tL_t)^T\Phi_{t+1}(A_t+B_tL_t)\right)s_t + \left(tr(\Sigma_t\Phi_{t+1}) + \Psi_{t+1}\right)\\ &=&s_t^T\left(-U_t - L_t^TV_tL_t + A_t^T\Phi_{t+1}A_t + A_t^T\Phi_{t+1}B_tL_t + L_t^TB_t^T\Phi_{t+1}A_t + L_t^TB_t^T\Phi_{t+1}B_tL_t\right)s_t + \left(tr(\Sigma_t\Phi_{t+1}) + \Psi_{t+1}\right)\\ &=& s_t^T\left(-U_t + A_t^T\Phi_{t+1}A_t + A_t^T\Phi_{t+1}B_tL_t - L_t^T(V_t- B_t^T\Phi_{t+1}B_t)L_t + L_t^TB_t^T\Phi_{t+1}A_t\right)s_t + \left(tr(\Sigma_t\Phi_{t+1}) + \Psi_{t+1}\right)\\ &=& s_t^T\left(-U_t + A_t^T\Phi_{t+1}A_t + A_t^T\Phi_{t+1}B_tL_t - L_t^TB_t^T\Phi_{t+1}A_t + L_t^TB_t^T\Phi_{t+1}A_t\right)s_t + \left(tr(\Sigma_t\Phi_{t+1}) + \Psi_{t+1}\right)\\ &=& s_t^T\left(-U_t + A_t^T\Phi_{t+1}A_t + A_t^T\Phi_{t+1}B_t(V_t-B^T\Phi_{t+1}^TB)^{-1}B^T\Phi_{t+1}^TA\right)s_t + \left(tr(\Sigma_t\Phi_{t+1}) + \Psi_{t+1}\right)\\ &=& s_t^T\left(-U_t + A_t^T\left(\Phi_{t+1} + \Phi_{t+1}B_t(V_t-B^T\Phi_{t+1}^TB)^{-1}B^T\Phi_{t+1}^T\right)A\right)s_t + \left(tr(\Sigma_t\Phi_{t+1}) + \Psi_{t+1}\right) \end{eqnarray} $$</center>

        where the second step follows from the identity $$\mathbb{E}[w_t\Phi_{t+1}w^T]=tr(\Sigma_t\Phi_{t+1})$$ and $$tr$$ is the trace operator. The proof of this identity is as follows:

        <center>$$ \begin{eqnarray} \mathbb{E}_{w_t \sim \mathcal{N}(0,\Sigma_t)}[w^T\Phi_{t+1}w] &=& \mathbb{E}_{w_t \sim \mathcal{N}(0,\Sigma_t)}[tr(w^T\Phi_{t+1}w)]\\ &=& \mathbb{E}_{w_t \sim \mathcal{N}(0,\Sigma_t)}[tr(ww^T\Phi_{t+1})]\\ &=& tr\left(\mathbb{E}_{w_t \sim \mathcal{N}(0,\Sigma_t)}[ww^T\Phi_{t+1}]\right)\\ &=& tr\left(\mathbb{E}_{w_t \sim \mathcal{N}(0,\Sigma_t)}[ww^T]\Phi_{t+1}\right)\\ &=& tr(\Sigma_t\Phi_{t+1}) \end{eqnarray} $$</center>

        where the second line comes from the properties of traces discussed in this [post]({{site.baseurl}}{%post_url /blog/mathematics/2018-08-15-matrix-derivatives%}) and third line is a consequence of the linearity of the trace and expectation operators. The final line comes from the fact that (we drop the subscript $$w_t \sim \mathcal{N}(0,\Sigma_t)$$ from $$\mathbb{E}$$ for readability):

        <center>$$ \begin{eqnarray} \Sigma &=& \mathbb{E}[(w-\mathbb{E}[w])(w-\mathbb{E}[w])^T]\\ &=& \mathbb{E}[ww^T] \end{eqnarray} $$</center>

        where the final line comes from the fact that $$\mathbb{E}[w]=0$$.

        Note that:

        <center>$$ V^*_t(s_t) = s_t^T\Phi_ts_t + \Psi_t $$</center>

        where:

        <center>$$ \begin{eqnarray} \Phi_t &=& -U_t + A_t^T\left(\Phi_{t+1} + \Phi_{t+1}B_t(V_t-B^T \Phi_{t+1}B)^{-1} B^T\Phi_{t+1} \right)A\\ \Psi_t &=& tr(\Sigma_t\Phi_{t+1}) + \Psi_{t+1} \end{eqnarray} $$</center>

        Also, note that that $$\Phi_t$$ is symmetric because of the following: $$B^T \Phi_{t+1}B$$ is symmetric as:

        <center>$$ \begin{eqnarray} (B^T \Phi_{t+1}B)^T &=& B^T \Phi_{t+1}^TB\\ &=& B^T \Phi_{t+1}B \end{eqnarray} $$</center>

        where the last step follows from the fact that $$\Phi_{t+1}$$ is symmetric. Also, subtracting the symmetric matrix $$B^T \Phi_{t+1}B$$ from another symmetric matrix $$V_t$$ (recall that $$V_t$$ is positive semi-definite and hence symmetric) results in another symmetric matrix. Repeating these arguments for the multiplication with $$\Phi_{t+1}B_t$$, addition to $$\Phi_{t+1}$$ multiplication with $$A_t$$ and the subtraction of $$U_t$$ proves that $$\Phi_t$$ is indeed a symmetric matrix.

        Note that because $$\Phi_T = U_T$$ is a symmetric matrix, all $$\Phi_t$$ for $$% <![CDATA[ t<T %]]>$$ are symmetric too and hence we were justified in assuming that $$\Phi_{t+1}$$ was symmetric in our derivation.

        Finally, note that $$\Phi_t$$ does not depend on the zero-mean Gaussian noise and so does the optimal action $$a^*$$. Hence, to make the algorithm run faster, we can only update $$\Phi_t$$ and not $$\Psi_t$$.

## 3\. From Non-Linear Dynamics to LQR

The LQR formulation explicitly assumed that the system dynamics were linear. If, however, the system dynamics are non-linear then we can reduce it to a linear system and use the LQR algorithm.

### 3.1 Linearization of Dynamics

Consider a system with non-linear dynamics. Suppose that at time $$t$$, it spends most of its time in state $$\bar{s}_t$$ and most of the actions are around $$a_t$$. An example of such a case is when a system reaches a stable state (such as a helicopter hovering above ground) and only deviates a little, only to return to that state. Note that $$t$$ can more appropriately be thought of as a time interval in this case than a specific point in time. We may then approximate $$s_{t+1}$$ using the Taylor expansion:

<center>$$ s_{t+1} \approx F\left(\bar{s}_t,\bar{a}_t) + \nabla_s F(\bar{s}_t,\bar{a}_t)(s_t-\bar{s}_t) + \nabla_s F(\bar{s}_t,\bar{a}_t)(a_t-\bar{a}_t\right) $$</center>

Or:

<center>$$ s_{t+1} \approx As_t + Ba_t + \kappa $$</center>

where $$A$$ and $$B$$ are some matrices and $$\kappa$$ is a constant. Note that we can absorb $$\kappa$$ in $$s_t$$ by increasing its dimension by one, i.e. $$% <![CDATA[ {s_{t}}_{new} = \begin{bmatrix}s_t &1\end{bmatrix}^T %]]>$$. Note that $$A$$ will also have an additional column. Hence:

<center>$$ s_{t+1} \approx A{s_t}_{new} + Ba_t $$</center>

which similar to the assumption in the LQR algorithm.

### 3.2 Differential Dynamic Programming (DDP)

Suppose that we require of a system with non-linear dynamics to travel a certain trajectory. Differential Dynamic Programming divides this trajectory into discrete time steps. It then linearizes the system dynamics in each time interval separately, defines intermediary goals and applies the LQR algorithm on each interval to find the optimal policy. A high level overview of the algorithm is as follows:

1.  Initialize a simple/naïve controller that approximates the trajectory with: $$\bar{s}_0,\bar{a}_0\rightarrow \bar{s}_1,\bar{a}_1 \rightarrow ...$$

2.  Linearize the dynamics around each point in the trajectory using the linearization technique in the previous subsection, i.e.:

    <center>$$ s_{t+1} \approx A_t{s_t} + B_ta_t $$</center>

    Note that the $$A$$ and $$B$$ matrices are time-dependent (i..e different for each time interval). Also, use a second-order Taylor expansion for the reward $$R^{(t)}$$. This would give:

    <center>$$ R(s_t,a_t) = -s_t^TU_ts_t-a_t^TV_ta_t $$</center>

3.  Use the LQR algorithm to find the optimal policy.

4.  Using the new policy, generate a new trajectory and go back to step 2\. Repeat until some stopping criteria is met. Note that to generate the new trajectory, we shall use the original non-linear mapping $$s_t \times a_t \mapsto s_{t+1}$$ and not its linear approximation.

## 4\. Partially Observable MDPs (POMDPs)

Consider some system. Let $$s_t$$ be the state at time $$t$$. Also, let $$o_t$$ be an observation of the state $$s_t$$, such that:

<center>$$ o_t \vert s_t \sim \mathcal{O}(o \vert s) $$</center>

i.e. $$o_t$$ follows some conditional distribution $$\mathcal{O}$$ given $$s_t$$. One way to proceed is to maintain a belief state of $$s_t$$ based on the observation $$o_1,...,o_t$$. A policy then maps these belief states to actions. Note that the POMDP is defined by the tuple $$\{\mathcal{S},\mathcal{O},\mathcal{A},\{P_{sa}\},T,R\}$$.

## 5\. Linear Quadratic Gaussian (LQG)

Let us present an extension to the LQR algorithm for POMDPs. Suppose that we observe $$y_t \in \mathbb{R}^m$$, with $$% <![CDATA[ m < n %]]>$$ where $$n$$ is the dimensionality of the state (i.e. $$s_t \in \mathbb{R}^n$$) such that:

<center>$$ \begin{eqnarray} y_t &=& Cs_t + v_t\\ s_{t+1} &=& As_t + Ba_t + w_t \end{eqnarray} $$</center>

where $$C \in \mathbb{R}^{m \times n}$$ is the compression matrix, $$v_t \sim \mathcal{N}(0,\Sigma_{v})$$ and the rest of the symbols are the same as in LQR. We leave the reward function as a function of the state (not the observation) and the action. The following is a high-level overview of the LQG algorithm:

1.  Find the distribution $$s_t \vert y_1, ...,y_t \sim \mathcal{N}(s_{t \vert t},\Sigma_{t \vert t})$$.
2.  Set the mean $$s_{t \vert t}$$ to be the best approximation for $$s_t$$.
3.  Set $$a_t := L_ts_{t \vert t}$$, where $$L_t$$ comes from the regular LQR algorithm.

To perform the first step efficiently we will make use of the Kalman filter. The Kalman filter has two steps (which we state them without proof):

1.  Predict step: Compute $$s_{t+1} \vert y_1,...,y_t \in \mathcal{N}(s_{t+1\vert t},\Sigma_{t+1\vert t})$$ where:

    <center>$$ \begin{eqnarray} s_{t+1\vert t} &=& As_{t \vert t}\\ \Sigma_{t+1\vert t} &=& A\Sigma_{t \vert t}A^T+\Sigma_{v} \end{eqnarray} $$</center>

2.  Update step: Compute $$s_{t+1} \vert y_1,...,y_{t+1} \in \mathcal{N}(s_{t+1\vert t+1},\Sigma_{t+1\vert t+1})$$

    <center>$$ \begin{eqnarray} s_{t+1\vert t+1} &=& s_{t+1\vert t} + K_t(y_{t+1} - Cs_{t+1\vert t})\\ \Sigma_{t+1\vert t+1} &=& \Sigma_{t+1\vert t} - K_tC\Sigma_{t+1\vert t} \end{eqnarray} $$</center>

    where $$K_t=\Sigma_{t+1\vert t}C^T(C\Sigma_{t+1\vert t}C^T+\Sigma_y)^{-1}$$.

Note that the update steps at $$t+1$$ only depend on the distributions at the previous time step, $$t$$ and not on time steps before that.
