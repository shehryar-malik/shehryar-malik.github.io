---
layout: post
title: Proof of Theorems on Ranks
permalink: blog/mathematics/proofs-of-theorems-on-ranks
categories: [Mathematics]
---

## Rank of the Sum of Matrices

Let $$A$$ and $$B$$ be two matrices of rank $$m$$ and $$n$$ respectively. Let $$\{C_{a_k}\}_{k=1}^m$$ and $$\{C_{b_k}\}_{k=1}^n$$ denote the sets of basis for the column space of matrix $$A$$ and $$B$$ respectively. Each column of $$A$$ is, thus, a linear combination of the elements in the set $$\{C_{a_k}\}_{k=1}^m$$and each column of $$B$$ is a linear combination of the elements in the set $$\{C_{b_k}\}_{k=1}^n$$. From this it follows that each column in $$A+B$$ is a linear combination of the elements in the combined set $$\{C_{a_k}\}_{k=1}^m \cup \{C_{b_k}\}_{k=1}^n$$. Let $$rank(C_A \cap C_B )$$ denote the rank of the intersection of the column spaces of $$A$$ and $$B$$. Note that this is, informally, equal to the number of elements common to the sets $$\{C_{b_k}\}_{k=1}^n$$ and $$\{C_{b_k}\}_{k=1}^n$$. Then:

<center>$$ rank(A+B) \leqslant rank(A) + rank(B) - rank(C_A \cap C_B) $$</center>

Similarly, letting $$R_A$$ and $$R_B$$ denote the row space of $$A$$ and $$B$$ respectively leads to:

<center>$$ rank(A+B) \leqslant rank(A) + rank(B) - rank(R_A \cap R_B) $$</center>

Now, because the rank due to the row space must be equal to the rank due to the column space, we have:

<center>$$ rank(A+B) \leqslant rank(A) + rank(B) - max(rank(C_A \cap C_B),rank(R_A \cap R_B)) $$</center>
