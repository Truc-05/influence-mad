# 3. Method

Our method starts from a simple reframing: a multi-agent debate is not just a
procedure for producing an answer, it is a small society whose members continually
try to persuade one another. Every time an agent reads its peers and rewrites its
answer, it leaves a trace of who moved whom. Our goal is to read those traces,
turn them into a quantitative picture of *who holds sway over the group*, judge
whether that sway is trustworthy, and finally let the trustworthy voices count for
more when the group decides. Concretely, the method unfolds as the four stages of
Figure 1 — **data collection**, **influence estimation**, **influence analysis**,
and **influence-aware voting** — built on a shared debate formulation that we
describe first.

### 3.1 Problem Setup: Debate as an Emerging Social Network

Let $\mathcal{A} = \{a_1, \dots, a_N\}$ be a set of $N$ heterogeneous LLM agents
(in our experiments $N=3$, instantiated as *careful*, *skeptic*, and *bigpicture*
personas backed by three different open-weight models). Given a question
$q$ drawn from a benchmark $\mathcal{D}$ with gold answer $y^\star(q)$, the agents
debate for up to $R$ synchronous rounds. We write $x_i^{(r)}(q)$ for the answer
proposed by agent $a_i$ in round $r$, and let $\bar{x}=\mathrm{norm}(x)$ denote a
normalized answer (lower-cased, whitespace- and punctuation-trimmed) so that
surface variants of the same answer are treated as equal.

The dynamics are deliberately **peer-symmetric**, which is what makes influence
measurable in every direction:

- **Round 0 (independent).** Every agent answers on its own,
  $x_i^{(0)}(q) = a_i(q)$, with no knowledge of the others. These independent
  answers seed the debate and later serve as an unbiased snapshot of each agent's
  standalone ability.
- **Rounds $r \ge 1$ (revision).** All agents update *simultaneously*, each
  conditioning on the previous round's answers of its peers,
  $$
  x_i^{(r)}(q) \;=\; a_i\!\Big(q,\; x_i^{(r-1)}(q),\; \{x_k^{(r-1)}(q)\}_{k \ne i}\Big).
  $$
  Synchronous (rather than sequential) updates avoid ordering artifacts and let us
  attribute a change at round $r$ to what the agent saw at round $r-1$.
- **Termination.** The debate stops early once the group reaches consensus,
  $|\{\bar{x}_i^{(r)}(q)\}_{i=1}^N| = 1$, and otherwise runs to $R$ rounds. The
  final answer of each agent is $x_i^\star(q) = x_i^{(r_{\text{end}})}(q)$.

A single debate therefore yields a transcript recording, for every round, the
answer of every agent. This is the raw material from which the entire influence
analysis is derived — no gold labels, no human annotation, and only black-box
access to the models are required.

### 3.2 Stage 1 — Data Collection

The first stage runs standard multi-agent debate over the benchmark and logs the
full interaction history. For each question we store, per round, the answer and
reasoning of every agent together with a change indicator
$$
c_i^{(r)}(q) \;=\; \mathbb{1}\!\left[\, \bar{x}_i^{(r)}(q) \neq \bar{x}_i^{(r-1)}(q) \,\right],
\qquad r \ge 1,
$$
which marks whether $a_i$ revised its answer relative to the previous round. We
also retain the independent answers $\{x_i^{(0)}(q)\}$ and the final answers
$\{x_i^\star(q)\}$. Logging is incremental and idempotent (each question is written
and flushed as it completes), so collection can be resumed after interruption and
repeated under multiple random seeds for variance estimation.

### 3.3 Stage 2 — Influence Score Estimation

The heart of the method converts these transcripts into a single influence score
per agent, in two steps: first we count *acts of persuasion* into an influence
matrix, then we let those acts propagate through a PageRank centrality.

#### 3.3.1 The Influence Matrix

We want $M_{ij}$ to capture *how strongly agent $a_i$ influences agent $a_j$* —
that is, how readily $a_j$ abandons its own answer in favor of $a_i$'s. Two
conditions must hold for a round to count as evidence about $i \to j$. First, $a_i$
must actually be *offering an alternative*: if $a_i$ and $a_j$ already agree going
into the round, nothing about influence can be learned. Second, $a_j$ must *move
onto $a_i$'s position*. We formalize these as an opportunity set and an adoption
event.

The set of **opportunities** for $a_i$ to influence $a_j$ is
$$
\mathcal{O}_{i \to j} \;=\; \Big\{ (q, r) \;:\; r \ge 1,\;\; \bar{x}_i^{(r-1)}(q) \neq \bar{x}_j^{(r-1)}(q) \Big\},
$$
i.e. all (question, round) pairs where, at the *start* of the round, $a_i$ held a
different answer than $a_j$. Within an opportunity, an **adoption** occurs when
$a_j$'s new answer matches $a_i$'s previous answer and differs from $a_j$'s own
previous answer:
$$
\mathrm{adopt}_{i \to j}(q, r) \;=\; \mathbb{1}\!\left[\, \bar{x}_j^{(r)}(q) = \bar{x}_i^{(r-1)}(q) \;\wedge\; \bar{x}_j^{(r)}(q) \neq \bar{x}_j^{(r-1)}(q) \,\right].
$$
The influence of $a_i$ on $a_j$ is then the empirical adoption rate over all
opportunities,
$$
\boxed{\;M_{ij} \;=\; \frac{\displaystyle\sum_{(q,r)\,\in\,\mathcal{O}_{i \to j}} \mathrm{adopt}_{i \to j}(q, r)}{\bigl|\mathcal{O}_{i \to j}\bigr|}\;}
\qquad (i \neq j), \qquad M_{ii} = 0 .
$$
$M_{ij} \in [0,1]$ is a conditional probability: *given that $a_i$ presented a
competing answer, how often did $a_j$ come around to it?* The diagonal is zero by
construction (self-persuasion is not influence), and the matrix is generally
**asymmetric** ($M_{ij} \neq M_{ji}$), which is exactly the point — a stubborn but
persuasive agent will have a heavy row and a light column. Rows or columns for
which no opportunity ever arises are left at zero rather than filled with a uniform
prior, so the matrix never fabricates influence that the data did not exhibit.

#### 3.3.2 PageRank Influence Score

Raw adoption rates measure *direct* persuasion, but influence is transitive: moving
an agent who in turn moves everyone else should count for more than moving a
peripheral agent. We therefore treat the influence matrix as a weighted directed
graph and rank agents by centrality, mirroring how PageRank ranks a web page highly
when it is endorsed by other highly ranked pages (Page et al., 1998).

The crucial modeling choice is edge direction. Since $M_{ij}$ means "$a_i$
influences $a_j$", and PageRank confers importance on nodes that are *pointed to*,
we orient each edge **from the influenced agent to the influencer**: for every
$M_{ij} > 0$ we add an edge $a_j \to a_i$ with weight $M_{ij}$. An agent thus
accumulates score by being the destination of many (and heavy) persuasion edges.
Let the out-strength of a node be $d_j = \sum_{k} M_{kj}$ (the total influence
exerted *on* $a_j$), giving the transition probability
$$
P_{j \to i} \;=\; \frac{M_{ij}}{\sum_{k} M_{kj}} \quad (d_j > 0).
$$
The influence score $s_i$ is the stationary distribution of the damped random walk,
$$
\boxed{\; s_i \;=\; \frac{1-\alpha}{N} \;+\; \alpha \!\!\sum_{j \,:\, d_j > 0} P_{j \to i}\, s_j \;+\; \frac{\alpha}{N}\!\!\sum_{j \,:\, d_j = 0} s_j \;}
$$
with damping factor $\alpha = 0.85$ and $\sum_i s_i = 1$. The first term is the
uniform teleportation that guarantees a unique solution; the second propagates
score along persuasion edges; the third redistributes the score of *dangling*
agents (those never influenced by anyone, $d_j = 0$) uniformly, so no probability
mass is lost. The resulting vector $\mathbf{s} = (s_1, \dots, s_N)$ is our
per-agent **influence score**: a high $s_i$ marks an agent who repeatedly pulls the
group — and especially pulls other influential agents — toward its own answers.

### 3.4 Stage 3 — Influence Analysis: Good vs. Bad Influence

Being influential is not the same as being right, and the distinction is the moral
of the whole paper: a debate can be *helped* by a correct agent who wins others
over, or *harmed* by a confident agent who spreads its error. To separate the two
we pair each agent's influence with its accuracy. Using the final answers, the
accuracy of $a_i$ is
$$
\mathrm{acc}_i \;=\; \frac{1}{|\mathcal{D}|} \sum_{q \in \mathcal{D}} \mathbb{1}\!\left[\, x_i^\star(q) \;\text{is correct w.r.t.}\; y^\star(q) \,\right].
$$
At the population level we summarize the relationship between influence and
correctness with the Pearson and Spearman correlations between the vectors
$\mathbf{s} = (s_i)$ and $\mathbf{acc} = (\mathrm{acc}_i)$ across agents; a positive
correlation indicates a healthy debate in which the loudest voices are also the
most reliable. At the per-agent level we assign a qualitative role using the median
influence $\tilde{s} = \mathrm{median}(\mathbf{s})$ and median accuracy
$\tilde{a} = \mathrm{median}(\mathbf{acc})$ as thresholds:
$$
\ell_i \;=\;
\begin{cases}
\textsf{good\_influence}, & s_i \ge \tilde{s} \;\wedge\; \mathrm{acc}_i \ge \tilde{a},\\[2pt]
\textsf{bad\_influence},  & s_i \ge \tilde{s} \;\wedge\; \mathrm{acc}_i < \tilde{a},\\[2pt]
\textsf{low\_influence},  & s_i < \tilde{s}.
\end{cases}
$$
An agent is **good** when it is both persuasive and accurate, **bad** when it is
persuasive yet unreliable (the error-cascade culprit), and **low** when it simply
does not move the group. These labels are the bridge from *diagnosis* to *action*:
they tell the final vote whose influence to trust.

### 3.5 Stage 4 — Influence-Aware Voting

The last stage decides the group's answer from the per-agent final answers
$\{x_i^\star(q)\}$. We compare three aggregation rules of increasing awareness.

**Majority voting** is the standard baseline inherited from self-consistency
(Wang et al., 2023); every agent counts equally and ties are broken by first
occurrence:
$$
\hat{y}_{\text{maj}}(q) \;=\; \arg\max_{c} \sum_{i=1}^{N} \mathbb{1}\!\left[\, \bar{x}_i^\star(q) = c \,\right].
$$

**Influence-based voting** replaces the head-count with a sum of influence scores,
so persuasive agents carry heavier ballots:
$$
\hat{y}_{\text{inf}}(q) \;=\; \arg\max_{c} \sum_{i=1}^{N} s_i \, \mathbb{1}\!\left[\, \bar{x}_i^\star(q) = c \,\right].
$$
This trusts influence blindly, and will therefore amplify a *bad*-influence agent
just as readily as a good one.

**Influence-aware voting** closes the loop by recalibrating each agent's weight
with its analyzed role before voting. Using the labels $\ell_i$ from Stage 3, we
define a role gain
$$
\gamma(\ell_i) \;=\;
\begin{cases}
1.2, & \ell_i = \textsf{good\_influence},\\
0.3, & \ell_i = \textsf{bad\_influence},\\
1.0, & \ell_i = \textsf{low\_influence},
\end{cases}
\qquad
w_i \;=\; \gamma(\ell_i)\, s_i,
$$
and cast the influence-weighted vote with the adjusted weights,
$$
\boxed{\; \hat{y}_{\text{aware}}(q) \;=\; \arg\max_{c} \sum_{i=1}^{N} w_i \, \mathbb{1}\!\left[\, \bar{x}_i^\star(q) = c \,\right]. \;}
$$
The gain $\gamma$ boosts agents whose influence is backed by accuracy and sharply
discounts persuasive-but-wrong agents, so that the same social pull which can
trigger an error cascade is instead redirected toward the reliable answer. In this
way the four stages form a single arc: the debate reveals who influences whom, the
matrix and PageRank quantify it, the analysis judges whether that influence is
earned, and the vote finally lets earned influence — and only earned influence —
decide.
