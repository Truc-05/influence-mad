# Influence-Aware Multi-Agent Debate

## 1. Introduction

Large language models (LLMs) have become remarkably strong reasoners, and much of
that progress comes not from new architectures but from *how we prompt and
orchestrate* them at inference time. Chain-of-thought prompting showed that simply
eliciting intermediate reasoning steps unlocks arithmetic, commonsense, and
symbolic reasoning abilities that remain hidden under direct prompting
(Wei et al., 2022). Self-consistency pushed this further by sampling many
reasoning paths from a *single* model and taking a majority vote over their final
answers, turning stochastic decoding into a reliable accuracy gain
(Wang et al., 2023). Yet a single model, no matter how many times it is sampled,
shares one set of parametric biases and blind spots: it tends to be internally
consistent even when it is consistently wrong, and it struggles to escape an
initial line of reasoning once committed to it (Madaan et al., 2023;
Shinn et al., 2023).

Multi-agent debate (MAD) was proposed to break this monoculture. Instead of one
model reasoning alone, several LLM instances independently propose answers, read
one another's responses, and revise over several rounds before a final answer is
extracted (Du et al., 2023). Framed as a computational "society of minds," debate
has been shown to improve factuality and reduce hallucination across reasoning
benchmarks (Du et al., 2023), to counteract the *degeneration-of-thought* problem
in which a confident model can no longer produce novel ideas (Liang et al., 2024),
and to benefit from diverse models and confidence-weighted consensus
(Chen et al., 2024). A rapidly growing literature now studies how to make debate
cheaper and better through sparse communication topologies
(Li et al., 2024) and targeted interventions on message quality
(Estornell and Liu, 2024).

Beneath these gains, however, debate remains a **social black box**. When accuracy
improves, we rarely know *why*: which agent moved the group, whether that movement
was toward truth or away from it, and whether the final vote actually reflects the
most reliable participant. This matters because debate is not a neutral averaging
procedure — it is a persuasion process. LLMs exhibit strong sycophancy, prioritizing
agreement over correctness (Sharma et al., 2024), and populations of agents display
human-like conformity, converging on a shared answer through asymmetric social
influence rather than independent verification (Zhang et al., 2024). The
consequence is that a persuasive but wrong agent can drag correct peers to its
answer, an *error cascade* that debate has no built-in defense against. Indeed,
careful re-evaluations find that MAD does not reliably beat well-tuned
self-consistency and is unusually sensitive to configuration (Smit et al., 2024).
If some influence is *good* (correct agents pulling others toward truth) and some
is *bad* (confident errors propagating), then treating every agent's final vote
equally — as majority voting does — leaves accuracy on the table.

We argue that the influence structure of a debate is a *first-class, measurable
object* rather than an incidental side effect, and that measuring it yields a
directly actionable signal for aggregation. Our key observation is that a debate
transcript induces a **directed influence network**: whenever agent *j* abandons
its own answer and adopts agent *i*'s position, that is a recorded act of *i*
influencing *j*. Aggregating these events across questions and rounds gives an
**influence matrix**, and applying a PageRank-style centrality
(Page et al., 1998) over that matrix produces a single **influence score** per
agent — a principled measure of how much each agent moves the group, that accounts
for indirect and transitive persuasion the same way PageRank accounts for a page
being linked by other important pages. Crucially, influence is not the same as
being right: by correlating each agent's influence score with its answer accuracy,
we can separate **good influence** (high influence, high accuracy) from **bad
influence** (high influence, low accuracy). This diagnostic then feeds a final
**influence-aware voting** rule that up-weights trustworthy influencers and
discounts persuasive-but-unreliable ones.

Concretely, we design a four-stage pipeline — data collection, influence
estimation, influence analysis, and influence-aware voting — and instantiate it
with three heterogeneous open-weight LLMs debating as peers with distinct personas.
This paper makes the following contributions:

- **An influence-estimation framework for MAD.** We formalize a debate transcript
  as a directed influence graph and construct an influence matrix directly from
  answer-change events, requiring only black-box access to the debating models and
  no additional annotation.
- **A PageRank-based influence score.** We adapt centrality analysis to rank agents
  by how much they move the group, capturing direct *and* transitive influence in a
  single interpretable score.
- **A good/bad influence diagnostic.** By relating influence to per-agent accuracy,
  we classify agents into good, bad, and low-influence roles, exposing when a debate
  is helped or harmed by its most persuasive members.
- **Influence-aware voting.** We introduce a final aggregation rule that reweights
  votes by (accuracy-calibrated) influence, and compare it against majority voting
  and raw influence-weighted voting.
- **A multi-seed empirical study.** We evaluate across arithmetic, commonsense, and
  factuality benchmarks with repeated seeds, reporting mean and standard deviation
  so that claimed improvements are separated from run-to-run noise — a discipline
  recent critiques show is essential for MAD (Smit et al., 2024).

Taken together, our results reframe multi-agent debate from an opaque accuracy
booster into an *analyzable social system*, and show that measuring who influences
whom — and whether that influence is trustworthy — is a lightweight lever for making
debate both more interpretable and more accurate.

## 2. Related Work

### 2.1 Prompting and Self-Correction for LLM Reasoning

A first line of work improves reasoning within a single model. Chain-of-thought
prompting elicits intermediate steps that dramatically raise performance on
multi-step problems (Wei et al., 2022), and self-consistency aggregates many
sampled reasoning paths by majority vote, trading extra computation for robustness
(Wang et al., 2023). A complementary family lets a model critique and revise its
own output: Self-Refine iterates generation and self-feedback (Madaan et al., 2023),
while Reflexion converts feedback into verbal memory that guides later attempts
(Shinn et al., 2023). These methods are powerful but fundamentally *intra-model*:
they recombine the reasoning of one network and therefore inherit its systematic
errors, offering no external source of disagreement. Our work keeps their
inference-time, black-box philosophy but moves to a *multi-model* setting where
disagreement — and thus influence — becomes possible and measurable.

### 2.2 Multi-Agent Debate and LLM Collaboration

Multi-agent debate instantiates the "society of minds" idea by having several LLMs
exchange and revise answers. Du et al. (2023) show that debate improves factuality
and mathematical reasoning across tasks with nothing more than black-box model
access. Liang et al. (2024) frame debate as a remedy for *degeneration of thought*,
using a "tit-for-tat" exchange moderated by a judge to keep agents generating
genuinely new arguments. ReConcile (Chen et al., 2024) assembles *diverse* models
in a round-table discussion and, importantly for us, replaces plain majority voting
with a **confidence-weighted** vote, showing that not all votes should count
equally. Beyond debate proper, general multi-agent frameworks such as AutoGen
(Wu et al., 2023) and CAMEL (Li et al., 2023) provide role-playing and conversational
scaffolding for cooperating LLMs. More recent work optimizes the debate itself:
Li et al. (2024) show that *sparse* communication topologies match or exceed
fully-connected debate at far lower token cost, and Estornell and Liu (2024) provide
a formal framework with quality-pruning and misconception-refutation interventions
that provably steer debate toward correct answers. These works largely treat the
communication structure as a *design choice imposed by the system builder*. We
instead treat influence as an *emergent property to be measured from the
transcript*, and use that measurement to inform aggregation rather than to hand-design
the topology.

### 2.3 Social Dynamics: Conformity, Sycophancy, and Influence

A growing body of work examines the *social* behavior of interacting LLMs, which
directly motivates our study. Zhang et al. (2024) analyze collaboration through a
social-psychology lens and document machine analogues of human conformity and
consensus-reaching, including cases where agents abandon correct answers under
social pressure. Sycophancy — the tendency of LLMs to agree with an interlocutor at
the expense of truth — is well documented at the user–model level (Sharma et al.,
2024) and is especially corrosive in debate, where inter-agent agreeability can
collapse discussion into premature, incorrect consensus. This is precisely the
*bad influence* our framework is designed to detect and down-weight. The tension is
also visible in critical evaluations: Smit et al. (2024) find that debate does not
reliably outperform tuned self-consistency and is highly configuration-sensitive,
implying that the *quality* of influence, not merely its presence, determines
whether debate helps. Compared with this literature, which primarily *diagnoses*
conformity and sycophancy or proposes prompt-level mitigations, we contribute a
*quantitative, graph-based* estimator of per-agent influence and connect it to a
concrete aggregation mechanism.

### 2.4 Answer Aggregation and Confidence Weighting

How debate outputs are combined is itself a design axis. The dominant choice is
majority voting, inherited from self-consistency (Wang et al., 2023), which assumes
every participant is equally reliable. ReConcile's confidence-weighted vote
(Chen et al., 2024) relaxes this by trusting self-reported confidence, and
LLM-as-a-judge approaches (Zheng et al., 2023) delegate the final decision to a
strong evaluator model — though such judges carry their own position, verbosity, and
self-enhancement biases. Our influence-aware voting differs in the *source* of the
weights: rather than self-reported confidence (which is itself vulnerable to
sycophancy and overconfidence) or a separate judge, we derive weights from *observed
influence calibrated by accuracy*, so that an agent earns voting power by being both
persuasive and correct, not merely by claiming certainty.

### 2.5 Influence and Centrality in Networks

Finally, our estimator builds on classical network analysis. PageRank
(Page et al., 1998) ranks nodes in a directed graph by treating a link as an
endorsement and rewarding nodes endorsed by other well-endorsed nodes, capturing
transitive importance beyond raw in-degree. We repurpose this idea for debate: an
agent that persuades others — especially agents who are themselves influential —
should receive a higher influence score. To our knowledge, casting a multi-agent
debate transcript as a directed influence network and applying centrality analysis
to obtain per-agent influence scores, then using those scores to reweight the final
vote, is a novel combination that connects the LLM-debate and network-analysis
literatures and turns the "society of minds" metaphor into a measurable object.

## References

Chen, J. C.-Y., Saha, S., and Bansal, M. (2024). *ReConcile: Round-Table
Conference Improves Reasoning via Consensus among Diverse LLMs.* In Proceedings of
the 62nd Annual Meeting of the Association for Computational Linguistics (ACL).
arXiv:2309.13007.

Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., and Mordatch, I. (2023).
*Improving Factuality and Reasoning in Language Models through Multiagent Debate.*
arXiv:2305.14325. (ICML 2024.)

Estornell, A. and Liu, Y. (2024). *Multi-LLM Debate: Framework, Principals, and
Interventions.* In Advances in Neural Information Processing Systems (NeurIPS).

Li, G., Hammoud, H. A. A. K., Itani, H., Khizbullin, D., and Ghanem, B. (2023).
*CAMEL: Communicative Agents for "Mind" Exploration of Large Language Model
Society.* In Advances in Neural Information Processing Systems (NeurIPS).
arXiv:2303.17760.

Li, Y., Du, Y., Zhang, J., Hou, L., Grabowski, P., Li, Y., and Ie, E. (2024).
*Improving Multi-Agent Debate with Sparse Communication Topology.* In Findings of
the Association for Computational Linguistics: EMNLP 2024. arXiv:2406.11776.

Liang, T., He, Z., Jiao, W., Wang, X., Wang, Y., Wang, R., Yang, Y., Shi, S., and
Tu, Z. (2024). *Encouraging Divergent Thinking in Large Language Models through
Multi-Agent Debate.* In Proceedings of the 2024 Conference on Empirical Methods in
Natural Language Processing (EMNLP). arXiv:2305.19118.

Madaan, A., Tandon, N., Gupta, P., Hallinan, S., Gao, L., Wiegreffe, S., Alon, U.,
Dziri, N., Prabhumoye, S., Yang, Y., Gupta, S., Majumder, B. P., Hermann, K.,
Welleck, S., Yazdanbakhsh, A., and Clark, P. (2023). *Self-Refine: Iterative
Refinement with Self-Feedback.* In Advances in Neural Information Processing
Systems (NeurIPS). arXiv:2303.17651.

Page, L., Brin, S., Motwani, R., and Winograd, T. (1998). *The PageRank Citation
Ranking: Bringing Order to the Web.* Technical Report SIDL-WP-1999-0120, Stanford
InfoLab.

Sharma, M., Tong, M., Korbak, T., Duvenaud, D., Askell, A., Bowman, S. R., Cheng, N.,
Durmus, E., Hatfield-Dodds, Z., Johnston, S. R., Kravec, S., Maxwell, T., McCandlish,
S., Ndousse, K., Rausch, O., Schiefer, N., Yan, D., Zhang, M., and Perez, E. (2024).
*Towards Understanding Sycophancy in Language Models.* In International Conference on
Learning Representations (ICLR). arXiv:2310.13548.

Shinn, N., Cassano, F., Berman, E., Gopinath, A., Narasimhan, K., and Yao, S.
(2023). *Reflexion: Language Agents with Verbal Reinforcement Learning.* In Advances
in Neural Information Processing Systems (NeurIPS). arXiv:2303.11366.

Smit, A., Duckworth, P., Grinsztajn, N., Barrett, T. D., and Pretorius, A. (2024).
*Should We Be Going MAD? A Look at Multi-Agent Debate Strategies for LLMs.* In
Proceedings of the 41st International Conference on Machine Learning (ICML).
arXiv:2311.17371.

Wang, X., Wei, J., Schuurmans, D., Le, Q. V., Chi, E. H., Narang, S., Chowdhery, A.,
and Zhou, D. (2023). *Self-Consistency Improves Chain of Thought Reasoning in
Language Models.* In International Conference on Learning Representations (ICLR).
arXiv:2203.11171.

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E., Le, Q.,
and Zhou, D. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large
Language Models.* In Advances in Neural Information Processing Systems (NeurIPS).
arXiv:2201.11903.

Wu, Q., Bansal, G., Zhang, J., Wu, Y., Li, B., Zhu, E., Jiang, L., Zhang, X., Zhang,
S., Liu, J., Awadallah, A. H., White, R. W., Burger, D., and Wang, C. (2023).
*AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation.*
arXiv:2308.08155.

Zhang, J., Xu, X., Zhang, N., Liu, R., Hooi, B., and Deng, S. (2024). *Exploring
Collaboration Mechanisms for LLM Agents: A Social Psychology View.* In Proceedings
of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL).
arXiv:2310.02124.

Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z.,
Li, D., Xing, E. P., Zhang, H., Gonzalez, J. E., and Stoica, I. (2023). *Judging
LLM-as-a-Judge with MT-Bench and Chatbot Arena.* In Advances in Neural Information
Processing Systems (NeurIPS), Datasets and Benchmarks Track. arXiv:2306.05685.
