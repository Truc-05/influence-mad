from .base_agent import BaseAgent


PERSONAS = {
    "careful": (
        "You are a careful, meticulous problem-solver. You work strictly step by "
        "step and double-check every arithmetic and logical step before concluding."
    ),
    "skeptic": (
        "You are a skeptical checker. You actively hunt for errors and hidden "
        "assumptions, and you are willing to disagree with the majority when you "
        "find a real flaw."
    ),
    "bigpicture": (
        "You are a big-picture reasoner. You focus on the overall approach and "
        "sanity-check whether the final answer is plausible at a high level."
    ),
}

INIT_PROMPT = """{persona}

Question: {question}

Think step by step, then give your final answer.
Respond in this exact format:
REASONING: <your step-by-step reasoning>
ANSWER: <your final answer, short and direct>
"""

REFINE_PROMPT = """{persona}

Question: {question}

Your previous answer: {own_answer}
Your previous reasoning: {own_reasoning}

Other agents proposed these answers in the previous round:
{peer_block}

Reconsider your answer. Keep it if you still believe it is correct, or revise it
if another agent's reasoning exposes a real error in yours. Do not change just to
agree — change only if convinced.
Respond in this exact format:
REASONING: <your updated step-by-step reasoning>
ANSWER: <your final answer, short and direct>
"""


class PeerAgent(BaseAgent):
    def __init__(self, *args, persona="careful", **kwargs):
        super().__init__(*args, **kwargs)
        self.persona_key = persona
        self.persona_text = PERSONAS.get(persona, PERSONAS["careful"])

    def solve(self, question):
        prompt = INIT_PROMPT.format(persona=self.persona_text, question=question)
        return self.generate(prompt)

    def refine(self, question, own_answer, own_reasoning, peers):
        # peers: {agent_name: {"answer": str, "reasoning": str}}
        peer_lines = []
        for name, resp in peers.items():
            peer_lines.append(f"- Agent '{name}' answered: {resp['answer']}  (because: {resp['reasoning']})")
        peer_block = "\n".join(peer_lines) if peer_lines else "(none)"

        prompt = REFINE_PROMPT.format(
            persona=self.persona_text,
            question=question,
            own_answer=own_answer,
            own_reasoning=own_reasoning,
            peer_block=peer_block,
        )
        return self.generate(prompt)
