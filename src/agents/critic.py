from .base_agent import BaseAgent

CRITIC_PROMPT = """You are reviewing another agent's answer to a reasoning problem.

Question: {question}

Proposed answer: {solver_answer}
Proposed reasoning: {solver_reasoning}

Carefully check whether the reasoning is logically sound and the answer is
correct. Only disagree if you find a clear factual or logical error.

Respond in this exact format:
REASONING: <your critique>
ANSWER: <one word: agree, disagree, or uncertain>
"""

CRITIC_INDEPENDENT_PROMPT = """You are solving a reasoning problem independently,
without seeing anyone else's answer.

Question: {question}

Think step by step, then give your final answer.
Respond in this exact format:
REASONING: <your step-by-step reasoning>
ANSWER: <your final answer, short and direct>
"""


class CriticAgent(BaseAgent):
    def critique(self, question, solver_answer, solver_reasoning):
        prompt = CRITIC_PROMPT.format(
            question=question,
            solver_answer=solver_answer,
            solver_reasoning=solver_reasoning,
        )
        response = self.generate(prompt)
        feedback = response.answer.strip().lower()
        if "disagree" in feedback:
            feedback = "disagree"
        elif "uncertain" in feedback:
            feedback = "uncertain"
        else:
            feedback = "agree"
        response.answer = feedback
        return response

    def answer_independently(self, question):
        prompt = CRITIC_INDEPENDENT_PROMPT.format(question=question)
        return self.generate(prompt)
