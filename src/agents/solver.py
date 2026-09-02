from .base_agent import BaseAgent

SOLVER_INIT_PROMPT = """You are solving a reasoning problem.

Question: {question}

Think step by step, then give your final answer.
Respond in this exact format:
REASONING: <your step-by-step reasoning>
ANSWER: <your final answer, short and direct>
"""

SOLVER_REFINE_PROMPT = """You are solving a reasoning problem.

Question: {question}

Your previous answer: {prev_answer}
Your previous reasoning: {prev_reasoning}

A critic reviewed your answer and gave this feedback:
{critic_feedback}: {critic_reasoning}

Reconsider your answer in light of this feedback. You may keep your original
answer if you believe it is correct, or revise it if the feedback identifies
a real error.
Respond in this exact format:
REASONING: <your updated step-by-step reasoning>
ANSWER: <your final answer, short and direct>
"""


class SolverAgent(BaseAgent):
    def solve(self, question):
        prompt = SOLVER_INIT_PROMPT.format(question=question)
        return self.generate(prompt)

    def refine(self, question, prev_answer, prev_reasoning, critic_feedback, critic_reasoning):
        prompt = SOLVER_REFINE_PROMPT.format(
            question=question,
            prev_answer=prev_answer,
            prev_reasoning=prev_reasoning,
            critic_feedback=critic_feedback,
            critic_reasoning=critic_reasoning,
        )
        return self.generate(prompt)
