from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class RoundLog:
    """Một vòng debate. Ở kiến trúc peers, mọi agent cùng trả lời trong mỗi vòng.

    answers: {agent_name: {"answer": str, "reasoning": str}}
    changed: {agent_name: bool}  # đáp án có khác vòng trước không
    """
    round_idx: int
    answers: Dict[str, Dict[str, str]]
    changed: Dict[str, bool]

    def to_dict(self):
        return {
            "round_idx": self.round_idx,
            "answers": self.answers,
            "changed": self.changed,
        }


@dataclass
class DebateLog:
    question_id: str
    question: str
    ground_truth: str
    rounds: List[RoundLog] = field(default_factory=list)
    independent_answers: Dict[str, str] = field(default_factory=dict)
    final_answers: Dict[str, str] = field(default_factory=dict)
    correct_answers: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "question_id": self.question_id,
            "question": self.question,
            "ground_truth": self.ground_truth,
            "rounds": [r.to_dict() for r in self.rounds],
            "independent_answers": self.independent_answers,
            "final_answers": self.final_answers,
            "correct_answers": self.correct_answers,
        }
