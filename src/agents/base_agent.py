import re
import time
import requests
from dataclasses import dataclass


@dataclass
class AgentResponse:
    answer: str
    reasoning: str
    raw_text: str


class BaseAgent:
    def __init__(self, model_name, role, temperature, base_url, max_tokens=512, timeout=180, max_retries=2):
        self.model_name = model_name
        self.role = role
        self.temperature = temperature
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries

    def _call_model(self, prompt):
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(url, json=payload, timeout=self.timeout)
                response.raise_for_status()
                return response.json()["response"]
            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(5)
                    continue
        raise last_error

    def _parse_response(self, raw_text):
        answer_match = re.search(r"ANSWER:\s*(.+?)(?:\n|$)", raw_text, re.IGNORECASE)
        reasoning_match = re.search(r"REASONING:\s*(.+?)(?=\nANSWER:|$)", raw_text, re.IGNORECASE | re.DOTALL)

        answer = answer_match.group(1).strip() if answer_match else raw_text.strip().split("\n")[0]
        reasoning = reasoning_match.group(1).strip() if reasoning_match else raw_text.strip()

        return AgentResponse(answer=answer, reasoning=reasoning, raw_text=raw_text)

    def generate(self, prompt):
        raw_text = self._call_model(prompt)
        return self._parse_response(raw_text)