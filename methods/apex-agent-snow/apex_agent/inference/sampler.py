from typing import List, Dict, Any

from apex_agent.sql_generation.agent import AgenticSqlAgent


class CandidateSampler:
    """Run the agentic loop n_samples times to produce candidate SQLs."""

    def __init__(self, agent: AgenticSqlAgent, n_samples: int = 1):
        self.agent = agent
        self.n_samples = n_samples

    def run(
        self,
        question: str,
        logical_plan: str,
        schema_repr: str,
        guidance: str,
        evidence: str,
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        for seed in range(self.n_samples):
            roll = self.agent.run(
                question=question,
                logical_plan=logical_plan,
                schema_repr=schema_repr,
                guidance=guidance,
                evidence=evidence,
            )
            roll["seed"] = seed
            candidates.append(roll)
        return candidates
