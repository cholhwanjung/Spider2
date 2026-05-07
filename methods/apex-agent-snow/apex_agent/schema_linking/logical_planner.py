from apex_agent.llm.client import LLMClient
from apex_agent.schema_linking.prompts import F_SL_PLAN, F_SL_AGG


class LogicalPlanner:
    """① Logical Planning — N candidate plans + 1 master plan."""

    def __init__(
        self,
        llm: LLMClient,
        n_paths: int = 2,
        sample_temperature: float = 0.8,
        agg_temperature: float = 0.2,
    ):
        self.llm = llm
        self.n_paths = n_paths
        self.sample_temperature = sample_temperature
        self.agg_temperature = agg_temperature

    def run(self, question: str) -> str:
        """The paper's F_SL_PLAN takes only `{question}`. External knowledge is
        handled separately via Evidence Linking (`doc_filter`)."""
        plans = []
        for _ in range(self.n_paths):
            prompt = F_SL_PLAN.format(question=question)
            resp = self.llm.prompt(system=None, user=prompt, temperature=self.sample_temperature)
            plans.append(resp["content"].strip())

        if self.n_paths == 1:
            return plans[0]

        plans_text = "\n\n--- PLAN ---\n".join(plans)
        agg_prompt = F_SL_AGG.format(
            question=question,
            n_plans=self.n_paths,
            plans_text=plans_text,
        )
        resp = self.llm.prompt(system=None, user=agg_prompt, temperature=self.agg_temperature)
        return resp["content"].strip()
