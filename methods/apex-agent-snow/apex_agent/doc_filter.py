from typing import Optional

from apex_agent.llm.client import LLMClient
from apex_agent.llm.tokenizer import count_tokens
from apex_agent.sql_generation.prompts import F_DOC_FILTER


class ExternalDocFilter:
    """Evidence Linking — extract relevant snippets from external knowledge.

    The paper's prompt asks for a verbatim copy-paste of relevant sections.
    """

    def __init__(
        self,
        llm: LLMClient,
        passthrough_token_limit: int = 3000,
        temperature: float = 0.2,
    ):
        self.llm = llm
        self.passthrough_token_limit = passthrough_token_limit
        self.temperature = temperature

    def run(
        self,
        question: str,
        knowledge_content: str,
        knowledge_file_name: Optional[str] = None,
    ) -> str:
        if not knowledge_content.strip():
            return ""
        if count_tokens(knowledge_content, model=self.llm.model) <= self.passthrough_token_limit:
            return knowledge_content

        prompt = F_DOC_FILTER.format(
            query=question,
            knowledge_file_name=knowledge_file_name or "(unnamed)",
            knowledge_content=knowledge_content,
        )
        resp = self.llm.prompt(system=None, user=prompt, temperature=self.temperature)
        return resp["content"].strip()
