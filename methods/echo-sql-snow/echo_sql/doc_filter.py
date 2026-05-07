"""Evidence Linking — extract relevant snippets from external knowledge.

Pass-through for short docs; otherwise asks the LLM to copy-paste relevant
sections (high-recall extraction, no rewriting).
"""
from typing import Optional

from echo_sql.llm.client import LLMClient
from echo_sql.llm.tokenizer import count_tokens


_PROMPT = """You are an expert Data Analyst Assistant supporting a Text-to-SQL system.
We have a User Query and an External Knowledge Document (Markdown format) that contains business rules,
calculation logic, or data dictionary definitions.
Your task is to **extract** every piece of information from the document that is relevant to the User Query.

### Input
- **User Query**: {query}
- **Knowledge File Name**: {knowledge_file_name}
- **Original Knowledge Content**:
```markdown
{knowledge_content}
```

### Instructions
1. **High recall**: keep anything potentially related (entities, metrics, formulas, codes).
2. **Preserve context**: keep whole paragraphs/list items/table rows; include dependent lines.
3. **Do not rewrite**: copy-paste exactly. No summarizing/paraphrasing.

### Output
Output ONLY the extracted markdown content, with no introductory or concluding text.
"""


class ExternalDocFilter:
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
        prompt = _PROMPT.format(
            query=question,
            knowledge_file_name=knowledge_file_name or "(unnamed)",
            knowledge_content=knowledge_content,
        )
        resp = self.llm.prompt(system=None, user=prompt, temperature=self.temperature)
        return resp["content"].strip()
