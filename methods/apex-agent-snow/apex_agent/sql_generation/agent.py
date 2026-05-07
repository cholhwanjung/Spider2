import logging
import re
from typing import List, Dict, Any, Optional

from apex_agent.llm.client import LLMClient
from apex_agent.exec.snowflake_runner import SnowflakeRunner
from apex_agent.exec.profiler import sigma_compress
from apex_agent.sql_generation.prompts import (
    F_SQL_ACTION_SYSTEM,
    F_SQL_ACTION_USER,
)


logger = logging.getLogger("apex_agent.action")


_TAG_RE = re.compile(r"\[(EXPLORE|REFINE|SQL|CONFIRM)\]", re.IGNORECASE)
_SQL_BLOCK_RE = re.compile(r"```sql\s*(.+?)```", re.DOTALL | re.IGNORECASE)
_GENERIC_BLOCK_RE = re.compile(r"```(?:[a-zA-Z0-9_]*)?\s*(.+?)```", re.DOTALL)


def _parse_action_tag(text: str) -> str:
    m = _TAG_RE.search(text)
    return m.group(1).upper() if m else ""


def _extract_sql_statements(text: str) -> List[str]:
    blocks = _SQL_BLOCK_RE.findall(text)
    if not blocks:
        # Fallback: any fenced block (the agent sometimes omits the language tag)
        blocks = [b for b in _GENERIC_BLOCK_RE.findall(text) if "select" in b.lower() or "with " in b.lower()]
    statements: List[str] = []
    for block in blocks:
        for stmt in block.split(";"):
            cleaned = "\n".join(
                line for line in stmt.splitlines() if not line.strip().startswith("--")
            ).strip()
            if cleaned:
                statements.append(cleaned)
    return statements


class AgenticSqlAgent:
    """⑤ Agentic Exploration Loop with [EXPLORE]/[REFINE]/[SQL]/[CONFIRM] tags."""

    def __init__(
        self,
        llm: LLMClient,
        runner: SnowflakeRunner,
        max_actions: int = 40,
        max_tokens: int = 56000,
        force_synth_action_threshold: int = 38,
        force_synth_token_threshold: int = 52000,
        temperature: float = 0.6,
    ):
        self.llm = llm
        self.runner = runner
        self.max_actions = max_actions
        self.max_tokens = max_tokens
        self.force_synth_action_threshold = force_synth_action_threshold
        self.force_synth_token_threshold = force_synth_token_threshold
        self.temperature = temperature

    def run(
        self,
        question: str,
        logical_plan: str,
        schema_repr: str,
        guidance: str,
        evidence: str,
    ) -> Dict[str, Any]:
        history: List[str] = []
        latest_obs: str = "(none yet)"
        trajectory: List[Dict[str, Any]] = []
        actions_used = 0
        tokens_used = 0
        final_sql: Optional[str] = None
        final_result: Optional[Dict[str, Any]] = None
        last_synth_sql: Optional[str] = None

        while actions_used < self.max_actions:
            force_synth = (
                actions_used >= self.force_synth_action_threshold
                or tokens_used >= self.force_synth_token_threshold
            )
            warning = (
                "*** Budget nearly exhausted — emit [SQL] or [CONFIRM] now. ***"
                if force_synth
                else ""
            )
            user_prompt = F_SQL_ACTION_USER.format(
                question=question,
                logical_plan=logical_plan,
                schema=schema_repr,
                evidence=evidence or "(none)",
                guidance=guidance or "(no tips activated)",
                actions_used=actions_used,
                max_actions=self.max_actions,
                tokens_used=tokens_used,
                max_tokens=self.max_tokens,
                force_synth_warning=warning,
                compressed_history="\n".join(history[-20:]) if history else "(empty)",
                latest_observations=latest_obs,
            )
            resp = self.llm.chat(
                messages=[
                    {"role": "system", "content": F_SQL_ACTION_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
            )
            tokens_used += resp.get("prompt_tokens", 0) + resp.get("completion_tokens", 0)
            content = resp["content"]
            tag = _parse_action_tag(content)

            logger.info(
                f"action #{actions_used+1}/{self.max_actions} tag={tag or 'NONE'} "
                f"tokens_used={tokens_used}/{self.max_tokens}"
            )

            step = {
                "step": actions_used + 1,
                "tag": tag,
                "raw_response": content,
            }

            if tag == "EXPLORE":
                sqls = _extract_sql_statements(content)
                exec_log = []
                obs_pieces = []
                for sql in sqls:
                    result = self.runner.execute(sql, with_limit=True, limit=1000)
                    obs = sigma_compress(result)
                    exec_log.append({"sql": sql, "ok": result.get("ok", False), "observation": obs})
                    obs_pieces.append(f"SQL: {sql}\nObs: {obs[:1200]}")
                step["executions"] = exec_log
                latest_obs = "\n\n".join(obs_pieces) if obs_pieces else "[no SQL extracted]"
                history.append(f"[EXPLORE #{actions_used+1}]\n{latest_obs[:1500]}")

            elif tag == "REFINE":
                history.append(f"[REFINE #{actions_used+1}]\n{content[:2000]}")
                latest_obs = "(refinement recorded — issue [EXPLORE] or [SQL] next)"
                step["observation"] = latest_obs

            elif tag == "SQL":
                sqls = _extract_sql_statements(content)
                sql = sqls[0] if sqls else ""
                if not sql:
                    history.append(f"[SQL #{actions_used+1}] (no SQL block found — retry)")
                    latest_obs = "[no SQL block in [SQL] action — emit ```sql\\n...\\n```]"
                    step["observation"] = latest_obs
                elif last_synth_sql is not None and sql == last_synth_sql:
                    history.append(f"[SQL #{actions_used+1}] DUPLICATE — change strategy.")
                    latest_obs = "[duplicate SQL — change strategy]"
                    step["observation"] = latest_obs
                else:
                    last_synth_sql = sql
                    result = self.runner.execute(sql, with_limit=False)
                    obs = sigma_compress(result)
                    step["sql"] = sql
                    step["ok"] = result.get("ok", False)
                    step["observation"] = obs
                    history.append(f"[SQL #{actions_used+1}]\n{sql}\n→ {obs[:1500]}")
                    latest_obs = obs
                    if result.get("ok"):
                        final_sql = sql
                        final_result = result

            elif tag == "CONFIRM":
                step["confirmation"] = content
                trajectory.append(step)
                actions_used += 1
                if final_sql is None and last_synth_sql is not None:
                    final_sql = last_synth_sql
                    final_result = self.runner.execute(final_sql, with_limit=False)
                    step["observation"] = sigma_compress(final_result)
                break

            else:
                step["observation"] = f"[unrecognized action tag: {content[:200]!r}]"
                latest_obs = step["observation"]

            trajectory.append(step)
            actions_used += 1

            if tokens_used >= self.max_tokens:
                break

        if final_sql is None and last_synth_sql is not None:
            final_sql = last_synth_sql
            if final_result is None:
                final_result = self.runner.execute(final_sql, with_limit=False)

        return {
            "final_sql": final_sql or "",
            "final_result": final_result,
            "trajectory": trajectory,
            "actions_used": actions_used,
            "tokens_used": tokens_used,
        }
