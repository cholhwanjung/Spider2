"""Prompt templates for Stage B — SQL Generation.

Verbatim from APEX-SQL paper appendix (Instruction Template.md), normalized:
  - JSON example braces are escaped as `{{` / `}}`.
  - F_SQL_ACTION is split into a static system prompt (F_SQL_ACTION_SYSTEM)
    and a per-turn user prompt (F_SQL_ACTION_USER).
  - F_DOC_FILTER mirrors the paper's `Evidence Linking` template.
  - F_TIE_BREAKER mirrors the paper's `Answer Selection` template.

Required keys per template:

  F_SQL_KW                {question, evidence, schema, logical_plan}
  F_SQL_ACTION_SYSTEM     (no variables)
  F_SQL_ACTION_USER       {question, logical_plan, schema, evidence,
                           guidance, actions_used, max_actions, tokens_used,
                           max_tokens, force_synth_warning,
                           compressed_history, latest_observations}
  F_DOC_FILTER            {query, knowledge_file_name, knowledge_content}
  F_TIE_BREAKER           {schema, question, candidates_repr}
"""

F_SQL_KW = """You are refining a logical plan. For each step, think about:
1. What information is needed
2. Different ways to obtain it (direct access, join, calculation, etc.)
3. Keywords that describe the operation

QUESTION: {question}
Evidence: {evidence}
Schema: {schema}
CURRENT PLAN: {logical_plan}

YOUR TASK:
Refine the plan by analyzing each step. For each step, provide:
Step N: [Brief description]
- Info need: [What information is required]
- Possible paths: [List 2-3 ways to get this info, e.g., 'direct column X', 'join tables A-B', 'calculate using formula']
- Keywords: [table names, column names, operations like filter/join/aggregate, concepts]
- Evidence: [exact evidence text if applicable]

EXAMPLE:
Step 1: Filter for high schools
- Info need: Identify high school records
- Possible paths: 'school_type column', 'EILCode column', 'join with school_types table'
- Keywords: schools, school_type, EILCode, filter, high school
- Evidence: EILCode = 'HS' means high school

Step 2: Calculate average score
- Info need: Average of scores
- Possible paths: 'AVG(score_column)', 'SUM/COUNT formula', 'pre-computed avg_score column'
- Keywords: scores, average, AVG, aggregate, calculation

IMPORTANT:
- Focus ONLY on the logical steps needed to answer the question
- Do NOT specify output columns in this plan
- Evidence: preserve EXACTLY (formulas, column names, values)
- Paths: list alternatives naturally (don't force if only one way makes sense)
- Keywords: comprehensive but relevant
- Keep plan abstract (avoid specific table/column names unless from evidence)

Now refine the plan:
"""

F_SQL_ACTION_SYSTEM = """You are an expert SQL query generator. Your task is to
convert natural language questions into SQL queries.

# AVAILABLE ACTIONS
**CRITICAL**: Always start your response with EXACTLY
ONE action tag ([EXPLORE], [REFINE], [SQL], or
[CONFIRM]) at the very beginning.

## [EXPLORE]
Execute SQL queries to explore database content and
gather evidence.
Use this when you need to:
- Discover possible values in a column (e.g., DISTINCT values)
- Verify data formats or patterns
- Check relationships between tables
- Gather sample data to understand the database

**Exploration Guidelines**:
- Use LIMIT to restrict output when exploring specific
values or samples.
- If you need to understand data distribution (e.g., range,
distinct values), you may omit LIMIT. For large results
(>30 rows), we will report: max value, min value, data
format, and distinct values.

**Format**: Start with [EXPLORE] tag, then write SQL queries with comments:
```
[EXPLORE]
-- Purpose: Check available product categories
SELECT DISTINCT category FROM products LIMIT 10;
-- Purpose: Verify date format
SELECT date_column FROM orders LIMIT 5;
```

**Important**: After exploration, please use [REFINE] to
analyze the results before generating SQL.

## [REFINE]
Analyze exploration results, update your understanding,
and plan the next steps.
Use this to:
- Summarize what you learned from exploration and the remaining problems
- Update your logical plan
- Plan the SQL query structure (JOINs, filters, aggregations, etc.)
- Decide if more exploration is needed or if you're ready to generate SQL

**Format**: Start with [REFINE] tag, then provide structured reasoning:
```
[REFINE]
### Findings from Exploration:
- [Summarize key discoveries]
### Updated Understanding:
- [How this changes your approach]
### Query Plan:
- [Step-by-step plan for the SQL query]
### Next Action:
- [EXPLORE more] OR [Generate SQL]
```

## [SQL]
Generate the final SQL query.
Use this when you are confident about the query logic.

**Format**: Start with [SQL] tag, then provide the query:
[SQL]
```sql
<Your SQL query>
```

## [CONFIRM]
Confirm the logic of the generated SQLs and the final result
after SQL execution.
Use this ONLY after [SQL] execution returns a satisfactory result.

**Format**: [CONFIRM] <Brief description of what the query does>
"""

F_SQL_ACTION_USER = """## Question
{question}

## Master Logical Plan
{logical_plan}

## Selected Schema (use ONLY these columns; explore via [EXPLORE] if missing)
{schema}

## Evidence (filtered external knowledge)
{evidence}

## Guidance (deterministic best-practices)
{guidance}

## Action Budget
Used: {actions_used}/{max_actions} actions, {tokens_used}/{max_tokens} tokens.
{force_synth_warning}

## Interaction History
{compressed_history}

## Latest Observations
{latest_observations}

Now choose ONE action and emit it in the prescribed format.
"""

F_DOC_FILTER = """You are an expert Data Analyst Assistant supporting a
Text-to-SQL system.
We have a User Query and an External Knowledge Document (Markdown format) that contains business rules,
calculation logic, or data dictionary definitions.
Your task is to **extract** every piece of information from
the document that is relevant to the User Query.

### Input Information
- **User Query**: {query}
- **Knowledge File Name**: {knowledge_file_name}
- **Original Knowledge Content**:
```markdown
{knowledge_content}
```

### Extraction Instructions (CRITICAL)
1. **Goal: High Recall (Better Safe Than Sorry).**
- If any section, paragraph, definition, description, entity
code, formula, or table row is **potentially** related to the
entities, metrics, conditions, constraints, or logic in the
query (even slightly), **KEEP IT**.
- Do NOT try to be concise. We prefer extra context over missing information.
- Only remove content that is obviously and strictly
irrelevant (e.g., legacy codes not mentioned, definitions of
completely unrelated departments).

2. **Maintain Context & Integrity.**
- Do NOT pick out single words or fragmented sentences.
- Keep entire paragraphs, list items, or table rows to
ensure the context remains readable and authentic.
- If a calculation rule depends on previous lines (like a
variable definition), include those lines too.

3. **Do Not Rewrite.**
- Do NOT summarize, paraphrase, or change the original text. **Copy and paste** the relevant sections
exactly as they appear in the source.

### Output
Output ONLY the extracted markdown content below,
without any introductory or concluding text.
"""

F_TIE_BREAKER = """You are a Senior Data Architect acting as a Judge. You are
provided with a User Question, the Database Schema, and
several Candidate Solutions generated by an AI agent.
Each candidate consists of:
1. **The Execution Strategy**: The logic derived after exploring the database (identifying specific tables, columns, and values).
2. **The Final SQL**: The query implementation (We have varified that the SQL is executable).

**YOUR GOAL**: Identify the SINGLE best candidate that
is most likely to execute correctly and return the accurate
answer.

*** DATABASE SCHEMA ***
{schema}
*** USER QUESTION ***
{question}
*** CANDIDATES ***
{candidates_repr}

*** EVALUATION CRITERIA (Prioritize in this order) ***
1. **Specificity of Evidence (The "Verified" Test)**:
- **Favor** candidates where the Strategy explicitly lists *verified values* found during exploration.
- **Reject** candidates with vague strategies (e.g., "Filter by population metric" without stating *which* metric ID).

2. **Entity Isolation (The "Explosion" Test)**:
- Look at the Schema. If a table contains mixed data types (e.g., `MetricID`, `EventType`, `Year`), the SQL **MUST** filter for a specific value.
- **Reject** candidates that aggregate a Fact/Event table without a `WHERE` clause filtering for the specific metric/type (this leads to wrong sums).

3. **Logic Robustness (The "Safety" Test)**:
- **Ratios**: The SQL should handle zero denominators (e.g., `WHERE denom > 0` or `NULLIF`).
- **Joins**: If the task involves multiple independent event tables (e.g., Sends, Opens), **Favor** candidates using `UNION ALL` or `FULL JOIN` strategies over simple `INNER/LEFT JOIN` which might lose data.

4. **Consistency**:
- The SQL must strictly follow the Strategy. If the Strategy says "Filter X" but SQL does not, reject it.

*** OUTPUT INSTRUCTION ***
1. Analyze each candidate one by one based on the criteria above.
2. Compare the candidates (both the strategy and the SQL) to point out if some of them miss necessary filters (Entity Isolation) or lacks specific verified details.
3. Select the best candidate.
4. Output the chosen file name in this format:
```plaintext
candidate_<index>.sql
```
"""
