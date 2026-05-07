"""Prompt templates for Stage A — Schema Linking.

Verbatim from APEX-SQL paper appendix (Instruction Template.md), with the
following normalizations so that Python `.format()` works:
  - JSON example braces are escaped as `{{` / `}}`.
  - Inline pseudo-variables in F_SL_FINAL/F_SL_EXP that were row-templates in
    the paper are replaced by a pre-rendered `{tables_repr}` / `{columns_repr}`.
  - F_SL_AGG's `{{len(candidates)}}` (literal) is renamed to `{n_plans}`.
  - F_SL_FINAL drops the multi-round `INTERACTIVE PROCESS` (Round 2 work).

Required keys per template:

  F_SL_PLAN          {question}
  F_SL_AGG           {question, n_plans, plans_text}
  F_SL_DEL           {question, logical_plan, schema, evidence}
  F_SL_SEL           {question, logical_plan, schema, evidence}
  F_SL_SEMANTICS     {CRITICAL_RULES, question, logical_plan, evidence, schema}
  F_SL_EXP           {CRITICAL_RULES, table_name, columns_repr, question,
                      semantic_role, evidence}
  F_SL_FINAL         {question, evidence, db_summary, tables_repr}
"""

F_SL_PLAN = """*** TASK CONTEXT ***
You are a Lead Data Architect. Your task is to break down
the User Question into abstract logical steps needed to
answer it.

**IMPORTANT**: Do NOT reference specific table or
column names yet. Focus purely on the logic (e.g., filter,
join, count, aggregate).

*** USER QUESTION ***
{question}

*** OUTPUT FORMAT ***
{{
"logical_steps": [
"1. Identify [Entity]...",
"2. Filter where [Condition]...",
"3. Link [Entity A] to [Entity B]...",
"4. Calculate [Aggregation]..."
]
}}
"""

F_SL_AGG = """*** TASK CONTEXT ***
We have collected {n_plans} draft logical plans.
Synthesize them into a single, comprehensive Master
Logical Plan. Ensure the steps cover all conditions, filters,
joins, and aggregations required.

*** USER QUESTION ***
{question}

*** DRAFT PLANS ***
{plans_text}

Output just the steps as a numbered list.
"""

F_SL_DEL = """*** TASK CONTEXT ***
You are a Lead Data Architect. You have a Logical Plan to
answer a query.
Your task: **Negative Pruning**. Identify database tables or columns that are **100% IRRELEVANT** to the plan.

*** USER QUESTION ***
{question}
*** MASTER LOGICAL PLAN ***
{logical_plan}
*** FULL DATABASE SCHEMA ***
{schema}
*** EVIDENCE ***
{evidence}

*** STRICT GUIDELINES ***
1. **High Recall (Safety)**:
- If the column name is related to the query (even 1%
chance), you should keep it. If not, check the desciption to
see if it is related to the query. Sometimes the description is
not clear, then you should pay close attention to the sample
rows of the table. If the sample values of some columns
are related to the query, you should keep these columns. If
all of these information are not clear enough, remove it.

2. **Definition of Relevance**: Relevance includes both
**Lexical Matching** and **Semantic Relatedness** over
column name and description.
- **Lexical**: If a word from the query appears in the
name (e.g., query mentions "school" -> keep `school_code`,
`school_type`, etc.), it MUST be retained.
- **Semantic**: Keep columns conceptually related to
the topic. For example, if the query asks about "patents that
were granted in ...", then the column `grant_date` should
be kept.
- **CRITICAL**: Do NOT remove discriminator columns
such as `xxx_id`, `xxx_name`, `xxx_code`, or `xxx_type` if
the table itself is kept.

3. **Output Removal List**:
- **Tables**: If a whole table is irrelevant, list it in `obviously_irrelevant_tables`. Then all columns of that table will
be removed. You do NOT need to list its columns separately.
- **Columns**: If specific columns of a table are noise,
list them in `obviously_irrelevant_columns`.

4. **Grouped Tables**: If multiple tables are presented as
sharing the same columns, you MUST list the removal
instructions for **EACH** table explicitly. Pay close
attention to name differences within the group (e.g.,
xx_2017 vs xx_2026), as these reflect specific data dimensions (like time) that determine relevance to the query.

*** OUTPUT FORMAT ***
```json
{{
"obviously_irrelevant_tables": ["table_unused_1", "table_unused_2"],
"obviously_irrelevant_columns": [
{{
"table": "t1",
"columns": ["col_unused_1", "col_unused_2"]
}}
]
}}
```
"""

F_SL_SEL = """*** TASK CONTEXT ***
You are a Lead Data Architect. You have a Logical Plan to
answer a query.
Your task: **Positive Selection**. Identify database tables
or columns that are **RELEVANT** or **NECESSARY** to
the plan.

*** USER QUESTION ***
{question}
*** MASTER LOGICAL PLAN ***
{logical_plan}
*** FULL DATABASE SCHEMA ***
{schema}
*** EVIDENCE ***
{evidence}

*** STRICT GUIDELINES ***
1. **High Recall (Safety)**: Select ALL columns that might
be useful for joining, filtering, grouping, or returning results. If you are not sure about the relevance of a column,
e.g., the name and the description are ambiguous, **PICK
IT**.

2. **Definition of Relevance**: Relevance includes both
**Lexical Matching** and **Semantic Relatedness** over
column name and description.
- **Lexical**: If a word from the query appears in the table or column name (e.g., query mentions "school" -> keep
`school_code`, `school_type`, etc.), it MUST be selected.
- **Semantic**: Identify columns conceptually related to
the topic. For example, if the query asks about "patents that
were granted in ...", then the column `grant_date` should
be kept.
- **Discriminators**: ALWAYS select primary keys and
common identifiers (`xxx_id`, `xxx_code`, `xxx_name`) for
relevant tables, as they are needed for joins.

3. **Output Selection List**:
- **Tables**: If a whole table is relevant, list it in `relevant_tables`.
- **Columns**: List specific useful columns in `relevant_columns`. If a table is already listed in `relevant_tables`, the columns can be omitted.

4. **Grouped Tables**: If multiple tables are presented as
sharing the same columns, you MUST list the selection
instructions for **EACH** table explicitly. Pay close
attention to name differences within the group (e.g.,
xx_2017 vs xx_2026), as these reflect specific data dimensions (like time) that determine relevance to the query.

*** OUTPUT FORMAT ***
```json
{{
"relevant_tables": ["table_useful_1"],
"relevant_columns": [
{{
"table": "t1",
"columns": ["col_useful_1", "col_pk_id"]
}}
]
}}
```
"""

F_SL_SEMANTICS = """*** TASK CONTEXT ***
You are a Senior Data Architect. You have full visibility of
the database schema and a user question.
Your goal is to perform **Semantic Linking**: Analyze the
database structure and how it grounds the user's intent.
{CRITICAL_RULES}

*** USER QUESTION ***
{question}
*** Logical Plan ***
{logical_plan}
*** Evidence ***
{evidence}
*** DATABASE SCHEMA ***
{schema}

*** YOUR TASKS ***
1. **Database Structure Overview**: Describe the database
structure in detail (e.g., 'A banking system with customers
and transactions...').

2. **Query-Specific Content Analysis**: Analyze the query
against the available columns. Identify which columns are
likely targets, filters, or join keys.

3. **Table Functional Analysis**: For EVERY potentially
relevant table, describe its specific function regarding this
query.
- Is it a **Target Table**? (Contains the answer columns)
- Is it a **Bridge Table**? (Doesn't have semantic data
but is needed to join Table A and Table B via Foreign Keys)
- Is it a **Filtering Table**? (Contains columns for
WHERE clauses)
- **CRITICAL**: A table may have multiple roles. If a
table is needed as a BRIDGE, you MUST explicitly state
that it connects Entity X and Entity Y, even if it looks
empty of content.

*** OUTPUT FORMAT ***
```json
{{
"database_structure": "Database structure overview...",
"query_specific_content_analysis": "Detailed mapping of query terms to DB columns/logic...",
"table_functions": {{
"table_name_1": "Acts as a bridge table connecting Students and Classes via student_id and class_id.",
"table_name_2": "Contains the 'score' column needed for calculation and 'exam_date' for filtering."
}}
}}
```
Perform the semantic linking analysis:
"""

F_SL_EXP = """*** TASK CONTEXT ***
You are an agent exploring a database table to verify its
relevance to a user question.
You must not explore randomly. You must verify if this
table fits its anticipated role.
{CRITICAL_RULES}

*** TARGET TABLE: {table_name} ***
Columns:
{columns_repr}

*** USER QUESTION ***
{question}

*** ANTICIPATED ROLE ***
This table was identified as: {semantic_role}. Use this to
guide your exploration.

*** Evidence ***
{evidence}

*** YOUR MISSION ***
Generate 3-8 Snowflake-dialect SQL queries to investigate. **Focus on
understanding the table's semantics and utility.**

**Motivation for Exploration**:
1. **Semantic Alignment**: Check distinct values to understand what the column *means* versus what the query
*needs*. (e.g., If column is 'type', does it contain the specific
categories? If 'status', does it contain values like 'Active'
or code '1'?)
2. **Granularity & Scope**: Verify the table's grain (e.g., is
it one row per Order or per Item?). This determines if it
supports the required aggregations.
3. **Bridge/Connectivity**: If this looks like a linking table,
verify the Foreign Keys are populated (not all NULL) to
ensure it can actually serve as a bridge.
4. **Data Quality**: Are critical columns (targets for filters
or answers) usable, or are they mostly NULL?

*** OUTPUT FORMAT ***
Provide SQL queries in a single `sql` block with comments
explaining the *motivation*.
```sql
-- Motivation: Checking distinct values in 'status' to see if
-- it aligns with the query's filter requirement
SELECT DISTINCT status FROM table_name LIMIT 10;
```
Generate your exploration queries:
"""

F_SL_FINAL = """*** TASK CONTEXT ***
You are the Lead Data Architect. We are synthesizing initial
exploration findings.
Review the [MARKED RELEVANT] and [MARKED
IRRELEVANT] tables. Fix blind spots.

*** USER QUESTION ***
{question}

*** EVIDENCE ***
{evidence}

*** SEMANTIC ANALYSIS ***
{db_summary}

*** SCHEMA STATUS ***
{tables_repr}

*** YOUR MISSION ***
Determine the final list of columns required to write the
SQL query.
You must ensure the selected columns form a connected
graph (tables can be joined) and cover all functional
requirements of the query.

*** SELECTION CRITERIA (FUNCTIONALITY) ***
Keep a column if it serves one of the following purposes:
1. **Identification**: Unique identifiers (IDs, Codes) needed
to count or distinguish entities (Primary keys).
2. **Linking**: Columns needed to join two tables together
(Foreign Keys).
3. **Filtering**: Columns involved in conditions (e.g., status='Active', date > 2023).
4. **Aggregation**: Numerical columns for calculations
(Sum, Avg, Max, Min).
5. **Grouping & Sorting**: Columns used for 'GROUP BY'
or 'ORDER BY'.
6. **Direct Result**: Columns explicitly requested in the
output.

**Note on Multi-Path**: If multiple columns might serve the
same purpose, KEEP ALL OF THEM. Alternative columns
might help to construct another solution paths.
**Note on Type of Entity**: DO NOT guess the type of an
unspecified entity even you have some prior knowledge,
e.g., if the query contains location entity like 'Riverside',
then ALL columns related to location (e.g., County, District,
etc.) should be kept. Another example is 'Fresno County Office of Education' which is actually a full name of a district.

*** REJECTION REQUIREMENTS ***
If a column was marked as **[MARKED RELEVANT]**
in the Schema Status but you decide to **REJECT** it,
you MUST include it in the `rejected_candidates` list with
a `reject_reason` explaining why it is unnecessary. You
can NOT reject a column for the reason that it is only a
potentially useful column.

*** OUTPUT FORMAT ***
You MUST explicitly list rejected candidates to prove you
considered them.
**IMPORTANT**: In 'rejected_candidates', ONLY list
columns that were previously marked RELEVANT but you
decided to reject, OR columns that look ambiguous. Do
NOT list obviously irrelevant columns to save space.
```json
{{
"refined_schema": {{
"table_name": {{
"relevant_columns": [
{{
"column_name": "...",
"relevance_reason": "Functional reason (e.g., Needed for Filtering)"
}}
]
}}
}},
"rejected_candidates": [
{{
"table": "t1",
"column": "c1",
"reject_reason": "Originally marked relevant, but rejected because..."
}}
]
}}
```
Begin refinement:
"""
