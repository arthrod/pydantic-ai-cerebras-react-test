# Cerebras Strict Mode Fix - Verification Report

## Problem
Cerebras API returned error 422:
```
Tools with mixed values for 'strict' are not allowed. 
Please set all tools to 'strict: true' or 'strict: false'
```

## Root Cause Analysis

### Package Source Code Verification

1. **ToolOutput class** (verified in `pydantic_ai/output.py:872-937`)
   - ✅ Supports `strict` parameter in constructor
   - ✅ Passes `strict` to generated ToolDefinition
   - Test proof:
     ```python
     tool_output = ToolOutput(MyModel, strict=True)
     # tool_output.strict == True
     ```

2. **@agent.tool() decorator** (verified in `pydantic_ai/agent/__init__.py:1153-1237`)
   - ✅ Accepts `strict: bool | None = None` parameter
   - ✅ Stores in Tool object's `strict` attribute
   - Test proof:
     ```python
     @agent.tool(strict=True)
     async def my_tool(...): ...
     # agent._function_toolset.tools['my_tool'].strict == True
     ```

3. **ToolDefinition class** (verified in `pydantic_ai/tools.py:472-502`)
   - ✅ Has `strict: bool | None = None` field
   - ✅ Used by both user tools and output tools
   - Documentation: "Whether to enforce vendor-specific strict JSON schema validation"

### The Issue
Pydantic AI creates TWO types of tools:
1. **User-defined tools** (from `@agent.tool()` decorators)
2. **Output tools** (automatically generated from `output_type`)

When these had mixed `strict` values, Cerebras rejected the request.

## Solution Applied

### Pattern 1: Pydantic Model Output
```python
from pydantic_ai.output import ToolOutput

pattern1_agent = Agent(
    get_model(),
    deps_type=Deps,
    output_type=ToolOutput(Pattern1Result, strict=True),  # ← Wrapped
    instructions=...,
)

@pattern1_agent.tool(strict=True)  # ← Added strict=True
async def _search(ctx: RunContext[Deps], query: str) -> list[dict[str, Any]]:
    ...
```

### Pattern 2: String Output
```python
# No ToolOutput needed for str output_type
pattern2_agent = Agent(
    get_model(),
    deps_type=Deps,
    output_type=str,  # ← No wrapper needed
    instructions=...,
)

@pattern2_agent.tool(strict=True)  # ← Added strict=True to all tools
async def __search(ctx: RunContext[Deps], query: str) -> list[dict[str, Any]]:
    ...
```

### Pattern 3: Pydantic Model Output
```python
pattern3_agent = Agent(
    get_model(),
    deps_type=Deps,
    output_type=ToolOutput(Pattern1Result, strict=True),  # ← Wrapped
    instructions=...,
)

@pattern3_agent.tool(strict=True)  # ← Added strict=True
async def ___search(ctx: RunContext[Deps], query: str) -> list[dict[str, Any]]:
    ...
```

### Pattern 4: Union of Pydantic Models
```python
DecisionOutputSpec: list[ToolOutput[ContinueAction] | ToolOutput[FinishAction]] = [
    ToolOutput(ContinueAction, strict=True),  # ← Wrapped each type
    ToolOutput(FinishAction, strict=True)
]

pattern4_agent = Agent(
    get_model(),
    deps_type=Deps,
    output_type=DecisionOutputSpec,
    instructions=...,
)

@pattern4_agent.tool(strict=True)  # ← Added strict=True
async def search(ctx: RunContext[Deps], query: str) -> list[dict[str, Any]]:
    ...
```

## Verification Results

All patterns now have **consistent strict=True** across ALL tools:

### Pattern 1
- ✅ User tool '_search': strict=True
- ✅ User tool '_fetch': strict=True
- ✅ User tool '_save': strict=True
- ✅ Output tool 'final_result': strict=True

### Pattern 2
- ✅ User tool '__search': strict=True
- ✅ User tool '__fetch': strict=True
- ✅ User tool '____save': strict=True
- ✅ User tool 'finish': strict=True
- ℹ️  No output toolset (output_type=str)

### Pattern 3
- ✅ User tool '___search': strict=True
- ✅ User tool '___fetch': strict=True
- ✅ User tool '__save': strict=True
- ✅ Output tool 'final_result': strict=True

### Pattern 4
- ✅ User tool 'search': strict=True
- ✅ User tool 'fetch': strict=True
- ✅ User tool 'save': strict=True
- ✅ Output tool 'final_result_ContinueAction': strict=True
- ✅ Output tool 'final_result_FinishAction': strict=True

## Key Learnings

1. **ToolOutput wrapper** is used when `output_type` is a Pydantic model
2. **No wrapper needed** when `output_type=str` (no output tools generated)
3. **All user tools** must also have `@agent.tool(strict=True)`
4. **Union outputs** need each type wrapped: `[ToolOutput(T1, strict=True), ToolOutput(T2, strict=True)]`

## References from Pydantic AI Source

Example from official code (`pydantic_ai/tools.py:116-121`):
```python
async def turn_on_strict_if_openai(
    ctx: RunContext[None], tool_defs: list[ToolDefinition]
) -> list[ToolDefinition] | None:
    if ctx.model.system == 'openai':
        return [replace(tool_def, strict=True) for tool_def in tool_defs]
    return tool_defs
```

This shows that `strict=True` is the intended way to enforce strict mode across all tools.
