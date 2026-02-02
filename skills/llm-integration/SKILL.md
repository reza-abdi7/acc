---
name: llm-integration
description: LLM API integration patterns, prompt engineering, structured outputs, and multi-provider support. Use when implementing AI analysis features.
---

# LLM Integration

Guide for integrating Large Language Models into applications with proper abstraction, error handling, and prompt engineering.

## When to Use This Skill

- Integrating LLM APIs (OpenAI, Anthropic, local models)
- Designing prompts for legal text analysis
- Handling structured outputs
- Managing API errors and rate limits

## Provider Abstraction

```python
from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import TypeVar, Generic

T = TypeVar("T", bound=BaseModel)

class LLMProvider(ABC):
    @abstractmethod
    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        pass
    
    @abstractmethod
    async def complete_structured(
        self,
        prompt: str,
        response_model: type[T],
        system_prompt: str | None = None,
    ) -> T:
        """Get response matching Pydantic model."""
        pass

class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "claude-3-opus-20240229"):
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model
    
    async def complete(self, prompt: str, **kwargs) -> str:
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", 4096),
            temperature=kwargs.get("temperature", 0.0),
            system=kwargs.get("system_prompt", ""),
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "gpt-4-turbo"):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model
    
    async def complete(self, prompt: str, **kwargs) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 4096),
            messages=[
                {"role": "system", "content": kwargs.get("system_prompt", "")},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content
```

## Structured Output

```python
import json
from pydantic import BaseModel, ValidationError

class StructuredOutputMixin:
    async def complete_structured(
        self,
        prompt: str,
        response_model: type[T],
        system_prompt: str | None = None,
    ) -> T:
        # Add JSON instruction to prompt
        json_prompt = f"""{prompt}

Respond with valid JSON matching this schema:
{json.dumps(response_model.model_json_schema(), indent=2)}

JSON Response:"""
        
        response = await self.complete(
            prompt=json_prompt,
            system_prompt=system_prompt or "You are a precise assistant. Always respond with valid JSON.",
            temperature=0.0,
        )
        
        # Parse and validate
        try:
            # Extract JSON from response
            json_str = self._extract_json(response)
            data = json.loads(json_str)
            return response_model.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            raise LLMParseError(f"Failed to parse response: {e}")
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from text, handling markdown code blocks."""
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            return text[start:end].strip()
        if "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            return text[start:end].strip()
        return text.strip()
```

## Prompt Engineering

### System Prompts

```python
LEGAL_ANALYST_SYSTEM = """You are a legal document analyst specializing in Terms & Conditions analysis.

Your task is to:
1. Identify clauses relevant to data usage, scraping, and AI
2. Classify each clause as permission, prohibition, or condition
3. Be conservative - when uncertain, flag for human review

Always respond with structured JSON. Never make assumptions about missing information."""
```

### Analysis Prompt Template

```python
ANALYSIS_PROMPT = """Analyze the following Terms & Conditions for compliance signals.

Document:
{document_text}

Extract the following signals:
1. scraping_allowed: Can data be programmatically extracted?
2. commercial_use_allowed: Can data be used commercially?
3. ai_training_allowed: Can data be used for AI/ML training?
4. attribution_required: Must the source be cited?

For each signal, provide:
- value: "allowed", "prohibited", "conditional", or "unclear"
- confidence: 0.0 to 1.0
- source_clause: The exact text supporting this determination
- conditions: Any conditions that apply (if conditional)

If a topic is not addressed in the document, mark as "unclear" with confidence 0.5."""
```

## Error Handling

```python
class LLMError(Exception):
    """Base LLM error."""
    pass

class LLMRateLimitError(LLMError):
    def __init__(self, retry_after: int | None = None):
        self.retry_after = retry_after
        super().__init__("Rate limit exceeded")

class LLMParseError(LLMError):
    """Failed to parse LLM response."""
    pass

async def call_with_retry(
    provider: LLMProvider,
    prompt: str,
    max_retries: int = 3,
) -> str:
    for attempt in range(max_retries):
        try:
            return await provider.complete(prompt)
        except LLMRateLimitError as e:
            if attempt == max_retries - 1:
                raise
            wait = e.retry_after or (2 ** attempt)
            await asyncio.sleep(wait)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)
```

## Best Practices

### DO:
- **Use low temperature** (0.0-0.2) for consistent analysis
- **Request structured output** (JSON) for reliable parsing
- **Validate responses** with Pydantic models
- **Handle rate limits** with exponential backoff
- **Log prompts and responses** for debugging

### DON'T:
- **Trust output blindly** - always validate
- **Use high temperature** for factual analysis
- **Hardcode prompts** - use templates
- **Ignore token limits** - chunk large documents
