---
name: acc-ai-analysis
description: AI-based legal text analysis, T&C classification, compliance rule application, and decision generation. Use when implementing or debugging the AI Analyzer component.
---

# ACC AI Analysis

The AI Analyzer component processes fetched legal documents (T&C, robots.txt, LLM.txt), extracts compliance-relevant signals, applies classification rules, and produces compliance decisions.

## When to Use This Skill

- Implementing AI analysis logic
- Designing prompts for legal text analysis
- Integrating with LLM providers
- Applying compliance rules
- Debugging classification issues
- Understanding decision confidence

## AI Analyzer Responsibilities

1. **Document Retrieval**: Query DB for fetched documents
2. **Clause Extraction**: Identify relevant legal clauses
3. **Signal Detection**: Extract permissions, prohibitions, conditions
4. **Rule Application**: Apply compliance rules from Compliance Manager
5. **Classification**: Label and categorize T&C content
6. **Decision Generation**: Produce final compliance decision with confidence

## Analysis Pipeline

```
┌─────────────────┐
│ Fetched Docs    │
│ (T&C, robots,   │
│  LLM.txt)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Pre-processing  │
│ - Chunking      │
│ - Cleaning      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Clause          │
│ Extraction      │
│ (LLM-based)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Signal          │
│ Detection       │
│ (LLM-based)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Rule            │
│ Application     │
│ (Logic-based)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Decision        │
│ Synthesis       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Compliance      │
│ Decision        │
└─────────────────┘
```

## Compliance Signals

### Signal Categories

| Category | Signals |
|----------|---------|
| **Usage** | scraping_allowed, crawling_allowed, api_access_allowed |
| **Commercial** | commercial_use_allowed, resale_allowed, derivative_works |
| **Attribution** | attribution_required, source_citation_required |
| **AI/ML** | ai_training_allowed, llm_use_allowed, model_training |
| **Data** | data_storage_allowed, data_processing_allowed, retention_limits |
| **Geographic** | geo_restrictions, jurisdiction_requirements |
| **Temporal** | time_limits, update_frequency_requirements |

### Signal Structure

```python
from pydantic import BaseModel
from typing import Literal, Optional

class ComplianceSignal(BaseModel):
    category: str
    signal_type: str
    value: Literal["allowed", "prohibited", "conditional", "unclear"]
    confidence: float  # 0.0 to 1.0
    source_clause: Optional[str]  # Original text that led to this signal
    conditions: list[str] = []  # If conditional, what are the conditions
    
class SignalSet(BaseModel):
    source_id: str
    signals: list[ComplianceSignal]
    analysis_timestamp: datetime
    model_version: str
```

## LLM Integration Patterns

### Provider Abstraction

```python
from abc import ABC, abstractmethod
from typing import AsyncGenerator

class LLMProvider(ABC):
    @abstractmethod
    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        pass
    
    @abstractmethod
    async def complete_structured(
        self,
        prompt: str,
        response_schema: type[BaseModel],
        system_prompt: Optional[str] = None,
    ) -> BaseModel:
        """Get structured output matching schema."""
        pass

class AnthropicProvider(LLMProvider):
    async def complete(self, prompt: str, **kwargs) -> str:
        response = await self.client.messages.create(
            model="claude-3-opus-20240229",
            messages=[{"role": "user", "content": prompt}],
            system=kwargs.get("system_prompt"),
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 4096),
        )
        return response.content[0].text

class OpenAIProvider(LLMProvider):
    async def complete(self, prompt: str, **kwargs) -> str:
        response = await self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": kwargs.get("system_prompt", "")},
                {"role": "user", "content": prompt},
            ],
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 4096),
        )
        return response.choices[0].message.content
```

### Prompt Templates

#### Clause Extraction Prompt

```python
CLAUSE_EXTRACTION_PROMPT = """
You are a legal document analyst. Extract clauses from the following Terms & Conditions that are relevant to:
1. Data scraping and crawling permissions
2. Commercial use of data
3. AI and machine learning usage
4. Attribution requirements
5. Data storage and processing

For each relevant clause:
- Quote the exact text
- Categorize it (scraping, commercial, ai, attribution, data)
- Indicate if it's a permission, prohibition, or condition

Terms & Conditions:
{tc_content}

Respond in JSON format:
{{
    "clauses": [
        {{
            "text": "exact quote from document",
            "category": "category name",
            "type": "permission|prohibition|condition",
            "summary": "brief plain-language summary"
        }}
    ],
    "no_relevant_clauses": false,
    "document_language": "detected language"
}}
"""
```

#### Signal Detection Prompt

```python
SIGNAL_DETECTION_PROMPT = """
You are a compliance analyst. Based on the following extracted clauses from a website's Terms & Conditions, determine the compliance signals.

Extracted Clauses:
{clauses_json}

For each signal category, determine:
- Whether the activity is ALLOWED, PROHIBITED, CONDITIONAL, or UNCLEAR
- Your confidence level (0.0 to 1.0)
- Any conditions that apply
- The source clause that supports your determination

Signal Categories to Evaluate:
1. scraping_allowed - Can we programmatically access and extract data?
2. commercial_use_allowed - Can we use the data for commercial purposes?
3. ai_training_allowed - Can we use the data to train AI models?
4. attribution_required - Must we cite the source?
5. data_storage_allowed - Can we store the data?

Respond in JSON format:
{{
    "signals": [
        {{
            "category": "signal category",
            "value": "allowed|prohibited|conditional|unclear",
            "confidence": 0.85,
            "conditions": ["condition 1", "condition 2"],
            "source_clause": "the clause text that supports this"
        }}
    ]
}}

Important:
- If a clause is ambiguous, mark as "unclear" with lower confidence
- If no clause addresses a category, mark as "unclear" with confidence 0.5
- Be conservative: when in doubt, lean toward "prohibited" or "conditional"
"""
```

### Structured Output Parsing

```python
from pydantic import BaseModel, Field

class ExtractedClause(BaseModel):
    text: str
    category: str
    type: Literal["permission", "prohibition", "condition"]
    summary: str

class ClauseExtractionResult(BaseModel):
    clauses: list[ExtractedClause]
    no_relevant_clauses: bool = False
    document_language: str = "unknown"

async def extract_clauses(tc_content: str, llm: LLMProvider) -> ClauseExtractionResult:
    """Extract relevant clauses using LLM."""
    prompt = CLAUSE_EXTRACTION_PROMPT.format(tc_content=tc_content)
    
    result = await llm.complete_structured(
        prompt=prompt,
        response_schema=ClauseExtractionResult,
        system_prompt="You are a precise legal document analyst. Always respond with valid JSON.",
    )
    
    return result
```

## Rule Application

### Rule Structure

```python
class ComplianceRule(BaseModel):
    id: str
    name: str
    description: str
    conditions: list[RuleCondition]
    outcome: RuleOutcome
    priority: int  # Higher priority rules evaluated first
    
class RuleCondition(BaseModel):
    signal: str  # e.g., "scraping_allowed"
    operator: Literal["equals", "not_equals", "in", "not_in"]
    value: Any
    
class RuleOutcome(BaseModel):
    status: Literal["ALLOWED", "CONDITIONAL", "BLOCKED", "REVIEW_REQUIRED"]
    constraints: list[str] = []
    requires_review: bool = False
```

### Rule Engine

```python
class RuleEngine:
    def __init__(self, rules: list[ComplianceRule]):
        # Sort by priority (highest first)
        self.rules = sorted(rules, key=lambda r: r.priority, reverse=True)
    
    def evaluate(self, signals: SignalSet) -> ComplianceDecision:
        """Evaluate signals against rules to produce decision."""
        signal_map = {s.signal_type: s for s in signals.signals}
        
        matched_rules = []
        for rule in self.rules:
            if self._rule_matches(rule, signal_map):
                matched_rules.append(rule)
        
        if not matched_rules:
            # No rules matched - requires review
            return ComplianceDecision(
                status="REVIEW_REQUIRED",
                confidence=0.5,
                reason="No matching rules",
            )
        
        # Use highest priority matched rule
        primary_rule = matched_rules[0]
        
        # Calculate confidence based on signal confidences
        relevant_signals = [
            signal_map[c.signal] 
            for c in primary_rule.conditions 
            if c.signal in signal_map
        ]
        avg_confidence = sum(s.confidence for s in relevant_signals) / len(relevant_signals)
        
        return ComplianceDecision(
            status=primary_rule.outcome.status,
            constraints=primary_rule.outcome.constraints,
            confidence=avg_confidence,
            matched_rule=primary_rule.id,
            requires_review=primary_rule.outcome.requires_review or avg_confidence < 0.7,
        )
    
    def _rule_matches(self, rule: ComplianceRule, signals: dict) -> bool:
        """Check if all rule conditions are met."""
        for condition in rule.conditions:
            if condition.signal not in signals:
                return False
            
            signal = signals[condition.signal]
            if not self._condition_matches(condition, signal):
                return False
        
        return True
```

### Example Rules

```python
COMPLIANCE_RULES = [
    ComplianceRule(
        id="rule_001",
        name="Explicit AI Prohibition",
        description="Block if AI training is explicitly prohibited",
        conditions=[
            RuleCondition(signal="ai_training_allowed", operator="equals", value="prohibited"),
        ],
        outcome=RuleOutcome(status="BLOCKED", constraints=["AI training prohibited"]),
        priority=100,
    ),
    ComplianceRule(
        id="rule_002",
        name="Scraping Prohibited",
        description="Block if scraping is explicitly prohibited",
        conditions=[
            RuleCondition(signal="scraping_allowed", operator="equals", value="prohibited"),
        ],
        outcome=RuleOutcome(status="BLOCKED", constraints=["Scraping prohibited"]),
        priority=90,
    ),
    ComplianceRule(
        id="rule_003",
        name="Conditional with Attribution",
        description="Allow with attribution requirement",
        conditions=[
            RuleCondition(signal="scraping_allowed", operator="in", value=["allowed", "conditional"]),
            RuleCondition(signal="attribution_required", operator="equals", value="allowed"),
        ],
        outcome=RuleOutcome(
            status="CONDITIONAL",
            constraints=["Attribution required"],
        ),
        priority=50,
    ),
    ComplianceRule(
        id="rule_004",
        name="Fully Allowed",
        description="Allow if scraping and commercial use permitted",
        conditions=[
            RuleCondition(signal="scraping_allowed", operator="equals", value="allowed"),
            RuleCondition(signal="commercial_use_allowed", operator="equals", value="allowed"),
        ],
        outcome=RuleOutcome(status="ALLOWED"),
        priority=40,
    ),
]
```

## Decision Synthesis

```python
class DecisionSynthesizer:
    def __init__(self, rule_engine: RuleEngine):
        self.rule_engine = rule_engine
    
    async def synthesize(
        self,
        source_id: str,
        tc_signals: SignalSet,
        robots_info: Optional[RobotsInfo],
        llm_txt_info: Optional[LLMTxtInfo],
    ) -> ComplianceDecision:
        """Synthesize final decision from all inputs."""
        
        # Start with T&C-based decision
        tc_decision = self.rule_engine.evaluate(tc_signals)
        
        # Apply robots.txt overrides
        if robots_info and robots_info.has_ai_restrictions:
            if tc_decision.status == "ALLOWED":
                tc_decision = tc_decision.copy(
                    status="CONDITIONAL",
                    constraints=tc_decision.constraints + ["robots.txt AI restrictions"],
                )
        
        # Apply LLM.txt if available
        if llm_txt_info:
            # LLM.txt can further restrict but not loosen
            if llm_txt_info.prohibits_use and tc_decision.status != "BLOCKED":
                tc_decision = tc_decision.copy(
                    status="BLOCKED",
                    constraints=["LLM.txt prohibits use"],
                )
        
        # Determine if review is needed
        requires_review = (
            tc_decision.requires_review
            or tc_decision.confidence < 0.7
            or tc_decision.status == "REVIEW_REQUIRED"
        )
        
        if requires_review and tc_decision.status != "BLOCKED":
            tc_decision = tc_decision.copy(status="REVIEW_REQUIRED")
        
        return ComplianceDecision(
            source_id=source_id,
            status=tc_decision.status,
            compliance_channel=self._determine_channel(tc_decision),
            constraints=tc_decision.constraints,
            confidence=tc_decision.confidence,
            review_flag=requires_review,
            version=DecisionVersion(
                model_version=settings.model_version,
                rules_version=settings.rules_version,
                timestamp=datetime.utcnow(),
            ),
        )
```

## Confidence Calculation

```python
def calculate_confidence(signals: list[ComplianceSignal]) -> float:
    """Calculate overall confidence from individual signal confidences."""
    if not signals:
        return 0.5  # Default uncertainty
    
    # Weight by signal importance
    weights = {
        "scraping_allowed": 1.5,
        "ai_training_allowed": 1.5,
        "commercial_use_allowed": 1.2,
        "attribution_required": 1.0,
        "data_storage_allowed": 1.0,
    }
    
    weighted_sum = 0.0
    total_weight = 0.0
    
    for signal in signals:
        weight = weights.get(signal.signal_type, 1.0)
        weighted_sum += signal.confidence * weight
        total_weight += weight
    
    return weighted_sum / total_weight if total_weight > 0 else 0.5
```

## Error Handling

```python
class AnalysisError(Exception):
    """Base analysis error."""
    pass

class LLMError(AnalysisError):
    """LLM API error."""
    def __init__(self, provider: str, reason: str):
        self.provider = provider
        self.reason = reason
        super().__init__(f"LLM error ({provider}): {reason}")

class ParseError(AnalysisError):
    """Failed to parse LLM response."""
    pass

class InsufficientDataError(AnalysisError):
    """Not enough data to make decision."""
    pass
```

## Testing AI Analysis

```python
@pytest.fixture
def mock_llm(mocker):
    """Mock LLM provider."""
    mock = mocker.AsyncMock(spec=LLMProvider)
    return mock

async def test_clause_extraction(mock_llm):
    mock_llm.complete_structured.return_value = ClauseExtractionResult(
        clauses=[
            ExtractedClause(
                text="You may not scrape this website",
                category="scraping",
                type="prohibition",
                summary="Scraping prohibited",
            )
        ],
        no_relevant_clauses=False,
    )
    
    analyzer = AIAnalyzer(llm=mock_llm)
    result = await analyzer.extract_clauses("Sample T&C content")
    
    assert len(result.clauses) == 1
    assert result.clauses[0].type == "prohibition"

async def test_decision_blocked_on_scraping_prohibition():
    signals = SignalSet(
        source_id="test",
        signals=[
            ComplianceSignal(
                category="usage",
                signal_type="scraping_allowed",
                value="prohibited",
                confidence=0.95,
            )
        ],
    )
    
    engine = RuleEngine(COMPLIANCE_RULES)
    decision = engine.evaluate(signals)
    
    assert decision.status == "BLOCKED"
```

## Best Practices

### DO:
- **Use low temperature** (0.0-0.2) for consistent analysis
- **Request structured output** (JSON) for reliable parsing
- **Include confidence scores** in all signals
- **Preserve source clauses** for auditability
- **Apply rules conservatively** - when in doubt, require review

### DON'T:
- **Trust LLM output blindly** - validate and verify
- **Skip edge cases** - handle unclear/missing signals
- **Ignore confidence** - low confidence should trigger review
- **Hardcode rules** - keep them configurable
- **Lose provenance** - always track which clause led to which signal
