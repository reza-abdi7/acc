---
name: ai-research-tutor
description: Guides AI through systematic research methodology when investigating new topics, technologies, or solutions. Use when exploring unfamiliar domains, evaluating options, or learning about new concepts.
---

# AI Research Tutor

This skill provides a structured approach for AI to conduct research systematically, ensuring thorough investigation, proper documentation, and actionable outcomes.

## When to Use This Skill

- Investigating new technologies, libraries, or frameworks
- Evaluating multiple solutions or approaches
- Learning about unfamiliar domains or concepts
- Gathering information to make informed decisions
- Understanding best practices in a new area

## Research Methodology

### Phase 1: Define the Research Scope

Before starting any research:

1. **Clarify the objective**: What specific question needs to be answered?
2. **Identify constraints**: Time, technology stack, organizational requirements
3. **Define success criteria**: What would a good answer look like?
4. **List known unknowns**: What do we already know we don't know?

### Phase 2: Information Gathering

#### Primary Sources (Prioritize in this order)

1. **Official documentation**: Always start with official docs
2. **Source code**: When available, read the actual implementation
3. **Academic papers**: For theoretical foundations
4. **Reputable technical blogs**: From known experts or organizations
5. **Community discussions**: GitHub issues, Stack Overflow (verify recency)

#### Source Evaluation Criteria

- **Recency**: Is the information current? Check publication dates
- **Authority**: Who wrote it? What are their credentials?
- **Accuracy**: Can claims be verified from multiple sources?
- **Relevance**: Does it directly address our question?
- **Bias**: Is there commercial or personal bias?

### Phase 3: Analysis and Synthesis

1. **Cross-reference findings**: Verify information across multiple sources
2. **Identify patterns**: What do multiple sources agree on?
3. **Note contradictions**: Where do sources disagree? Why?
4. **Assess applicability**: How does this apply to our specific context?

### Phase 4: Documentation

Document findings with:

- **Summary**: Key findings in 2-3 sentences
- **Details**: Comprehensive explanation
- **Sources**: Links and references
- **Limitations**: What we still don't know
- **Recommendations**: Actionable next steps

## Research Anti-Patterns to Avoid

### DO NOT:

- **Assume without verification**: Always verify claims
- **Rely on single sources**: Cross-reference everything
- **Ignore publication dates**: Technology changes rapidly
- **Skip official documentation**: It's usually the most accurate
- **Confuse popularity with correctness**: Popular answers can be wrong
- **Make up information**: If you don't know, say so clearly

### DO:

- **Acknowledge uncertainty**: "Based on available information..." or "I'm not certain, but..."
- **Provide confidence levels**: High/Medium/Low confidence in findings
- **Suggest verification steps**: How the user can verify your findings
- **Recommend further research**: When the topic needs deeper investigation

## Structured Research Output Template

When presenting research findings, use this structure:

```markdown
## Research: [Topic]

### Objective
[What we're trying to learn]

### Key Findings
1. [Finding 1]
2. [Finding 2]
3. [Finding 3]

### Detailed Analysis
[Comprehensive explanation]

### Sources
- [Source 1 with link]
- [Source 2 with link]

### Confidence Level
[High/Medium/Low] - [Explanation of confidence]

### Limitations
- [What we don't know]
- [Areas needing more research]

### Recommendations
1. [Action item 1]
2. [Action item 2]
```

## Domain-Specific Research Guidelines

### For Technology/Library Evaluation

- Check GitHub stars, issues, last commit date
- Review the maintainer's track record
- Look for production usage examples
- Assess documentation quality
- Check for security vulnerabilities

### For Architecture Decisions

- Look for case studies from similar organizations
- Consider scalability implications
- Evaluate operational complexity
- Assess team expertise requirements

### For Best Practices

- Prioritize official style guides
- Look for industry standards (RFCs, specifications)
- Consider context-specific variations
- Note when practices are opinionated vs. universal

## Asking Clarifying Questions

When research scope is unclear, ask:

1. What is the specific problem you're trying to solve?
2. What constraints exist (time, budget, technology)?
3. What have you already tried or considered?
4. What would an ideal solution look like?
5. Are there any non-negotiable requirements?

## Continuous Learning Mindset

- **Stay curious**: Ask "why" not just "what"
- **Challenge assumptions**: Including your own
- **Update knowledge**: Be willing to revise understanding
- **Share uncertainty**: It's better to be honest than wrong
