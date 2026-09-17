# Token Usage and Cost Analysis Report

Final evaluation run summary for the HackerRank Orchestrate "Buy or Wait?" challenge.

## Executive Summary
- **Model Provider:** Google GenAI
- **Model Name:** `gemini-3.5-flash & gemini-3.1-flash-lite`
- **Reasoning Mode:** Extended Thinking (`thinking_budget=1024`)
- **Total Requests Evaluated:** 250
- **Total Model Calls:** 93

## Token Consumption
| Metric | Count |
| :--- | :--- |
| **Total Prompt (Input) Tokens** | 292,544 |
| **Total Candidate (Output + Thinking) Tokens** | 20,221 |
| **Total Tokens** | 417,605 |
| **Average Tokens Per Request** | 1,670.42 |

## Cost Analysis
- **Pricing Tier:** Gemini 3.7 Flash ($0.10/1M input, $0.40/1M output)
- **Total Estimated Cost:** `$0.0373` USD
- **Average Cost Per Request:** `$0.000149` USD

## Configuration & Notes
- Structured Outputs via Pydantic `AffordabilityEvaluation` schema.
- Concurrency managed with `asyncio.Semaphore(8)`.
- Image amount extraction performed with `gemini-3.7-flash` and cached idempotently.
- Deterministic token headroom pruning applied to 90-day cashflow windows before prompt construction.
