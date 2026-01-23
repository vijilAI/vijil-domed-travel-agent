# Vijil Domed Travel Agent

Enterprise travel assistant protected by **Vijil Dome** guardrails.

This is the **DOMED** version of the travel agent, demonstrating the Diamond-Dome-Darwin loop:

- **Diamond**: Evaluates agent behavior and computes trust scores
- **Dome**: Runtime guardrails protecting against adversarial attacks
- **Darwin**: Evolves the agent genome based on detected issues

For the unprotected baseline, see [vijil-travel-agent](https://github.com/vijilAI/vijil-travel-agent).

## Features

### Agent Capabilities
- Search and book flights
- Manage traveler profiles and documents
- Process payments and refunds
- Handle loyalty point redemptions
- Auto-rebook during disruptions
- Check policy compliance
- Submit travel expenses

### Dome Protection
- **Prompt Injection Detection**: Encoding heuristics + mBERT classifier
- **Toxicity Filtering**: Flashtext banlist + DeBERTa classifier
- **PII Masking**: Presidio entity recognition

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set your API key
export GROQ_API_KEY="your-key"

# Run the domed agent
python agent.py
```

## Testing Dome Guardrails

```bash
# Run adversarial tests
python test_adversarial.py
```

## Detection Logging

All Dome detections are logged to `dome_detections.jsonl` for Darwin's evolution feedback loop:

```json
{
  "timestamp": "2026-01-23T12:00:00Z",
  "agent_id": "vijil-domed-travel-agent",
  "direction": "input",
  "blocked": true,
  "flagged": true,
  "text_preview": "Ignore all previous instructions...",
  "trace": {...},
  "exec_time": 0.45
}
```

## Architecture

```
User Request
    │
    ▼
┌─────────────────┐
│ Dome Middleware │ ── Blocked → Return safety message
└────────┬────────┘
         │ Safe
         ▼
┌─────────────────┐
│  Travel Agent   │
│  (Strands A2A)  │
└────────┬────────┘
         │
         ▼
    Agent Response
```

## Related Projects

- [vijil-travel-agent](https://github.com/vijilAI/vijil-travel-agent) - Unprotected baseline
- [vijil-dome](https://github.com/vijilAI/vijil-dome) - Guardrail library
- [vijil-diamond](https://github.com/vijilAI/vijil-diamond) - Evaluation framework
