# When Tool Calls Fail: Recovery Policies for a Minimal Agent at a Matched Budget

**Team:** [Name 1 (Student ID)], [Name 2 (Student ID)], [Name 3 (Student ID)] · CSC5051 NLP, Fall 2026
**Track and brief:** Track A, brief A4 (A minimal agent with a recovery mechanism)

## 1. Question and motivation

When a tool call fails inside a minimal agent, which recovery policy maximizes success per unit cost at a matched attempt budget: plain retry, typed error feedback, or a bounded repair policy? Tool failures — timeouts, rate limits, invalid arguments — are the norm in real deployments, yet recovery policy is usually an engineering afterthought rather than a measured variable. By the end of this project we will know, with controlled evidence, how much typed error feedback buys over blind retry, and whether treating transient and semantic failures differently keeps the gain at a lower model-call cost.

## 2. Closest prior work and what we do differently

- Shinn et al., *Reflexion* (NeurIPS 2023): verbal error feedback improves agent performance — but mostly on LLM-judged benchmarks, without a cost-normalized comparison of recovery policies.
- Madaan et al., *Self-Refine* (NeurIPS 2023): iterative self-correction with feedback as a general loop.
- Qin et al., *Tool Learning with Foundation Models* (2023): survey identifying failure handling as an open problem in tool-using agents.
- Anthropic, *Building Effective Agents* (2024) and Stanford CS329Z HW1: the "minimal agent, no framework" setup our environment follows.

We differ in three ways: a controlled three-arm comparison of recovery policies under a **matched attempt budget**; **deterministic injected failures** so every arm faces identical failures; and **executable ground-truth checks** instead of LLM judges, reporting success per model call.

## 3. Data / environment / held-out split

A minimal company assistant: one chat-completion call, two tools (`book_meeting`, `send_email`), and our own control loop (no agent framework). Backends are in-memory and reset per run for isolation. 50 tasks in three types: single-tool, two-tool chained, and ambiguous (the agent should ask or abstain). Each task carries a fixed failure schedule (timeout / rate limit / invalid argument), identical across all arms. **20 dev + 30 held-out tasks, frozen in week 1.**

## 4. Baseline

Plain retry (C0): on failure, re-issue the identical tool call up to a fixed retry budget, never re-consulting the model. Expected: 50–70% success on held-out — retries fix transient errors but not semantic ones. (Hypothesis, to be verified.)

## 5. The one change we test

The recovery policy. C1 *typed error feedback*: append `(error_type, message)` to the conversation and let the model re-plan. C2 *bounded repair*: auto-retry with backoff on transient errors (timeout, rate limit), one model re-plan on semantic errors (invalid argument), repair turns capped. All arms share the same attempt budget (e.g., 6 tool executions per task).

## 6. Metrics

Primary: task success rate by executable state checks (never self-judgment), plus abstention rate on ambiguous tasks. Cost metric: model calls (and tokens) per successful task; wall-clock reported separately. ≥ 2 seeds (3 planned).

## 7. Resource budget

No GPU. One hosted chat model via API (e.g., a GLM-5.x or Qwen3.x flash/turbo tier; exact model TBD and logged with version and date). ≈ 3,000 calls planned, estimated ¥50–150 total. Accounts: one API account; a GitHub repository.

## 8. Four-week plan

- **W1 (Nov 13–20):** freeze task set, failure schedule, and held-out split; build backends, failure injection, checker; minimal loop with C0 running end-to-end.
- **W2 (Nov 21–27):** implement C1/C2; smoke-test all three arms; task set to 50.
- **W3 (Nov 28–Dec 4):** matched-budget runs (3 arms × 3 seeds × 2 budgets), ablation by error type, error analysis on failed trajectories.
- **W4 (Dec 5–11):** reproduce the headline result from a clean clone; figures; poster draft.

Deliverables follow the course schedule: poster Dec 17, presentation Dec 18, report Dec 25.

## 9. Fallback experiment

If C2 is not ready or the budget runs short: compare C0 vs C1 only on 20 held-out tasks with 2 seeds — this still answers whether error feedback helps. If all arms trivially succeed, strengthen the injected failures or switch to a weaker model.

## 10. Risks and what we need from the teaching team

- Model robustness may wash out the effect → tune failure strength; a weaker model as an ablation.
- Task and failure design determines everything → we would welcome early feedback on our task/failure schedule in week 1.
- No other dependencies: no GPU, no external data.
