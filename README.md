# The best free AI model aggregator

One Node server that fronts ten free-tier AI providers behind a single pair of
APIs: the OpenAI Chat Completions API (`POST /v1/chat/completions`) and the
Anthropic Messages API (`POST /v1/messages`), each in JSON and in server-sent
events, plus `GET /v1/models` answered in the wire the caller's auth header
names (`Authorization: Bearer` gets the OpenAI list, `x-api-key` the Anthropic
one). It is an entry for make.fun's `free-model-aggregator-01` bounty and runs
on Node 22's standard library alone — nothing to install.

At startup the server reads the trial brief (`$MAKEFUN_PROMPT_FILE`): the
provider directory (each provider's wire, auth scheme, base path and key), the
quality table (every model's rating, context length, and whether it takes
tools, images and JSON), and the workload manifest (the requests to come, in
order, with their requirements and weights). It probes each provider's own
catalog route, builds one catalog document, registers it with the env
(`POST $MAKEFUN_ENV_URL/register`, protocol version 1), serves the workload,
polls `GET $MAKEFUN_ENV_URL/work`, and exits 0 when the env says the workload
is done.

## How it routes

- **Planned, not greedy.** Before the first request arrives, the whole
  workload is assigned to providers by an exact min-cost flow that maximises
  the sum of request weight × model rating, subject to a conservative
  per-provider success budget: the provider's published free-tier minimum
  minus the two scheduled refusals a trial can hide in it. Scarce top-rated
  capacity is therefore reserved for the weight-2 requests and for the
  requests whose fallbacks are weakest.
- **Only free models serve.** A paid model is never a candidate, whatever its
  rating. A candidate must meet the request's requirements (tools, vision,
  JSON mode, minimum context). An OpenAI-wire JSON-mode request is never sent
  to an Anthropic-wire provider, which cannot echo JSON mode back.
- **Failover with memory.** A refusal (402/429) is counted and the request is
  answered by the next-best legal model. A provider whose own error says its
  budget is spent or that it is out is marked dead and never called again;
  three unexplained refusals in a row retire a provider too. After every
  request the plan for the remainder is recomputed from the live state.
  Requests are served one at a time, so a provider never sees a second call
  from this server while its first is in flight.
- **Wire-faithful translation.** Requests are rewritten into the provider's
  wire (OpenAI, Anthropic or Gemini) through one canonical form
  (`lib/wires.ts`), carrying the system prompt, user text, images, tool
  definitions and results, tool choice, `max_tokens`, stop sequences and JSON
  mode exactly as the client sent them — the provider echoes what it parsed,
  and the client compares that echo with what it sent. Answers come back in
  the client's wire with the provider's text whole, its usage counts, and its
  stop reason mapped (`end_turn` ↔ `stop`, `max_tokens` ↔ `length`,
  `tool_use` ↔ `tool_calls`, `refusal` ↔ `content_filter`), streamed or not as
  the client asked.
- **Deterministic.** Every choice derives from the brief, the requests and
  the providers' answers. No wall clock and no randomness reach a request or
  a response; response ids and listing timestamps are fixed constants.

## Run it

In a trial the harness runs it for you:

```
node main.ts
```

with `MAKEFUN_ENV_URL` and `MAKEFUN_PROMPT_FILE` set. To print the catalog
document the server registers (the same bytes, built from the same probes):

```
node main.ts --list
```

Run outside a trial, with no brief to read, `--list` prints `{"providers":[]}`
and exits 0.

## Files

- `main.ts` — the brief, the catalog probe, the planner, the failover
  executor, the HTTP routes, registration and the work poll.
- `lib/wires.ts` — the canonical request/answer form and every translation
  between the two client wires and the three provider wires, JSON and SSE.
- `LICENSE` — MIT.
