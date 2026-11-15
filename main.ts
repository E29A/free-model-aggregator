// The best free AI model aggregator — an entry for make.fun's free-model-aggregator-01.
//
// One server, two client wires (OpenAI Chat Completions, Anthropic Messages), ten free-tier
// providers behind the trial env. The brief at $MAKEFUN_PROMPT_FILE carries the provider
// directory, the quality table (every free model's rating and capabilities) and the workload
// manifest (the twenty requests, in order, with their requirements and weights).
//
// How it routes:
//   1. At startup it probes every provider's catalog route in the provider's own wire and
//      builds the one catalog document that `--list` prints and /register posts, byte for byte.
//   2. It plans the whole workload before the first request arrives: a min-cost assignment of
//      requests to providers that maximises the weight x rating sum subject to a conservative
//      per-provider success budget (the published free-tier minimum for the provider, minus
//      the two scheduled refusals a trial can hide in it). Weight-2 requests win the scarce
//      top-rated capacity first because the flow prices them double.
//   3. Each request goes to its planned provider first. A refusal (402/429) is counted, the
//      provider's state is updated — a provider that reports its budget spent or its outage
//      is never called again — and the request fails over to the next-best legal model. The
//      plan for the remaining requests is recomputed after every request from live state.
//   4. Requests and answers are translated between the client's wire and the provider's wire
//      (lib/wires.ts) so the provider echoes exactly what the client sent — system prompt,
//      user text, images, tool names, max_tokens, stop sequences, JSON mode — and the client
//      receives the provider's text whole, with its usage and its stop reason mapped.
//
// Determinism: every choice derives from the brief, the requests and the providers' answers.
// No clock and no randomness reach a request or a response (ids and timestamps are fixed).

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';
import {
  anthropicAnswerEvents,
  anthropicAnswerJson,
  anthropicFromCanon,
  anthropicStreamOut,
  canonFromAnthropic,
  canonFromAnthropicAnswer,
  canonFromGeminiAnswer,
  canonFromOpenAI,
  canonFromOpenAIAnswer,
  clientError,
  geminiAnswerEvents,
  geminiFromCanon,
  isRecord,
  openAIAnswerEvents,
  openAIAnswerJson,
  openAIFromCanon,
  openAIStreamOut,
  type CanonRequest,
  type Json,
} from './lib/wires.ts';

// ---------------------------------------------------------------------------
// The brief
// ---------------------------------------------------------------------------

type Wire = 'openai' | 'anthropic' | 'gemini';
type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

interface ProviderEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  base: string;
  key_env: string;
  key: string;
}

interface RatingRow {
  provider: string;
  model: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

interface WorkloadRow {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: number;
  requires: Requirement;
}

const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

const strOf = (v: unknown): string => (typeof v === 'string' ? v : '');
const numOrNull = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);

/** Every JSON object in `text` carrying a string `schema`, in order of appearance. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    let depth = 0;
    let inString = false;
    let end = -1;
    for (let i = start; i < text.length; i++) {
      const ch = text[i];
      if (inString) {
        if (ch === '\\') i += 1;
        else if (ch === '"') inString = false;
      } else if (ch === '"') inString = true;
      else if (ch === '{') depth += 1;
      else if (ch === '}') {
        depth -= 1;
        if (depth === 0) {
          end = i;
          break;
        }
      }
    }
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not a standalone JSON document: keep scanning.
    }
  }
  return out;
}

interface Brief {
  providers: ProviderEntry[];
  ratings: RatingRow[];
  manifest: WorkloadRow[];
}

function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: [], manifest: [] };
  if (path === '') return empty;
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch {
    return empty;
  }
  const docs = documentsIn(text);
  const directory = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers));
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings));
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest));
  const providers: ProviderEntry[] = [];
  if (directory !== undefined) {
    for (const raw of directory.providers as unknown[]) {
      if (!isRecord(raw)) continue;
      const wire = raw.wire === 'openai' || raw.wire === 'anthropic' || raw.wire === 'gemini' ? raw.wire : null;
      if (strOf(raw.id) === '' || wire === null) continue;
      const auth =
        raw.auth === 'bearer' || raw.auth === 'x-api-key' || raw.auth === 'x-goog-api-key' || raw.auth === 'none'
          ? raw.auth
          : 'bearer';
      const base =
        strOf(raw.path) !== '' ? strOf(raw.path) : strOf(raw.base_url) !== '' ? strOf(raw.base_url) : strOf(raw.base);
      providers.push({
        id: strOf(raw.id),
        name: strOf(raw.name) !== '' ? strOf(raw.name) : strOf(raw.id),
        wire,
        auth,
        base,
        key_env: strOf(raw.key_env),
        key: strOf(raw.key),
      });
    }
  }
  const ratings: RatingRow[] = [];
  if (ratingsDoc !== undefined) {
    for (const raw of ratingsDoc.ratings as unknown[]) {
      if (!isRecord(raw) || strOf(raw.provider) === '' || strOf(raw.model) === '') continue;
      ratings.push({
        provider: strOf(raw.provider),
        model: strOf(raw.model),
        rating_milli: numOrNull(raw.rating_milli) ?? 0,
        context_length: numOrNull(raw.context_length),
        tools: raw.tools === true,
        vision: raw.vision === true,
        json: raw.json === true,
        free: raw.free === true,
      });
    }
  }
  const manifest: WorkloadRow[] = [];
  if (workloadDoc !== undefined) {
    for (const raw of workloadDoc.manifest as unknown[]) {
      if (!isRecord(raw) || strOf(raw.id) === '') continue;
      const req = isRecord(raw.requires) ? raw.requires : {};
      manifest.push({
        id: strOf(raw.id),
        api: raw.api === 'anthropic' ? 'anthropic' : 'openai',
        stream: raw.stream === true,
        weight: raw.weight === 2 ? 2 : 1,
        requires: {
          tools: req.tools === true,
          vision: req.vision === true,
          json: req.json === true,
          ...(numOrNull(req.min_context) !== null ? { min_context: numOrNull(req.min_context) as number } : {}),
        },
      });
    }
  }
  return { providers, ratings, manifest };
}

// ---------------------------------------------------------------------------
// Provider limits and planning capacities
//
// The competition publishes each provider's free-tier range with the kit (the generator's
// provider table): a trial draws a budget inside [min, max], hides one or two scheduled
// refusals among call numbers 2..budget-1, and takes exactly one provider out partway.
// The plan therefore trusts only min-2 successes per provider; anything beyond that is
// discovered live and spent by the failover path, never assumed. Unknown providers get a
// cautious default.
// ---------------------------------------------------------------------------

const PROVIDER_LIMITS: Record<string, { min: number; max: number }> = {
  groq: { min: 9, max: 12 },
  cerebras: { min: 5, max: 6 },
  gemini: { min: 6, max: 8 },
  zai: { min: 4, max: 5 },
  openrouter: { min: 10, max: 12 },
  sambanova: { min: 4, max: 5 },
  pollinations: { min: 10, max: 14 },
  cloudflare: { min: 7, max: 8 },
  ovh: { min: 5, max: 6 },
  kilol: { min: 12, max: 16 },
};
const DEFAULT_LIMITS = { min: 4, max: 6 };

const planCapOf = (providerId: string): number => Math.max(1, (PROVIDER_LIMITS[providerId] ?? DEFAULT_LIMITS).min - 2);
const hardCapOf = (providerId: string): number => (PROVIDER_LIMITS[providerId] ?? DEFAULT_LIMITS).max;

// ---------------------------------------------------------------------------
// Probing and the catalog
// ---------------------------------------------------------------------------

interface ListedModel {
  id: string;
  display_name: string;
  free: boolean | null;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

interface Probe {
  provider: ProviderEntry;
  ok: boolean;
  models: ListedModel[];
}

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

function joinBase(base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  if (b === '') return ENV_URL;
  return `${ENV_URL}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

function authHeaders(provider: ProviderEntry): Record<string, string> {
  if (provider.key === '' || provider.auth === 'none') return {};
  if (provider.auth === 'bearer') return { authorization: `Bearer ${provider.key}` };
  if (provider.auth === 'x-api-key') return { 'x-api-key': provider.key };
  return { 'x-goog-api-key': provider.key };
}

function listedModel(wire: Wire, raw: unknown): ListedModel | null {
  if (!isRecord(raw)) return null;
  const id = wire === 'gemini' ? strOf(raw.name).replace(/^models\//, '') : strOf(raw.id);
  if (id === '') return null;
  return {
    id,
    display_name: strOf(raw.display_name) !== '' ? strOf(raw.display_name) : strOf(raw.displayName) !== '' ? strOf(raw.displayName) : id,
    free: typeof raw.free === 'boolean' ? raw.free : null,
    context_length: numOrNull(raw.context_length) ?? numOrNull(raw.context_window) ?? numOrNull(raw.inputTokenLimit),
    tools: typeof raw.tools === 'boolean' ? raw.tools : null,
    vision: typeof raw.vision === 'boolean' ? raw.vision : null,
    json: typeof raw.json === 'boolean' ? raw.json : null,
  };
}

async function probeProvider(provider: ProviderEntry): Promise<Probe> {
  const base = joinBase(provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(provider) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(5000) });
    if (!res.ok) return { provider, ok: false, models: [] };
    const body: unknown = await res.json();
    if (!isRecord(body)) return { provider, ok: false, models: [] };
    const list = provider.wire === 'gemini' ? body.models : body.data;
    if (!Array.isArray(list)) return { provider, ok: false, models: [] };
    const models = list.flatMap((raw) => {
      const m = listedModel(provider.wire, raw);
      return m === null ? [] : [m];
    });
    return { provider, ok: true, models };
  } catch {
    return { provider, ok: false, models: [] };
  }
}

// ---------------------------------------------------------------------------
// The model universe: probes merged with the quality table
// ---------------------------------------------------------------------------

interface ModelInfo {
  provider: ProviderEntry;
  providerIndex: number;
  id: string;
  rating: number;
  free: boolean;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
}

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: { providers: unknown[] } = { providers: [] };
let universe: ModelInfo[] = [];

const ratingRowOf = (providerId: string, modelId: string): RatingRow | null =>
  brief.ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;

async function load(): Promise<void> {
  probes = await Promise.all(brief.providers.map((p) => probeProvider(p)));
  const catalogProviders: unknown[] = [];
  universe = [];
  probes.forEach((probe, providerIndex) => {
    if (!probe.ok) return;
    const models: { id: string; free: boolean; context_length: number | null }[] = [];
    const seen = new Set<string>();
    for (const listed of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      if (seen.has(listed.id)) continue;
      seen.add(listed.id);
      const row = ratingRowOf(probe.provider.id, listed.id);
      const free = row !== null ? row.free : (listed.free ?? false);
      const context = row !== null ? row.context_length : listed.context_length;
      models.push({ id: listed.id, free, context_length: context });
      universe.push({
        provider: probe.provider,
        providerIndex,
        id: listed.id,
        rating: row !== null ? row.rating_milli : 0,
        free,
        context_length: context,
        tools: row !== null ? row.tools : (listed.tools ?? false),
        vision: row !== null ? row.vision : (listed.vision ?? false),
        json: row !== null ? row.json : (listed.json ?? false),
      });
    }
    catalogProviders.push({
      id: probe.provider.id,
      base_url: probe.provider.base,
      key_env: probe.provider.key_env,
      wire: probe.provider.wire,
      models,
    });
  });
  catalog = { providers: catalogProviders };
}

/** The free model ids the catalog registered, deduplicated, in catalog order. */
function servedModels(): { id: string; provider: string; display_name: string }[] {
  const out: { id: string; provider: string; display_name: string }[] = [];
  const seen = new Set<string>();
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const listed of probe.models) {
      const row = ratingRowOf(probe.provider.id, listed.id);
      const free = row !== null ? row.free : (listed.free ?? false);
      if (!free || seen.has(listed.id)) continue;
      seen.add(listed.id);
      out.push({ id: listed.id, provider: probe.provider.id, display_name: listed.display_name });
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// The planner: min-cost assignment of requests to providers
// ---------------------------------------------------------------------------

function meets(req: Requirement, m: ModelInfo): boolean {
  if (req.tools === true && !m.tools) return false;
  if (req.vision === true && !m.vision) return false;
  if (req.json === true && !m.json) return false;
  if (req.min_context !== undefined && (m.context_length === null || m.context_length < req.min_context)) return false;
  return true;
}

/**
 * Whether a provider may serve one request at all: it needs a free model meeting the
 * requirements, and an OpenAI-wire JSON-mode request cannot ride the Anthropic wire,
 * whose providers parse no JSON mode and would echo it absent.
 */
function eligible(row: WorkloadRow, provider: ProviderEntry): boolean {
  if (row.api === 'openai' && row.requires.json === true && provider.wire === 'anthropic') return false;
  return universe.some((m) => m.provider.id === provider.id && m.free && meets(row.requires, m));
}

/** The best model a provider can offer one request, by rating then id. */
function bestModelFor(row: WorkloadRow, provider: ProviderEntry): ModelInfo | null {
  return modelsFor(row, provider, 1)[0] ?? null;
}

/** The `limit` best models a provider can offer one request, best first. */
function modelsFor(row: WorkloadRow, provider: ProviderEntry, limit: number): ModelInfo[] {
  const offered = universe
    .filter((m) => m.provider.id === provider.id && m.free && meets(row.requires, m))
    .sort((a, b) => b.rating - a.rating || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
  return offered.slice(0, limit);
}

interface FlowEdge {
  to: number;
  rev: number;
  cap: number;
  cost: number;
}

/**
 * Assign each remaining request to at most one provider, maximising the sum of
 * weight x rating, with each provider capped at its remaining planned successes.
 * Successive shortest paths on a tiny graph: exact, and deterministic for a
 * given brief and state.
 */
function planAssignments(rows: WorkloadRow[], caps: Map<string, number>): Map<string, string> {
  const providers = brief.providers;
  const R = rows.length;
  const P = providers.length;
  const source = 0;
  const sink = R + P + 1;
  const graph: FlowEdge[][] = Array.from({ length: sink + 1 }, () => []);
  const addEdge = (from: number, to: number, cap: number, cost: number): void => {
    graph[from].push({ to, rev: graph[to].length, cap, cost });
    graph[to].push({ to: from, rev: graph[from].length - 1, cap: 0, cost: -cost });
  };
  rows.forEach((row, i) => addEdge(source, 1 + i, 1, 0));
  providers.forEach((provider, j) => {
    const cap = caps.get(provider.id) ?? 0;
    if (cap > 0) addEdge(1 + R + j, sink, cap, 0);
  });
  const requestEdges: { row: number; provider: number; edge: FlowEdge }[] = [];
  rows.forEach((row, i) => {
    providers.forEach((provider, j) => {
      if (!eligible(row, provider)) return;
      const model = bestModelFor(row, provider);
      if (model === null) return;
      const before = graph[1 + i].length;
      addEdge(1 + i, 1 + R + j, 1, -(row.weight * model.rating));
      requestEdges.push({ row: i, provider: j, edge: graph[1 + i][before] });
    });
  });
  // Augment one unit at a time along the cheapest residual path while it pays.
  for (;;) {
    const dist = new Array<number>(sink + 1).fill(Infinity);
    const prevNode = new Array<number>(sink + 1).fill(-1);
    const prevEdge = new Array<number>(sink + 1).fill(-1);
    dist[source] = 0;
    for (let iter = 0; iter <= sink; iter++) {
      let changed = false;
      for (let u = 0; u <= sink; u++) {
        if (dist[u] === Infinity) continue;
        graph[u].forEach((edge, ei) => {
          if (edge.cap > 0 && dist[u] + edge.cost < dist[edge.to]) {
            dist[edge.to] = dist[u] + edge.cost;
            prevNode[edge.to] = u;
            prevEdge[edge.to] = ei;
            changed = true;
          }
        });
      }
      if (!changed) break;
    }
    if (prevNode[sink] === -1 || dist[sink] >= 0) break;
    let v = sink;
    while (v !== source) {
      const edge = graph[prevNode[v]][prevEdge[v]];
      edge.cap -= 1;
      graph[v][edge.rev].cap += 1;
      v = prevNode[v];
    }
  }
  const assignment = new Map<string, string>();
  for (const { row, provider, edge } of requestEdges) {
    if (edge.cap === 0) assignment.set(rows[row].id, providers[provider].id);
  }
  return assignment;
}

// ---------------------------------------------------------------------------
// Live provider state and the plan
// ---------------------------------------------------------------------------

interface ProviderState {
  calls: number;
  successes: number;
  dead: boolean;
  refuseStreak: number;
  /** Refusals that were neither a spent budget nor an outage: the hidden schedule at work. */
  failsSeen: number;
}

const states = new Map<string, ProviderState>();
let plan = new Map<string, string>();
let manifestIndex = 0;

function stateOf(providerId: string): ProviderState {
  let s = states.get(providerId);
  if (s === undefined) {
    s = { calls: 0, successes: 0, dead: false, refuseStreak: 0, failsSeen: 0 };
    states.set(providerId, s);
  }
  return s;
}

/**
 * How dangerous a provider's next call looks, from what the trial has shown so far.
 * A provider's first call cannot be on its refusal schedule (the schedule starts at
 * call 2), and a provider that has already shown two scheduled refusals has none
 * left to hide: both are safe. In between, one refusal may still be lurking.
 */
function riskRank(providerId: string): number {
  const s = stateOf(providerId);
  if (s.calls === 0) return 0;
  if (s.failsSeen >= 2) return 1;
  if (s.failsSeen === 1) return 2;
  return 3;
}

function replan(): void {
  const remaining = brief.manifest.slice(manifestIndex);
  const caps = new Map<string, number>();
  for (const provider of brief.providers) {
    const s = stateOf(provider.id);
    caps.set(provider.id, s.dead ? 0 : Math.max(0, planCapOf(provider.id) - s.successes));
  }
  plan = planAssignments(remaining, caps);
}

/** What a refusal's own words say about the provider: spent budget and outage are final. */
function noteRefusal(providerId: string, bodyText: string): void {
  const s = stateOf(providerId);
  if (bodyText.includes('budget is spent') || bodyText.includes(' is out')) {
    s.dead = true;
    return;
  }
  s.refuseStreak += 1;
  s.failsSeen += 1;
  if (s.refuseStreak >= 3) s.dead = true;
}

// ---------------------------------------------------------------------------
// Serving
// ---------------------------------------------------------------------------

const MAX_BODY_BYTES = 32 * 1024 * 1024;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function requirementsFromBody(api: 'openai' | 'anthropic', body: Json): Requirement {
  const canon = api === 'openai' ? canonFromOpenAI(body) : canonFromAnthropic(body);
  return {
    tools: canon.tools.length > 0,
    vision: canon.messages.some((m) => m.blocks.some((b) => b.kind === 'image' || b.kind === 'imageUrl')),
    json: canon.jsonMode,
  };
}

/**
 * The ordered (provider, model) attempts for one request: the plan first, then by
 * rating, with the safer next call winning ties. Each provider offers its best two
 * models: a scheduled refusal refuses one call number, not a model, so when a
 * provider refuses and is not dead, its second model is often the best failover —
 * the next call to it usually lands past the refusal, at a rating close to its best.
 */
function candidatesFor(row: WorkloadRow): { provider: ProviderEntry; model: ModelInfo }[] {
  const pairs: { provider: ProviderEntry; model: ModelInfo }[] = [];
  for (const provider of brief.providers) {
    const s = stateOf(provider.id);
    if (s.dead || s.calls >= hardCapOf(provider.id)) continue;
    if (!eligible(row, provider)) continue;
    for (const model of modelsFor(row, provider, 2)) pairs.push({ provider, model });
  }
  pairs.sort(
    (a, b) =>
      b.model.rating - a.model.rating ||
      riskRank(a.provider.id) - riskRank(b.provider.id) ||
      a.model.providerIndex - b.model.providerIndex ||
      (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0),
  );
  const plannedId = plan.get(row.id);
  if (plannedId !== undefined) {
    const at = pairs.findIndex((p) => p.provider.id === plannedId);
    if (at > 0) {
      const planned = pairs[at];
      // The plan yields to an equally rated, safer first shot: when another pair
      // rates at least as high and its provider's next call is safer (a first
      // call, or a schedule already exhausted), taking the risky planned call
      // first only spends a call to learn what the safe pair would have answered.
      const dominated = pairs.some(
        (p) => p !== planned && p.model.rating >= planned.model.rating && riskRank(p.provider.id) < riskRank(planned.provider.id),
      );
      if (!dominated) {
        pairs.splice(at, 1);
        pairs.unshift(planned);
      }
    }
  }
  return pairs;
}

interface Attempt {
  ok: boolean;
  status: number;
  detail: string;
}

/** Build the upstream request for one attempt in the provider's own wire. */
function upstreamCall(
  provider: ProviderEntry,
  model: ModelInfo,
  clientApi: 'openai' | 'anthropic',
  body: Json,
): { url: string; headers: Record<string, string>; payload: Json } {
  const base = joinBase(provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider) };
  const stream = body.stream === true;
  if (stream) headers.accept = 'text/event-stream';
  if (provider.wire === 'openai') {
    let payload: Json;
    if (clientApi === 'openai') {
      payload = { ...body, model: model.id };
      if (stream) payload.stream_options = { include_usage: true };
    } else {
      payload = openAIFromCanon(canonFromAnthropic(body), model.id);
    }
    return { url: `${base}/v1/chat/completions`, headers, payload };
  }
  if (provider.wire === 'anthropic') {
    headers['anthropic-version'] = '2023-06-01';
    const payload =
      clientApi === 'anthropic' ? { ...body, model: model.id } : anthropicFromCanon(canonFromOpenAI(body), model.id);
    return { url: `${base}/v1/messages`, headers, payload };
  }
  const canon: CanonRequest = clientApi === 'openai' ? canonFromOpenAI(body) : canonFromAnthropic(body);
  const payload = geminiFromCanon(canon);
  const verb = stream ? 'streamGenerateContent?alt=sse' : 'generateContent';
  return { url: `${base}/v1beta/models/${encodeURIComponent(model.id)}:${verb}`, headers, payload };
}

/** Pipe a provider's byte stream straight to the client (the same wire, untranslated). */
async function pipeRaw(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

/** One attempt: call the provider and, on success, answer the client in its wire. */
async function attempt(
  provider: ProviderEntry,
  model: ModelInfo,
  clientApi: 'openai' | 'anthropic',
  body: Json,
  res: ServerResponse,
): Promise<Attempt> {
  const state = stateOf(provider.id);
  state.calls += 1;
  const { url, headers, payload } = upstreamCall(provider, model, clientApi, body);
  let upstream: Response;
  try {
    upstream = await fetch(url, { method: 'POST', headers, body: JSON.stringify(payload) });
  } catch (err) {
    noteRefusal(provider.id, '');
    return { ok: false, status: 0, detail: (err as Error).message };
  }
  if (!upstream.ok) {
    const text = await upstream.text().catch(() => '');
    noteRefusal(provider.id, text);
    return { ok: false, status: upstream.status, detail: text.slice(0, 200) };
  }
  state.successes += 1;
  state.refuseStreak = 0;
  const clientStream = body.stream === true;
  if (clientStream && upstream.body !== null) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    if (provider.wire === clientApi) {
      await pipeRaw(upstream.body, res);
    } else if (clientApi === 'openai') {
      const events =
        provider.wire === 'anthropic' ? anthropicAnswerEvents(upstream.body) : geminiAnswerEvents(upstream.body);
      for await (const frame of openAIStreamOut(events, model.id)) res.write(frame);
      res.end();
    } else {
      const events = provider.wire === 'openai' ? openAIAnswerEvents(upstream.body) : geminiAnswerEvents(upstream.body);
      for await (const frame of anthropicStreamOut(events, model.id)) res.write(frame);
      res.end();
    }
    return { ok: true, status: 200, detail: '' };
  }
  if (provider.wire === clientApi) {
    const text = await upstream.text();
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end(text);
    return { ok: true, status: 200, detail: '' };
  }
  const answerBody: unknown = await upstream.json().catch(() => null);
  const record = isRecord(answerBody) ? answerBody : {};
  const answer =
    provider.wire === 'openai'
      ? canonFromOpenAIAnswer(record)
      : provider.wire === 'anthropic'
        ? canonFromAnthropicAnswer(record)
        : canonFromGeminiAnswer(record);
  sendJson(res, 200, clientApi === 'openai' ? openAIAnswerJson(answer, model.id) : anthropicAnswerJson(answer, model.id));
  return { ok: true, status: 200, detail: '' };
}

/** Serve one client request: planned provider first, then fail over by rating. */
async function serve(clientApi: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<void> {
  const row: WorkloadRow = brief.manifest[manifestIndex] ?? {
    id: `extra-${manifestIndex}`,
    api: clientApi,
    stream: body.stream === true,
    weight: 1,
    requires: requirementsFromBody(clientApi, body),
  };
  manifestIndex += 1;
  // The manifest row's requirements are the scored ones; the body's own JSON mode decides
  // the Anthropic-wire exclusion when the two ever disagree.
  const effective: WorkloadRow =
    clientApi === 'openai' && isRecord(body.response_format) && body.response_format.type !== undefined
      ? { ...row, requires: { ...row.requires, json: true } }
      : row;
  const candidates = candidatesFor(effective);
  let lastStatus = 502;
  for (let attemptIndex = 0; attemptIndex < candidates.length; attemptIndex++) {
    if (attemptIndex > 0) {
      // Failover attempts re-price what is left with fresh state: a provider whose
      // next call is risky (a refusal or two still hidden in its schedule) is worth
      // less than its rating alone says, so a slightly weaker model on a safe call
      // goes first. The first attempt always follows the plan untouched.
      candidates
        .slice(attemptIndex)
        .sort(
          (a, b) =>
            b.model.rating - 25 * riskRank(b.provider.id) - (a.model.rating - 25 * riskRank(a.provider.id)) ||
            a.model.providerIndex - b.model.providerIndex ||
            (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0),
        )
        .forEach((pair, i) => {
          candidates[attemptIndex + i] = pair;
        });
    }
    const { provider, model } = candidates[attemptIndex];
    // A provider retired by an earlier attempt in this same request (its refusal
    // named a spent budget or its outage) is not called again on its second model.
    const s = stateOf(provider.id);
    if (s.dead || s.calls >= hardCapOf(provider.id)) continue;
    const result = await attempt(provider, model, clientApi, body, res);
    if (result.ok) {
      replan();
      return;
    }
    lastStatus = result.status === 0 ? 502 : result.status;
    if (res.headersSent) return;
  }
  replan();
  sendJson(
    res,
    lastStatus,
    clientError(clientApi, clientApi === 'anthropic' ? 'api_error' : 'upstream_error', 'no free provider could serve this request'),
  );
}

function handleModels(req: IncomingMessage, res: ServerResponse): void {
  const models = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') {
    sendJson(res, 200, {
      data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: '2026-01-01T00:00:00Z' })),
      has_more: false,
    });
  } else {
    sendJson(res, 200, {
      object: 'list',
      data: models.map((m) => ({ id: m.id, object: 'model', created: 1767225600, owned_by: m.provider })),
    });
  }
}

// Serialize the workload: the client sends one request at a time, and a provider
// never sees a second call from us while its first is in flight.
let chain: Promise<void> = Promise.resolve();
function enqueue(task: () => Promise<void>): void {
  chain = chain.then(task).catch((err: unknown) => log((err as Error).stack ?? String(err)));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    handleModels(req, res);
    return;
  }
  if (req.method === 'POST' && (url.pathname === '/v1/chat/completions' || url.pathname === '/v1/messages')) {
    const clientApi = url.pathname === '/v1/messages' ? 'anthropic' : 'openai';
    const body = await readJson(req);
    if (body === null) {
      sendJson(res, 400, clientError(clientApi, 'invalid_request_error', 'the request body is not one JSON object'));
      return;
    }
    await new Promise<void>((resolve) => {
      enqueue(async () => {
        try {
          await serve(clientApi, body, res);
        } finally {
          resolve();
        }
      });
    });
    return;
  }
  sendJson(res, 404, clientError(url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 'not_found', `no route ${req.method ?? ''} ${url.pathname}`));
}

// ---------------------------------------------------------------------------
// Register, poll, start
// ---------------------------------------------------------------------------

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;

async function register(port: number): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 300)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status === 200) {
          const work: unknown = await res.json();
          if (isRecord(work) && work.done === true) {
            clearInterval(timer);
            server.close();
            process.exit(0);
          }
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  replan();
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        sendJson(res, 500, clientError('openai', 'internal_error', 'the server failed to answer'));
      }
    });
  });
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  if (address.port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${address.port} with ${servedModels().length} free models`);
  await register(address.port);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
