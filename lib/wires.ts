// wires.ts — translation between the two client wires (OpenAI Chat Completions, Anthropic
// Messages) and the three provider wires (OpenAI, Anthropic, Gemini).
//
// Every conversion goes through one canonical form (CanonRequest / CanonAnswer / AnswerEvent),
// so each wire is written once in each direction and the six pairings cannot drift apart.
// The functions are pure: the same input gives the same bytes, which the trial's replay
// depends on. Nothing here reads a clock or a random source; response ids and timestamps
// are fixed strings.

export type Json = Record<string, unknown>;

export const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

const str = (v: unknown): string => (typeof v === 'string' ? v : '');
const numOrNull = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const numOrZero = (v: unknown): number => (typeof v === 'number' && Number.isFinite(v) ? v : 0);

// ---------------------------------------------------------------------------
// Canonical request
// ---------------------------------------------------------------------------

export type Block =
  | { kind: 'text'; text: string }
  | { kind: 'image'; mediaType: string; data: string }
  | { kind: 'imageUrl'; url: string }
  | { kind: 'toolUse'; id: string; name: string; input: Json }
  | { kind: 'toolResult'; toolUseId: string; name: string; content: string };

export interface CanonMessage {
  role: 'user' | 'assistant';
  blocks: Block[];
}

export interface CanonTool {
  name: string;
  description: string;
  parameters: Json;
}

export type CanonToolChoice =
  | { mode: 'auto' }
  | { mode: 'any' }
  | { mode: 'none' }
  | { mode: 'tool'; name: string }
  | null;

export interface CanonRequest {
  systemText: string;
  messages: CanonMessage[];
  tools: CanonTool[];
  toolChoice: CanonToolChoice;
  maxTokens: number | null;
  temperature: number | null;
  topP: number | null;
  stops: string[];
  jsonMode: boolean;
  stream: boolean;
}

/** The text blocks of a string-or-blocks content value, joined as the wires join them. */
function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

/** A base64 image carried as a data: URL becomes an image block; any other URL stays a URL. */
function imageBlockFromUrl(url: string): Block {
  const m = /^data:([^;,]+);base64,(.*)$/s.exec(url);
  if (m !== null) return { kind: 'image', mediaType: m[1], data: m[2] };
  return { kind: 'imageUrl', url };
}

function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = blocksText(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** An OpenAI Chat Completions request body in canonical form. */
export function canonFromOpenAI(body: Json): CanonRequest {
  const systemParts: string[] = [];
  const messages: CanonMessage[] = [];
  const toolNames = new Map<string, string>();
  const raw = Array.isArray(body.messages) ? body.messages : [];
  for (const item of raw) {
    if (!isRecord(item)) continue;
    if (item.role === 'system') {
      const text = blocksText(item.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (item.role === 'tool') {
      messages.push({
        role: 'user',
        blocks: [
          {
            kind: 'toolResult',
            toolUseId: str(item.tool_call_id),
            name: toolNames.get(str(item.tool_call_id)) ?? '',
            content: toolResultText(item.content),
          },
        ],
      });
      continue;
    }
    const role = item.role === 'assistant' ? 'assistant' : 'user';
    const blocks: Block[] = [];
    if (typeof item.content === 'string') {
      if (item.content !== '') blocks.push({ kind: 'text', text: item.content });
    } else if (Array.isArray(item.content)) {
      for (const part of item.content) {
        if (!isRecord(part)) continue;
        if (part.type === 'text' && typeof part.text === 'string') blocks.push({ kind: 'text', text: part.text });
        else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
          blocks.push(imageBlockFromUrl(part.image_url.url));
        }
      }
    }
    if (role === 'assistant' && Array.isArray(item.tool_calls)) {
      for (const call of item.tool_calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        const id = str(call.id);
        const name = str(call.function.name);
        if (id !== '') toolNames.set(id, name);
        let input: Json = {};
        if (typeof call.function.arguments === 'string' && call.function.arguments.trim() !== '') {
          try {
            const parsed: unknown = JSON.parse(call.function.arguments);
            if (isRecord(parsed)) input = parsed;
          } catch {
            input = {};
          }
        }
        blocks.push({ kind: 'toolUse', id, name, input });
      }
    }
    messages.push({ role, blocks });
  }
  const tools: CanonTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const t of body.tools) {
      if (!isRecord(t) || !isRecord(t.function) || typeof t.function.name !== 'string') continue;
      tools.push({
        name: t.function.name,
        description: str(t.function.description),
        parameters: isRecord(t.function.parameters) ? t.function.parameters : { type: 'object', properties: {} },
      });
    }
  }
  let toolChoice: CanonToolChoice = null;
  if (body.tool_choice === 'auto') toolChoice = { mode: 'auto' };
  else if (body.tool_choice === 'none') toolChoice = { mode: 'none' };
  else if (body.tool_choice === 'required') toolChoice = { mode: 'any' };
  else if (isRecord(body.tool_choice) && isRecord(body.tool_choice.function) && typeof body.tool_choice.function.name === 'string') {
    toolChoice = { mode: 'tool', name: body.tool_choice.function.name };
  }
  const stops: string[] =
    typeof body.stop === 'string' ? [body.stop] : Array.isArray(body.stop) ? body.stop.filter((s): s is string => typeof s === 'string') : [];
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    systemText: systemParts.join('\n'),
    messages,
    tools,
    toolChoice,
    maxTokens: numOrNull(body.max_tokens) ?? numOrNull(body.max_completion_tokens),
    temperature: numOrNull(body.temperature),
    topP: numOrNull(body.top_p),
    stops,
    jsonMode: format === 'json_object' || format === 'json_schema',
    stream: body.stream === true,
  };
}

/** An Anthropic Messages request body in canonical form. */
export function canonFromAnthropic(body: Json): CanonRequest {
  const messages: CanonMessage[] = [];
  const toolNames = new Map<string, string>();
  const raw = Array.isArray(body.messages) ? body.messages : [];
  for (const item of raw) {
    if (!isRecord(item)) continue;
    const role = item.role === 'assistant' ? 'assistant' : 'user';
    const blocks: Block[] = [];
    if (typeof item.content === 'string') {
      if (item.content !== '') blocks.push({ kind: 'text', text: item.content });
    } else if (Array.isArray(item.content)) {
      for (const block of item.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') blocks.push({ kind: 'text', text: block.text });
        else if (block.type === 'image' && isRecord(block.source)) {
          if (block.source.type === 'base64' && typeof block.source.data === 'string') {
            blocks.push({ kind: 'image', mediaType: str(block.source.media_type) || 'image/png', data: block.source.data });
          } else if (block.source.type === 'url' && typeof block.source.url === 'string') {
            blocks.push({ kind: 'imageUrl', url: block.source.url });
          }
        } else if (block.type === 'tool_use') {
          const id = str(block.id);
          const name = str(block.name);
          if (id !== '') toolNames.set(id, name);
          blocks.push({ kind: 'toolUse', id, name, input: isRecord(block.input) ? block.input : {} });
        } else if (block.type === 'tool_result') {
          const id = str(block.tool_use_id);
          blocks.push({ kind: 'toolResult', toolUseId: id, name: toolNames.get(id) ?? '', content: toolResultText(block.content) });
        }
      }
    }
    messages.push({ role, blocks });
  }
  const tools: CanonTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const t of body.tools) {
      if (!isRecord(t) || typeof t.name !== 'string') continue;
      tools.push({
        name: t.name,
        description: str(t.description),
        parameters: isRecord(t.input_schema) ? t.input_schema : { type: 'object', properties: {} },
      });
    }
  }
  let toolChoice: CanonToolChoice = null;
  if (isRecord(body.tool_choice)) {
    if (body.tool_choice.type === 'auto') toolChoice = { mode: 'auto' };
    else if (body.tool_choice.type === 'any') toolChoice = { mode: 'any' };
    else if (body.tool_choice.type === 'none') toolChoice = { mode: 'none' };
    else if (body.tool_choice.type === 'tool' && typeof body.tool_choice.name === 'string') toolChoice = { mode: 'tool', name: body.tool_choice.name };
  }
  return {
    systemText: blocksText(body.system),
    messages,
    tools,
    toolChoice,
    maxTokens: numOrNull(body.max_tokens),
    temperature: numOrNull(body.temperature),
    topP: numOrNull(body.top_p),
    stops: Array.isArray(body.stop_sequences) ? body.stop_sequences.filter((s): s is string => typeof s === 'string') : [],
    jsonMode: false,
    stream: body.stream === true,
  };
}

// ---------------------------------------------------------------------------
// Canonical request -> provider wire
// ---------------------------------------------------------------------------

function openAIParts(blocks: Block[]): unknown {
  const parts: Json[] = [];
  for (const b of blocks) {
    if (b.kind === 'text') parts.push({ type: 'text', text: b.text });
    else if (b.kind === 'image') parts.push({ type: 'image_url', image_url: { url: `data:${b.mediaType};base64,${b.data}` } });
    else if (b.kind === 'imageUrl') parts.push({ type: 'image_url', image_url: { url: b.url } });
  }
  if (parts.length === 1 && parts[0].type === 'text') return parts[0].text;
  return parts;
}

/** The canonical request as an OpenAI Chat Completions body naming `model`. */
export function openAIFromCanon(req: CanonRequest, model: string): Json {
  const messages: Json[] = [];
  if (req.systemText !== '') messages.push({ role: 'system', content: req.systemText });
  for (const msg of req.messages) {
    if (msg.role === 'assistant') {
      const text = msg.blocks.flatMap((b) => (b.kind === 'text' ? [b.text] : [])).join('\n');
      const calls = msg.blocks.flatMap((b) =>
        b.kind === 'toolUse'
          ? [{ id: b.id, type: 'function', function: { name: b.name, arguments: JSON.stringify(b.input) } }]
          : [],
      );
      messages.push({ role: 'assistant', content: text !== '' ? text : null, ...(calls.length > 0 ? { tool_calls: calls } : {}) });
      continue;
    }
    for (const b of msg.blocks) {
      if (b.kind === 'toolResult') messages.push({ role: 'tool', tool_call_id: b.toolUseId, content: b.content });
    }
    const rest = msg.blocks.filter((b) => b.kind !== 'toolResult');
    if (rest.length > 0) messages.push({ role: 'user', content: openAIParts(rest) });
  }
  const out: Json = { model, messages };
  if (req.tools.length > 0) {
    out.tools = req.tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.parameters } }));
  }
  if (req.toolChoice !== null) {
    if (req.toolChoice.mode === 'auto') out.tool_choice = 'auto';
    else if (req.toolChoice.mode === 'none') out.tool_choice = 'none';
    else if (req.toolChoice.mode === 'any') out.tool_choice = 'required';
    else out.tool_choice = { type: 'function', function: { name: req.toolChoice.name } };
  }
  if (req.maxTokens !== null) out.max_tokens = req.maxTokens;
  if (req.temperature !== null) out.temperature = req.temperature;
  if (req.topP !== null) out.top_p = req.topP;
  if (req.stops.length > 0) out.stop = req.stops;
  if (req.jsonMode) out.response_format = { type: 'json_object' };
  if (req.stream) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** The canonical request as an Anthropic Messages body naming `model`. */
export function anthropicFromCanon(req: CanonRequest, model: string): Json {
  const messages: Json[] = [];
  for (const msg of req.messages) {
    const blocks: Json[] = [];
    for (const b of msg.blocks) {
      if (b.kind === 'text') blocks.push({ type: 'text', text: b.text });
      else if (b.kind === 'image') blocks.push({ type: 'image', source: { type: 'base64', media_type: b.mediaType, data: b.data } });
      else if (b.kind === 'imageUrl') blocks.push({ type: 'image', source: { type: 'url', url: b.url } });
      else if (b.kind === 'toolUse') blocks.push({ type: 'tool_use', id: b.id, name: b.name, input: b.input });
      else if (b.kind === 'toolResult') blocks.push({ type: 'tool_result', tool_use_id: b.toolUseId, content: b.content });
    }
    if (typeof msg.blocks === 'object' && blocks.length === 0) blocks.push({ type: 'text', text: '' });
    messages.push({ role: msg.role, content: blocks });
  }
  const out: Json = { model, messages };
  if (req.systemText !== '') out.system = req.systemText;
  if (req.tools.length > 0) {
    out.tools = req.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
  }
  if (req.toolChoice !== null) {
    if (req.toolChoice.mode === 'auto') out.tool_choice = { type: 'auto' };
    else if (req.toolChoice.mode === 'none') out.tool_choice = { type: 'none' };
    else if (req.toolChoice.mode === 'any') out.tool_choice = { type: 'any' };
    else out.tool_choice = { type: 'tool', name: req.toolChoice.name };
  }
  if (req.maxTokens !== null) out.max_tokens = req.maxTokens;
  if (req.temperature !== null) out.temperature = req.temperature;
  if (req.topP !== null) out.top_p = req.topP;
  if (req.stops.length > 0) out.stop_sequences = req.stops;
  if (req.stream) out.stream = true;
  return out;
}

/** The canonical request as a Gemini generateContent body (the model goes in the URL). */
export function geminiFromCanon(req: CanonRequest): Json {
  const contents: Json[] = [];
  for (const msg of req.messages) {
    const parts: Json[] = [];
    for (const b of msg.blocks) {
      if (b.kind === 'text') parts.push({ text: b.text });
      else if (b.kind === 'image') parts.push({ inlineData: { mimeType: b.mediaType, data: b.data } });
      else if (b.kind === 'imageUrl') parts.push({ fileData: { fileUri: b.url } });
      else if (b.kind === 'toolUse') parts.push({ functionCall: { name: b.name, args: b.input } });
      else if (b.kind === 'toolResult') parts.push({ functionResponse: { name: b.name !== '' ? b.name : 'tool', response: { result: b.content } } });
    }
    contents.push({ role: msg.role === 'assistant' ? 'model' : 'user', parts });
  }
  const out: Json = { contents };
  if (req.systemText !== '') out.systemInstruction = { parts: [{ text: req.systemText }] };
  if (req.tools.length > 0) {
    out.tools = [
      { functionDeclarations: req.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })) },
    ];
  }
  if (req.toolChoice !== null) {
    const mode =
      req.toolChoice.mode === 'auto' ? 'AUTO' : req.toolChoice.mode === 'any' ? 'ANY' : req.toolChoice.mode === 'none' ? 'NONE' : 'ANY';
    out.toolConfig = {
      functionCallingConfig: {
        mode,
        ...(req.toolChoice.mode === 'tool' ? { allowedFunctionNames: [req.toolChoice.name] } : {}),
      },
    };
  }
  const config: Json = {};
  if (req.maxTokens !== null) config.maxOutputTokens = req.maxTokens;
  if (req.temperature !== null) config.temperature = req.temperature;
  if (req.topP !== null) config.topP = req.topP;
  if (req.stops.length > 0) config.stopSequences = req.stops;
  if (req.jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

// ---------------------------------------------------------------------------
// Canonical answer
// ---------------------------------------------------------------------------

export type StopKind = 'end' | 'length' | 'tools' | 'filter';

export interface CanonToolCall {
  id: string;
  name: string;
  input: Json;
}

export interface CanonAnswer {
  text: string;
  toolCalls: CanonToolCall[];
  stop: StopKind;
  inputTokens: number;
  outputTokens: number;
}

function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

export function stopFromOpenAI(finish: unknown): StopKind {
  if (finish === 'length') return 'length';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tools';
  if (finish === 'content_filter') return 'filter';
  return 'end';
}

export function stopFromAnthropic(reason: unknown): StopKind {
  if (reason === 'max_tokens') return 'length';
  if (reason === 'tool_use') return 'tools';
  if (reason === 'refusal') return 'filter';
  return 'end';
}

export function stopFromGemini(reason: unknown): StopKind {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY' || reason === 'RECITATION' || reason === 'BLOCKLIST' || reason === 'PROHIBITED_CONTENT') return 'filter';
  return 'end';
}

export const stopToOpenAI = (s: StopKind): string =>
  s === 'length' ? 'length' : s === 'tools' ? 'tool_calls' : s === 'filter' ? 'content_filter' : 'stop';

export const stopToAnthropic = (s: StopKind): string =>
  s === 'length' ? 'max_tokens' : s === 'tools' ? 'tool_use' : s === 'filter' ? 'refusal' : 'end_turn';

/** A Chat Completions answer body in canonical form. */
export function canonFromOpenAIAnswer(body: Json): CanonAnswer {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  let text = '';
  if (typeof message.content === 'string') text = message.content;
  else if (Array.isArray(message.content)) {
    text = message.content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('');
  }
  const toolCalls: CanonToolCall[] = [];
  if (Array.isArray(message.tool_calls)) {
    for (const call of message.tool_calls) {
      if (!isRecord(call) || !isRecord(call.function)) continue;
      toolCalls.push({ id: str(call.id), name: str(call.function.name), input: argumentsOf(call.function.arguments) });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text,
    toolCalls,
    stop: stopFromOpenAI(choice.finish_reason),
    inputTokens: numOrZero(usage.prompt_tokens),
    outputTokens: numOrZero(usage.completion_tokens),
  };
}

/** A Messages answer body in canonical form. */
export function canonFromAnthropicAnswer(body: Json): CanonAnswer {
  let text = '';
  const toolCalls: CanonToolCall[] = [];
  if (Array.isArray(body.content)) {
    for (const block of body.content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') text += block.text;
      else if (block.type === 'tool_use') {
        toolCalls.push({ id: str(block.id), name: str(block.name), input: isRecord(block.input) ? block.input : {} });
      }
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text,
    toolCalls,
    stop: stopFromAnthropic(body.stop_reason),
    inputTokens: numOrZero(usage.input_tokens),
    outputTokens: numOrZero(usage.output_tokens),
  };
}

/** A generateContent answer body in canonical form. */
export function canonFromGeminiAnswer(body: Json): CanonAnswer {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  let text = '';
  const toolCalls: CanonToolCall[] = [];
  if (Array.isArray(content.parts)) {
    let n = 0;
    for (const part of content.parts) {
      if (!isRecord(part)) continue;
      if (typeof part.text === 'string') text += part.text;
      else if (isRecord(part.functionCall)) {
        n += 1;
        toolCalls.push({
          id: `toolu_gemini_${n}`,
          name: str(part.functionCall.name),
          input: isRecord(part.functionCall.args) ? part.functionCall.args : {},
        });
      }
    }
  }
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text,
    toolCalls,
    stop: stopFromGemini(candidate.finishReason),
    inputTokens: numOrZero(usage.promptTokenCount),
    outputTokens: numOrZero(usage.candidatesTokenCount),
  };
}

/** The canonical answer as a Chat Completions body naming the model that served it. */
export function openAIAnswerJson(ans: CanonAnswer, model: string): Json {
  const message: Json = { role: 'assistant', content: ans.text !== '' ? ans.text : null };
  if (ans.toolCalls.length > 0) {
    message.tool_calls = ans.toolCalls.map((c) => ({
      id: c.id,
      type: 'function',
      function: { name: c.name, arguments: JSON.stringify(c.input) },
    }));
  }
  if (ans.text === '' && ans.toolCalls.length === 0) message.content = '';
  return {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion',
    created: 1767225600,
    model,
    choices: [{ index: 0, message, finish_reason: stopToOpenAI(ans.stop), logprobs: null }],
    usage: {
      prompt_tokens: ans.inputTokens,
      completion_tokens: ans.outputTokens,
      total_tokens: ans.inputTokens + ans.outputTokens,
    },
  };
}

/** The canonical answer as a Messages body naming the model that served it. */
export function anthropicAnswerJson(ans: CanonAnswer, model: string): Json {
  const content: Json[] = [];
  if (ans.text !== '') content.push({ type: 'text', text: ans.text });
  for (const c of ans.toolCalls) content.push({ type: 'tool_use', id: c.id, name: c.name, input: c.input });
  if (content.length === 0) content.push({ type: 'text', text: '' });
  return {
    id: 'msg_aggregate',
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopToAnthropic(ans.stop),
    stop_sequence: null,
    usage: { input_tokens: ans.inputTokens, output_tokens: ans.outputTokens },
  };
}

// ---------------------------------------------------------------------------
// Streaming: provider SSE -> answer events -> client SSE
// ---------------------------------------------------------------------------

export type AnswerEvent =
  | { type: 'text'; text: string }
  | { type: 'tool_start'; index: number; id: string; name: string }
  | { type: 'tool_args'; index: number; args: string }
  | { type: 'finish'; stop: StopKind; inputTokens: number; outputTokens: number };

/** One SSE frame: an optional event name and its joined data payload. */
export interface SseFrame {
  event: string | null;
  data: string;
}

/** Split a byte stream into SSE frames as they arrive. */
export async function* sseFrames(stream: ReadableStream<Uint8Array>): AsyncGenerator<SseFrame> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  const framesOf = (text: string): SseFrame[] => {
    const frames: SseFrame[] = [];
    for (const raw of text.split('\n\n')) {
      if (raw.trim() === '') continue;
      let event: string | null = null;
      const data: string[] = [];
      for (const line of raw.split('\n')) {
        if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
        else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
      }
      if (data.length > 0) frames.push({ event, data: data.join('\n') });
    }
    return frames;
  };
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const piece = buffer.slice(0, at + 2);
        buffer = buffer.slice(at + 2);
        for (const frame of framesOf(piece)) yield frame;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const frame of framesOf(buffer)) yield frame;
  } finally {
    reader.releaseLock();
  }
}

function parseJson(text: string): Json | null {
  try {
    const v: unknown = JSON.parse(text);
    return isRecord(v) ? v : null;
  } catch {
    return null;
  }
}

/** A Chat Completions SSE stream as answer events. */
export async function* openAIAnswerEvents(stream: ReadableStream<Uint8Array>): AsyncGenerator<AnswerEvent> {
  let stop: StopKind = 'end';
  let inputTokens = 0;
  let outputTokens = 0;
  for await (const frame of sseFrames(stream)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') break;
    const chunk = parseJson(payload);
    if (chunk === null) continue;
    if (isRecord(chunk.usage)) {
      inputTokens = numOrZero(chunk.usage.prompt_tokens);
      outputTokens = numOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string' && delta.content !== '') yield { type: 'text', text: delta.content };
    if (Array.isArray(delta.tool_calls)) {
      for (const call of delta.tool_calls) {
        if (!isRecord(call)) continue;
        const index = typeof call.index === 'number' ? call.index : 0;
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof call.id === 'string' || typeof fn.name === 'string') {
          yield { type: 'tool_start', index, id: str(call.id), name: str(fn.name) };
        }
        if (typeof fn.arguments === 'string' && fn.arguments !== '') yield { type: 'tool_args', index, args: fn.arguments };
      }
    }
    if (typeof choice.finish_reason === 'string') stop = stopFromOpenAI(choice.finish_reason);
  }
  yield { type: 'finish', stop, inputTokens, outputTokens };
}

/** A Messages SSE stream as answer events. */
export async function* anthropicAnswerEvents(stream: ReadableStream<Uint8Array>): AsyncGenerator<AnswerEvent> {
  let stop: StopKind = 'end';
  let inputTokens = 0;
  let outputTokens = 0;
  let toolCount = 0;
  for await (const frame of sseFrames(stream)) {
    const data = parseJson(frame.data);
    if (data === null) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      inputTokens = numOrZero(data.message.usage.input_tokens);
    } else if (frame.event === 'content_block_start' && isRecord(data.content_block) && data.content_block.type === 'tool_use') {
      toolCount += 1;
      yield {
        type: 'tool_start',
        index: toolCount - 1,
        id: str(data.content_block.id),
        name: str(data.content_block.name),
      };
    } else if (frame.event === 'content_block_delta' && isRecord(data.delta)) {
      if (data.delta.type === 'text_delta' && typeof data.delta.text === 'string' && data.delta.text !== '') {
        yield { type: 'text', text: data.delta.text };
      } else if (data.delta.type === 'input_json_delta' && typeof data.delta.partial_json === 'string' && data.delta.partial_json !== '') {
        yield { type: 'tool_args', index: Math.max(0, toolCount - 1), args: data.delta.partial_json };
      }
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stop = stopFromAnthropic(data.delta.stop_reason);
      if (isRecord(data.usage)) outputTokens = numOrZero(data.usage.output_tokens);
    }
  }
  yield { type: 'finish', stop, inputTokens, outputTokens };
}

/** A Gemini SSE stream as answer events. */
export async function* geminiAnswerEvents(stream: ReadableStream<Uint8Array>): AsyncGenerator<AnswerEvent> {
  let stop: StopKind = 'end';
  let inputTokens = 0;
  let outputTokens = 0;
  let toolCount = 0;
  for await (const frame of sseFrames(stream)) {
    const data = parseJson(frame.data);
    if (data === null) continue;
    if (isRecord(data.usageMetadata)) {
      inputTokens = numOrZero(data.usageMetadata.promptTokenCount);
      outputTokens = numOrZero(data.usageMetadata.candidatesTokenCount);
    }
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate === null) continue;
    const content = isRecord(candidate.content) ? candidate.content : {};
    if (Array.isArray(content.parts)) {
      for (const part of content.parts) {
        if (!isRecord(part)) continue;
        if (typeof part.text === 'string' && part.text !== '') yield { type: 'text', text: part.text };
        else if (isRecord(part.functionCall)) {
          yield { type: 'tool_start', index: toolCount, id: `toolu_gemini_${toolCount + 1}`, name: str(part.functionCall.name) };
          yield { type: 'tool_args', index: toolCount, args: JSON.stringify(isRecord(part.functionCall.args) ? part.functionCall.args : {}) };
          toolCount += 1;
        }
      }
    }
    if (typeof candidate.finishReason === 'string') stop = stopFromGemini(candidate.finishReason);
  }
  yield { type: 'finish', stop, inputTokens, outputTokens };
}

/** One server-sent event frame for the client. */
export function sseOut(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Answer events as a Chat Completions SSE stream naming the model that served them. */
export async function* openAIStreamOut(events: AsyncGenerator<AnswerEvent>, model: string): AsyncGenerator<string> {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-aggregate',
    object: 'chat.completion.chunk',
    created: 1767225600,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  yield sseOut(null, chunk({ role: 'assistant', content: '' }, null));
  for await (const ev of events) {
    if (ev.type === 'text') {
      yield sseOut(null, chunk({ content: ev.text }, null));
    } else if (ev.type === 'tool_start') {
      yield sseOut(
        null,
        chunk({ tool_calls: [{ index: ev.index, id: ev.id, type: 'function', function: { name: ev.name, arguments: '' } }] }, null),
      );
    } else if (ev.type === 'tool_args') {
      yield sseOut(null, chunk({ tool_calls: [{ index: ev.index, function: { arguments: ev.args } }] }, null));
    } else {
      yield sseOut(null, chunk({}, stopToOpenAI(ev.stop)));
      yield sseOut(null, {
        id: 'chatcmpl-aggregate',
        object: 'chat.completion.chunk',
        created: 1767225600,
        model,
        choices: [],
        usage: {
          prompt_tokens: ev.inputTokens,
          completion_tokens: ev.outputTokens,
          total_tokens: ev.inputTokens + ev.outputTokens,
        },
      });
    }
  }
  yield 'data: [DONE]\n\n';
}

/** Answer events as a Messages SSE stream naming the model that served them. */
export async function* anthropicStreamOut(events: AsyncGenerator<AnswerEvent>, model: string): AsyncGenerator<string> {
  yield sseOut('message_start', {
    type: 'message_start',
    message: {
      id: 'msg_aggregate',
      type: 'message',
      role: 'assistant',
      model,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: 0, output_tokens: 0 },
    },
  });
  let nextIndex = 0;
  let openIndex = -1;
  let openKind: 'text' | 'tool' | null = null;
  const closeBlock = (): string | null => {
    if (openIndex === -1) return null;
    const frame = sseOut('content_block_stop', { type: 'content_block_stop', index: openIndex });
    openIndex = -1;
    openKind = null;
    return frame;
  };
  for await (const ev of events) {
    if (ev.type === 'text') {
      if (openKind !== 'text') {
        const closed = closeBlock();
        if (closed !== null) yield closed;
        openIndex = nextIndex;
        nextIndex += 1;
        openKind = 'text';
        yield sseOut('content_block_start', {
          type: 'content_block_start',
          index: openIndex,
          content_block: { type: 'text', text: '' },
        });
      }
      yield sseOut('content_block_delta', {
        type: 'content_block_delta',
        index: openIndex,
        delta: { type: 'text_delta', text: ev.text },
      });
    } else if (ev.type === 'tool_start') {
      const closed = closeBlock();
      if (closed !== null) yield closed;
      openIndex = nextIndex;
      nextIndex += 1;
      openKind = 'tool';
      yield sseOut('content_block_start', {
        type: 'content_block_start',
        index: openIndex,
        content_block: { type: 'tool_use', id: ev.id !== '' ? ev.id : `toolu_${openIndex}`, name: ev.name, input: {} },
      });
    } else if (ev.type === 'tool_args') {
      if (openKind === 'tool') {
        yield sseOut('content_block_delta', {
          type: 'content_block_delta',
          index: openIndex,
          delta: { type: 'input_json_delta', partial_json: ev.args },
        });
      }
    } else {
      const closed = closeBlock();
      if (closed !== null) yield closed;
      yield sseOut('message_delta', {
        type: 'message_delta',
        delta: { stop_reason: stopToAnthropic(ev.stop), stop_sequence: null },
        usage: { output_tokens: ev.outputTokens },
      });
      yield sseOut('message_stop', { type: 'message_stop' });
    }
  }
}

/** An error body in the client's wire. */
export function clientError(wire: 'openai' | 'anthropic', type: string, message: string): Json {
  if (wire === 'anthropic') return { type: 'error', error: { type, message } };
  return { error: { message, type, code: null } };
}
