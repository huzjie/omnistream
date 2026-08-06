/**
 * OmniStream TypeScript SDK
 *
 * 实时全双工多模态客户端：连接 OmniStream 网关，收发
 * 文本/音频/视频/工具调用事件，支持打断与主动记忆注入。
 */

export enum EventKind {
  CONNECTED = 'connected',
  AUDIO_DELTA = 'audio_delta',
  AUDIO_DONE = 'audio_done',
  TEXT_DELTA = 'text_delta',
  TEXT_DONE = 'text_done',
  TOOL_CALL = 'tool_call',
  RESPONSE_DONE = 'response_done',
  RESPONSE_CANCELLED = 'response_cancelled',
  USAGE = 'usage',
  ERROR = 'error',
  CLOSED = 'closed',
}

export interface ClientEvent {
  kind: EventKind | string
  ts_ms: number
  text?: string
  audio?: string // base64
  tool_call?: { call_id: string; name: string; arguments: Record<string, unknown> }
  stop_reason?: string
  role?: string
  is_final?: boolean
  usage?: Record<string, number>
  error?: string
}

export interface SessionOptions {
  provider?: string
  model?: string
  modalities?: string[]
  instructions?: string
  voice?: string
  tools?: Array<{ name: string; description?: string; parameters?: unknown }>
}

const API_ROOT = 'http://localhost:8000'

/**
 * OmniStream 实时客户端（WebSocket 二进制帧 + JSON 事件）。
 */
export class RealtimeClient {
  private ws: WebSocket | null = null
  private listeners = new Map<string, Set<(ev: ClientEvent) => void>>()
  private sessionId: string | null = null

  constructor(
    private opts: SessionOptions = {},
    private baseUrl: string = API_ROOT,
  ) {}

  get connected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN
  }

  get id(): string | null {
    return this.sessionId
  }

  async connect(): Promise<string> {
    // 1. 创建会话
    const resp = await fetch(`${this.baseUrl}/api/sessions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        provider: this.opts.provider ?? 'mock',
        model: this.opts.model ?? null,
        modalities: this.opts.modalities ?? ['text', 'audio'],
        instructions: this.opts.instructions ?? '你是一个自然、简洁、可被随时打断的语音助手。',
        voice: this.opts.voice ?? 'default',
        tools: this.opts.tools ?? [],
      }),
    })
    if (!resp.ok) {
      throw new Error(`创建会话失败: ${resp.status} ${await resp.text()}`)
    }
    const data = await resp.json()
    this.sessionId = data.id ?? data.session_id

    // 2. 建立实时通道
    const wsUrl = this.baseUrl.replace(/^http/, 'ws') + `/api/sessions/${this.sessionId}/realtime`
    this.ws = new WebSocket(wsUrl)
    await new Promise<void>((resolve, reject) => {
      this.ws!.onopen = () => resolve()
      this.ws!.onerror = (e) => reject(new Error('WebSocket 连接失败'))
    })
    this.ws.onmessage = (msg) => this.dispatch(this.decode(msg.data))
    return this.sessionId
  }

  private decode(raw: unknown): ClientEvent {
    if (typeof raw === 'string') {
      return JSON.parse(raw) as ClientEvent
    }
    // ArrayBuffer 二进制帧：前 2 字节为事件码（暂以 JSON 文本事件为准）
    const text = new TextDecoder().decode(raw as ArrayBuffer)
    try {
      return JSON.parse(text) as ClientEvent
    } catch {
      return { kind: EventKind.AUDIO_DELTA, ts_ms: Date.now(), audio: btoa(text) }
    }
  }

  private dispatch(ev: ClientEvent): void {
    const set = this.listeners.get(ev.kind)
    if (set) set.forEach((fn) => fn(ev))
    const all = this.listeners.get('*')
    if (all) all.forEach((fn) => fn(ev))
  }

  on(kind: EventKind | string | '*', fn: (ev: ClientEvent) => void): () => void {
    const key = kind
    if (!this.listeners.has(key)) this.listeners.set(key, new Set())
    this.listeners.get(key)!.add(fn)
    return () => this.listeners.get(key)?.delete(fn)
  }

  sendText(text: string): void {
    this.send({ type: 'text', text })
  }

  /** 请求模型生成一轮回复（可在生成中途调用触发打断+重答）。 */
  requestResponse(instructionsSuffix?: string): void {
    this.send({ type: 'request_response', instructions_suffix: instructionsSuffix })
  }

  cancelResponse(): void {
    this.send({ type: 'cancel_response' })
  }

  submitToolResult(callId: string, result: unknown): void {
    this.send({ type: 'tool_result', call_id: callId, result })
  }

  private send(payload: unknown): void {
    if (!this.connected) throw new Error('未连接')
    this.ws!.send(JSON.stringify(payload))
  }

  close(): void {
    if (this.ws) this.ws.close()
    this.ws = null
  }
}

export default RealtimeClient
