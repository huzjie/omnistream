import { useEffect, useState } from 'react'
import { createSession, fetchSessions } from '../../api/client'

export default function SessionsPage() {
  const [sessions, setSessions] = useState<any[]>([])
  const [provider, setProvider] = useState('mock')
  const [loading, setLoading] = useState(false)

  const load = () => {
    fetchSessions({ limit: 50 })
      .then((d) => setSessions(d?.items ?? d ?? []))
      .catch(() => setSessions([]))
  }

  useEffect(load, [])

  const handleCreate = async () => {
    setLoading(true)
    try {
      await createSession({
        provider,
        modalities: ['text', 'audio'],
        end_user_id: 'console',
      })
      load()
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-xl font-semibold">会话管理</h1>
        <div className="flex gap-2">
          <select
            value={provider}
            onChange={(e) => setProvider(e.target.value)}
            className="border border-slate-300 rounded-md px-2 py-1.5 text-sm"
          >
            <option value="mock">mock</option>
            <option value="doubao_seedrealtime">doubao_seedrealtime</option>
            <option value="openai_realtime">openai_realtime</option>
            <option value="gemini_live">gemini_live</option>
            <option value="qwen_omni">qwen_omni</option>
            <option value="deepseek_v4">deepseek_v4</option>
            <option value="kimi_k3">kimi_k3</option>
            <option value="local_pipeline">local_pipeline</option>
          </select>
          <button
            onClick={handleCreate}
            disabled={loading}
            className="bg-blue-600 text-white text-sm px-4 py-1.5 rounded-md hover:bg-blue-700 disabled:opacity-50"
          >
            {loading ? '创建中…' : '新建会话'}
          </button>
        </div>
      </div>

      <div className="card overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-slate-500 border-b bg-slate-50">
              <th className="px-4 py-2">ID</th>
              <th>Provider</th>
              <th>状态</th>
              <th>模型</th>
              <th>创建时间</th>
            </tr>
          </thead>
          <tbody>
            {sessions.map((s: any) => (
              <tr key={s.id} className="border-b last:border-0 hover:bg-slate-50">
                <td className="px-4 py-2 font-mono text-xs">{s.id}</td>
                <td>{s.provider}</td>
                <td>{s.state ?? s.status ?? '—'}</td>
                <td>{s.model ?? '—'}</td>
                <td className="text-slate-500">{s.created_at ?? '—'}</td>
              </tr>
            ))}
            {sessions.length === 0 && (
              <tr>
                <td colSpan={5} className="px-4 py-8 text-center text-slate-400">
                  暂无会话
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}
