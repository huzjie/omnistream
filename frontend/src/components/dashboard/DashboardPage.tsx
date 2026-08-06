import { useEffect, useState } from 'react'
import { fetchMetrics, fetchSessions, fetchProviders } from '../../api/client'

interface MetricSummary {
  active_sessions?: number
  total_sessions?: number
  avg_rtt_ms?: number
  avg_ttft_ms?: number
  providers?: number
}

export default function DashboardPage() {
  const [metrics, setMetrics] = useState<MetricSummary | null>(null)
  const [sessions, setSessions] = useState<any[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.allSettled([fetchMetrics(), fetchSessions({ limit: 10 }), fetchProviders()])
      .then(([m, s]) => {
        if (m.status === 'fulfilled') setMetrics(m.value)
        if (s.status === 'fulfilled') setSessions(s.value?.items ?? s.value ?? [])
      })
      .finally(() => setLoading(false))
  }, [])

  const cards = [
    { label: '活跃会话', value: metrics?.active_sessions ?? '—' },
    { label: '累计会话', value: metrics?.total_sessions ?? '—' },
    { label: '平均 RTT (ms)', value: metrics?.avg_rtt_ms ?? '—' },
    { label: '平均 TTFT (ms)', value: metrics?.avg_ttft_ms ?? '—' },
  ]

  return (
    <div>
      <h1 className="text-xl font-semibold mb-4">实时交互总览</h1>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        {cards.map((c) => (
          <div key={c.label} className="card p-4">
            <div className="stat-label">{c.label}</div>
            <div className="stat-value mt-1">{c.value}</div>
          </div>
        ))}
      </div>

      <div className="card p-4">
        <h2 className="text-base font-medium mb-3">最近会话</h2>
        {loading ? (
          <div className="text-sm text-slate-400">加载中…</div>
        ) : sessions.length === 0 ? (
          <div className="text-sm text-slate-400">
            暂无会话。可在「会话」页创建第一个实时会话。
          </div>
        ) : (
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-slate-500 border-b">
                <th className="py-2">ID</th>
                <th>Provider</th>
                <th>状态</th>
                <th>创建时间</th>
              </tr>
            </thead>
            <tbody>
              {sessions.map((s: any) => (
                <tr key={s.id} className="border-b last:border-0">
                  <td className="py-2 font-mono text-xs">{s.id}</td>
                  <td>{s.provider}</td>
                  <td>{s.state ?? s.status ?? '—'}</td>
                  <td className="text-slate-500">{s.created_at ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
