import { useEffect, useState } from 'react'
import { fetchProviders } from '../../api/client'

export default function ProvidersPage() {
  const [providers, setProviders] = useState<any[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchProviders()
      .then((data) => setProviders(data?.items ?? data?.providers ?? []))
      .catch(() => setProviders([]))
      .finally(() => setLoading(false))
  }, [])

  return (
    <div>
      <h1 className="text-xl font-semibold mb-4">Provider 适配器</h1>
      {loading ? (
        <div className="text-sm text-slate-400">加载中…</div>
      ) : (
        <div className="grid md:grid-cols-2 gap-4">
          {providers.map((p: any) => (
            <div key={p.name} className="card p-4">
              <div className="flex items-center justify-between">
                <span className="font-medium">{p.display_name ?? p.name}</span>
                <span
                  className={`px-2 py-0.5 rounded text-xs ${
                    p.configured
                      ? 'bg-emerald-50 text-emerald-700'
                      : 'bg-slate-100 text-slate-500'
                  }`}
                >
                  {p.configured ? '已配置' : '未配置'}
                </span>
              </div>
              <div className="mt-2 text-xs text-slate-500">
                类型: {p.kind} · 模型: {(p.models ?? []).join(', ') || '—'}
              </div>
              <div className="mt-2 flex flex-wrap gap-1">
                {Object.entries(p.capabilities ?? {})
                  .filter(([, v]) => v === true)
                  .map(([k]) => (
                    <span
                      key={k}
                      className="px-1.5 py-0.5 bg-blue-50 text-blue-600 rounded text-[10px]"
                    >
                      {k}
                    </span>
                  ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
