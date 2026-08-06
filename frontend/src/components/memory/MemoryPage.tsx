import { useEffect, useState } from 'react'
import { fetchMemory, ingestMemory, runMemoryMaintenance } from '../../api/client'

export default function MemoryPage() {
  const [items, setItems] = useState<any[]>([])
  const [text, setText] = useState('')
  const [layer, setLayer] = useState('stable_knowledge')

  const load = () => {
    fetchMemory({ limit: 100 })
      .then((d) => setItems(d?.items ?? d ?? []))
      .catch(() => setItems([]))
  }

  useEffect(load, [])

  const handleIngest = async () => {
    if (!text.trim()) return
    await ingestMemory({ text, layer, role: 'user' })
    setText('')
    load()
  }

  const handleMaintenance = async () => {
    await runMemoryMaintenance()
    load()
  }

  const layerColor: Record<string, string> = {
    session: 'bg-amber-50 text-amber-700',
    private_state: 'bg-rose-50 text-rose-700',
    stable_knowledge: 'bg-emerald-50 text-emerald-700',
    procedural: 'bg-indigo-50 text-indigo-700',
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-xl font-semibold">主动记忆治理</h1>
        <button
          onClick={handleMaintenance}
          className="text-sm border border-slate-300 px-3 py-1.5 rounded-md hover:bg-slate-50"
        >
          执行维护（遗忘/压缩）
        </button>
      </div>

      <div className="card p-4 mb-4 flex gap-2">
        <input
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="输入一条要记忆的内容…"
          className="flex-1 border border-slate-300 rounded-md px-3 py-2 text-sm"
        />
        <select
          value={layer}
          onChange={(e) => setLayer(e.target.value)}
          className="border border-slate-300 rounded-md px-2 py-2 text-sm"
        >
          <option value="session">session</option>
          <option value="private_state">private_state</option>
          <option value="stable_knowledge">stable_knowledge</option>
          <option value="procedural">procedural</option>
        </select>
        <button
          onClick={handleIngest}
          className="bg-blue-600 text-white text-sm px-4 rounded-md hover:bg-blue-700"
        >
          摄入
        </button>
      </div>

      <div className="space-y-2">
        {items.map((it: any) => (
          <div key={it.id} className="card p-3 flex items-start gap-3">
            <span
              className={`px-2 py-0.5 rounded text-[10px] shrink-0 ${
                layerColor[it.layer] ?? 'bg-slate-100 text-slate-600'
              }`}
            >
              {it.layer}
            </span>
            <div className="flex-1">
              <div className="text-sm">{it.text}</div>
              <div className="mt-1 text-xs text-slate-400">
                role={it.role} · confidence={it.confidence?.toFixed(2)} · reinforced=
                {it.reinforced ?? 0}
              </div>
            </div>
          </div>
        ))}
        {items.length === 0 && (
          <div className="card p-8 text-center text-slate-400 text-sm">
            暂无记忆。摄入后，对话编排器将通过三闸门选择性注入。
          </div>
        )}
      </div>
    </div>
  )
}
