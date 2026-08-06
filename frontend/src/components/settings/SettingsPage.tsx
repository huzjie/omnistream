import { useState } from 'react'

export default function SettingsPage() {
  const [apiKey, setApiKey] = useState(localStorage.getItem('oms_token') ?? '')

  const save = () => {
    localStorage.setItem('oms_token', apiKey.trim())
    alert('已保存 API Token')
  }

  return (
    <div>
      <h1 className="text-xl font-semibold mb-4">设置</h1>
      <div className="card p-5 max-w-md">
        <label className="block text-sm font-medium mb-1">
          API Token（可选，用于控制台鉴权）
        </label>
        <input
          type="password"
          value={apiKey}
          onChange={(e) => setApiKey(e.target.value)}
          placeholder="oms_xxx 或 JWT"
          className="w-full border border-slate-300 rounded-md px-3 py-2 text-sm mb-3"
        />
        <button
          onClick={save}
          className="bg-blue-600 text-white text-sm px-4 py-1.5 rounded-md hover:bg-blue-700"
        >
          保存
        </button>
        <p className="mt-4 text-xs text-slate-400 leading-relaxed">
          完整配置请通过后端 .env 管理（OMS_ 前缀环境变量），例如
          OMS_PROVIDERS__DEFAULT=mock 可直接以 mock 模式运行，
          无需任何外部 API Key。
        </p>
      </div>
    </div>
  )
}
