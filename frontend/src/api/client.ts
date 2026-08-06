import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 15_000,
})

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('oms_token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

export default api

// ---------- 会话 ----------
export const fetchSessions = (params?: Record<string, unknown>) =>
  api.get('/sessions', { params }).then((r) => r.data)

export const createSession = (data: Record<string, unknown>) =>
  api.post('/sessions', data).then((r) => r.data)

export const fetchSessionDetail = (id: string) =>
  api.get(`/sessions/${id}`).then((r) => r.data)

export const closeSession = (id: string) =>
  api.post(`/sessions/${id}/close`).then((r) => r.data)

// ---------- Providers ----------
export const fetchProviders = () =>
  api.get('/providers').then((r) => r.data)

// ---------- 指标 ----------
export const fetchMetrics = () =>
  api.get('/metrics/summary').then((r) => r.data)

// ---------- 记忆 ----------
export const fetchMemory = (params?: Record<string, unknown>) =>
  api.get('/memory', { params }).then((r) => r.data)

export const ingestMemory = (data: Record<string, unknown>) =>
  api.post('/memory/ingest', data).then((r) => r.data)

export const runMemoryMaintenance = () =>
  api.post('/memory/maintenance').then((r) => r.data)
