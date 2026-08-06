import { Routes, Route } from 'react-router-dom'
import Layout from './components/layout/Layout'
import DashboardPage from './components/dashboard/DashboardPage'
import ProvidersPage from './components/providers/ProvidersPage'
import SessionsPage from './components/sessions/SessionsPage'
import MemoryPage from './components/memory/MemoryPage'
import SettingsPage from './components/settings/SettingsPage'

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<DashboardPage />} />
        <Route path="/providers" element={<ProvidersPage />} />
        <Route path="/sessions" element={<SessionsPage />} />
        <Route path="/memory" element={<MemoryPage />} />
        <Route path="/settings" element={<SettingsPage />} />
      </Routes>
    </Layout>
  )
}
