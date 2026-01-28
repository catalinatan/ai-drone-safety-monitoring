import { useState } from 'react';
import MainControlTab from './components/MainControlTab';
import DroneControlTab from './components/DroneControlTab';

type Tab = 'main' | 'drone';

function App() {
  const [activeTab, setActiveTab] = useState<Tab>('main');

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100">
      <header className="bg-gray-800 border-b border-gray-700 px-6 py-4 shadow-lg">
        <h1 className="text-2xl font-bold text-white">Safety Command Center</h1>
      </header>

      <div className="border-b border-gray-700 bg-gray-800">
        <div className="flex px-6">
          <button
            onClick={() => setActiveTab('main')}
            className={`px-6 py-3 text-sm font-medium transition-all ${
              activeTab === 'main'
                ? 'text-cyan-400 border-b-2 border-cyan-400 bg-gray-750'
                : 'text-gray-400 hover:text-gray-200 hover:bg-gray-750'
            }`}
          >
            Main Control
          </button>
          <button
            onClick={() => setActiveTab('drone')}
            className={`px-6 py-3 text-sm font-medium transition-all ${
              activeTab === 'drone'
                ? 'text-cyan-400 border-b-2 border-cyan-400 bg-gray-750'
                : 'text-gray-400 hover:text-gray-200 hover:bg-gray-750'
            }`}
          >
            Drone Control
          </button>
        </div>
      </div>

      <main className="p-6">
        {activeTab === 'main' ? <MainControlTab /> : <DroneControlTab />}
      </main>
    </div>
  );
}

export default App;
