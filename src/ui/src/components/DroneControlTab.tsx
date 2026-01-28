import { useState } from 'react';
import { DroneMode } from '../types';
import DroneVideoPanel from './DroneVideoPanel';
import DroneControlsPanel from './DroneControlsPanel';

export default function DroneControlTab() {
  const [mode, setMode] = useState<DroneMode>('automatic');
  const [batteryLevel] = useState(78);
  const [isConnected] = useState(true);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <div className="lg:col-span-2">
        <DroneVideoPanel
          mode={mode}
          batteryLevel={batteryLevel}
          isConnected={isConnected}
        />
      </div>

      <div className="lg:col-span-1">
        <DroneControlsPanel mode={mode} onModeChange={setMode} />
      </div>
    </div>
  );
}
