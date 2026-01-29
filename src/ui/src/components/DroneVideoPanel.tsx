import { DroneMode } from '../types';
import { Battery, Wifi } from 'lucide-react';

interface DroneVideoPanelProps {
  mode: DroneMode;
  batteryLevel: number;
  isConnected: boolean;
}

export default function DroneVideoPanel({ mode, batteryLevel, isConnected }: DroneVideoPanelProps) {
  return (
    <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
      <div className="bg-gray-750 px-4 py-3 border-b border-gray-700">
        <h3 className="text-lg font-semibold">Drone Camera Feed</h3>
      </div>

      <div className="relative">
        <div className="absolute top-4 left-4 z-10 flex gap-3">
          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium ${
            isConnected ? 'bg-green-600/90' : 'bg-red-600/90'
          }`}>
            <Wifi size={14} />
            <span>{isConnected ? 'Connected' : 'Disconnected'}</span>
          </div>

          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium ${
            batteryLevel > 50 ? 'bg-green-600/90' : batteryLevel > 20 ? 'bg-yellow-600/90' : 'bg-red-600/90'
          }`}>
            <Battery size={14} />
            <span>Battery: {batteryLevel}%</span>
          </div>

          <div className="px-3 py-1.5 rounded-lg text-xs font-medium bg-cyan-600/90">
            Mode: {mode === 'automatic' ? 'Automatic' : 'Manual'}
          </div>
        </div>

        <div className="aspect-video bg-gray-900 relative flex items-center justify-center overflow-hidden">
          {isConnected ? (
            <img 
              src="http://localhost:8000/video_feed" 
              alt="Drone Camera Feed" 
              className="w-full h-full object-contain"
            />
          ) : (
             <div className="text-center text-gray-500">
              <div className="text-6xl mb-3">🚁</div>
              <div className="text-xl font-medium">Drone Offline</div>
              <div className="text-sm text-gray-600 mt-2">Waiting for connection...</div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
