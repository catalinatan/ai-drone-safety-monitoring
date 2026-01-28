import { useState } from 'react';
import { DroneMode } from '../types';
import { ArrowUp, ArrowDown, ArrowLeft, ArrowRight, Home, Package } from 'lucide-react';

interface DroneControlsPanelProps {
  mode: DroneMode;
  onModeChange: (mode: DroneMode) => void;
}

export default function DroneControlsPanel({ mode, onModeChange }: DroneControlsPanelProps) {
  const [lastCommand, setLastCommand] = useState<string>('');
  const [dropConfirm, setDropConfirm] = useState(false);
  const [showHomeConfirm, setShowHomeConfirm] = useState(false);

  const isManual = mode === 'manual';

  const handleMovement = (direction: string) => {
    if (!isManual) {
      setLastCommand('Switch to Manual mode to control movement');
      return;
    }
    setLastCommand(`Move ${direction} command sent`);
    console.log(`Drone movement: ${direction}`);
  };

  const handleReturnHome = () => {
    setShowHomeConfirm(true);
    setLastCommand('Return-to-home command issued');
    console.log('Return to home initiated');
    setTimeout(() => setShowHomeConfirm(false), 3000);
  };

  const handleDropEquipment = () => {
    if (!dropConfirm) {
      setDropConfirm(true);
      setLastCommand('Click "Drop safety equipment" again to confirm');
      return;
    }
    setLastCommand('Drop safety equipment command executed');
    console.log('Safety equipment dropped');
    setDropConfirm(false);
  };

  return (
    <div className="bg-gray-800 rounded-lg border border-gray-700">
      <div className="bg-gray-750 px-4 py-3 border-b border-gray-700">
        <h3 className="text-lg font-semibold">Drone Controls</h3>
      </div>

      <div className="p-6 space-y-6">
        <div>
          <label className="text-sm font-medium mb-2 block">Control Mode</label>
          <div className="flex rounded-lg overflow-hidden border border-gray-600">
            <button
              onClick={() => onModeChange('automatic')}
              className={`flex-1 px-4 py-2.5 text-sm font-medium transition-colors ${
                mode === 'automatic'
                  ? 'bg-cyan-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-650'
              }`}
            >
              Automatic
            </button>
            <button
              onClick={() => onModeChange('manual')}
              className={`flex-1 px-4 py-2.5 text-sm font-medium transition-colors ${
                mode === 'manual'
                  ? 'bg-cyan-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-650'
              }`}
            >
              Manual
            </button>
          </div>
          {!isManual && (
            <p className="text-xs text-yellow-500 mt-2">
              Switch to Manual to directly control the drone
            </p>
          )}
        </div>

        <div>
          <label className="text-sm font-medium mb-3 block">Movement Controls</label>
          <div className="flex gap-6">
            <div className="flex-1">
              <p className="text-xs text-gray-400 mb-2">Directional</p>
              <div className="grid grid-cols-3 gap-2 max-w-[180px]">
                <div></div>
                <button
                  onClick={() => handleMovement('forward')}
                  disabled={!isManual}
                  className={`p-4 rounded-lg transition-all ${
                    isManual
                      ? 'bg-gray-700 hover:bg-cyan-600 active:bg-cyan-700'
                      : 'bg-gray-800 text-gray-600 cursor-not-allowed'
                  }`}
                  title="Forward"
                >
                  <ArrowUp size={24} className="mx-auto" />
                </button>
                <div></div>

                <button
                  onClick={() => handleMovement('left')}
                  disabled={!isManual}
                  className={`p-4 rounded-lg transition-all ${
                    isManual
                      ? 'bg-gray-700 hover:bg-cyan-600 active:bg-cyan-700'
                      : 'bg-gray-800 text-gray-600 cursor-not-allowed'
                  }`}
                  title="Left"
                >
                  <ArrowLeft size={24} className="mx-auto" />
                </button>
                <div className="flex items-center justify-center text-gray-600">
                  <div className="w-12 h-12 rounded-full border-2 border-gray-700"></div>
                </div>
                <button
                  onClick={() => handleMovement('right')}
                  disabled={!isManual}
                  className={`p-4 rounded-lg transition-all ${
                    isManual
                      ? 'bg-gray-700 hover:bg-cyan-600 active:bg-cyan-700'
                      : 'bg-gray-800 text-gray-600 cursor-not-allowed'
                  }`}
                  title="Right"
                >
                  <ArrowRight size={24} className="mx-auto" />
                </button>

                <div></div>
                <button
                  onClick={() => handleMovement('backward')}
                  disabled={!isManual}
                  className={`p-4 rounded-lg transition-all ${
                    isManual
                      ? 'bg-gray-700 hover:bg-cyan-600 active:bg-cyan-700'
                      : 'bg-gray-800 text-gray-600 cursor-not-allowed'
                  }`}
                  title="Backward"
                >
                  <ArrowDown size={24} className="mx-auto" />
                </button>
                <div></div>
              </div>
            </div>

            <div>
              <p className="text-xs text-gray-400 mb-2">Altitude</p>
              <div className="flex flex-col gap-2">
                <button
                  onClick={() => handleMovement('up')}
                  disabled={!isManual}
                  className={`p-4 rounded-lg transition-all ${
                    isManual
                      ? 'bg-gray-700 hover:bg-cyan-600 active:bg-cyan-700'
                      : 'bg-gray-800 text-gray-600 cursor-not-allowed'
                  }`}
                  title="Altitude Up"
                >
                  <ArrowUp size={24} className="mx-auto" />
                </button>
                <button
                  onClick={() => handleMovement('down')}
                  disabled={!isManual}
                  className={`p-4 rounded-lg transition-all ${
                    isManual
                      ? 'bg-gray-700 hover:bg-cyan-600 active:bg-cyan-700'
                      : 'bg-gray-800 text-gray-600 cursor-not-allowed'
                  }`}
                  title="Altitude Down"
                >
                  <ArrowDown size={24} className="mx-auto" />
                </button>
              </div>
            </div>
          </div>
        </div>

        <div className="space-y-3 pt-4 border-t border-gray-700">
          <button
            onClick={handleReturnHome}
            className="w-full flex items-center justify-center gap-3 px-4 py-3 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors font-medium"
          >
            <Home size={20} />
            Return to Home
          </button>

          <button
            onClick={handleDropEquipment}
            className={`w-full flex items-center justify-center gap-3 px-4 py-3 rounded-lg transition-all font-medium ${
              dropConfirm
                ? 'bg-orange-600 hover:bg-orange-700 ring-2 ring-orange-400 animate-pulse'
                : 'bg-orange-500 hover:bg-orange-600'
            }`}
          >
            <Package size={20} />
            {dropConfirm ? 'Click again to confirm drop' : 'Drop safety equipment'}
          </button>
        </div>

        {lastCommand && (
          <div className="bg-gray-750 rounded-lg p-3 border border-gray-600">
            <p className="text-xs text-gray-400 mb-1">Last Command:</p>
            <p className="text-sm font-medium">{lastCommand}</p>
          </div>
        )}

        {showHomeConfirm && (
          <div className="bg-blue-600/20 border border-blue-500 rounded-lg p-3">
            <p className="text-sm text-blue-300">Return-to-home sequence initiated</p>
          </div>
        )}
      </div>
    </div>
  );
}
