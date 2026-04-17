import { useState } from 'react';
import { Camera } from '../types';
import { X, Square, Eraser } from 'lucide-react';

interface EditAnnotationsPanelProps {
  camera: Camera;
  onSave: () => void;
  onCancel: () => void;
}

type ToolType = 'red' | 'yellow' | 'green' | 'eraser' | null;

export default function EditAnnotationsPanel({ camera, onSave, onCancel }: EditAnnotationsPanelProps) {
  const [selectedTool, setSelectedTool] = useState<ToolType>(null);
  const [showSaveToast, setShowSaveToast] = useState(false);

  const handleSave = () => {
    setShowSaveToast(true);
    setTimeout(() => {
      setShowSaveToast(false);
      onSave();
    }, 1500);
  };

  return (
    <div className="mt-6 bg-gray-800 rounded-lg border border-gray-700 shadow-xl">
      <div className="flex items-center justify-between p-4 border-b border-gray-700">
        <div>
          <h3 className="text-lg font-semibold">Edit Hazardous Zones - {camera.name}</h3>
          <p className="text-xs text-gray-400 mt-1">
            Future: persist annotations to database and apply auto-segmentation
          </p>
        </div>
        <button
          onClick={onCancel}
          className="p-2 hover:bg-gray-700 rounded transition-colors"
          title="Close"
        >
          <X size={20} />
        </button>
      </div>

      <div className="p-4">
        <div className="flex flex-col lg:flex-row gap-4">
          <div className="lg:w-48 space-y-2">
            <h4 className="text-sm font-semibold mb-3">Annotation Tools</h4>

            <button
              onClick={() => setSelectedTool('red')}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all ${
                selectedTool === 'red'
                  ? 'bg-red-600 ring-2 ring-red-400'
                  : 'bg-gray-700 hover:bg-gray-600'
              }`}
            >
              <Square className="fill-red-500" size={20} />
              <div className="text-left">
                <div className="font-medium">High Hazard</div>
                <div className="text-xs text-gray-300">Red</div>
              </div>
            </button>

            <button
              onClick={() => setSelectedTool('yellow')}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all ${
                selectedTool === 'yellow'
                  ? 'bg-yellow-600 ring-2 ring-yellow-400'
                  : 'bg-gray-700 hover:bg-gray-600'
              }`}
            >
              <Square className="fill-yellow-500" size={20} />
              <div className="text-left">
                <div className="font-medium">Medium Hazard</div>
                <div className="text-xs text-gray-300">Yellow</div>
              </div>
            </button>

            <button
              onClick={() => setSelectedTool('green')}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all ${
                selectedTool === 'green'
                  ? 'bg-green-600 ring-2 ring-green-400'
                  : 'bg-gray-700 hover:bg-gray-600'
              }`}
            >
              <Square className="fill-green-500" size={20} />
              <div className="text-left">
                <div className="font-medium">Low Hazard</div>
                <div className="text-xs text-gray-300">Green</div>
              </div>
            </button>

            <button
              onClick={() => setSelectedTool('eraser')}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all ${
                selectedTool === 'eraser'
                  ? 'bg-gray-600 ring-2 ring-gray-400'
                  : 'bg-gray-700 hover:bg-gray-600'
              }`}
            >
              <Eraser size={20} />
              <div className="text-left">
                <div className="font-medium">Eraser</div>
                <div className="text-xs text-gray-300">Clear zones</div>
              </div>
            </button>
          </div>

          <div className="flex-1">
            <div className="aspect-video bg-gray-700 rounded-lg flex items-center justify-center text-gray-500 border-2 border-dashed border-gray-600 relative overflow-hidden cursor-crosshair">
              <div className="text-center relative z-10">
                <div className="text-5xl mb-3">📹</div>
                <div className="text-lg font-medium">{camera.name}</div>
                <div className="text-sm text-gray-600 mt-2">
                  {selectedTool
                    ? `${selectedTool === 'eraser' ? 'Eraser' : selectedTool.toUpperCase()} tool selected - Click and drag to annotate`
                    : 'Select a tool to start annotating'}
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="flex justify-end gap-3 mt-6 pt-4 border-t border-gray-700">
          <button
            onClick={onCancel}
            className="px-6 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            className="px-6 py-2 bg-cyan-600 hover:bg-cyan-700 rounded-lg transition-colors font-medium"
          >
            Save annotations
          </button>
        </div>
      </div>

      {showSaveToast && (
        <div className="fixed top-4 right-4 bg-green-600 text-white px-6 py-3 rounded-lg shadow-lg z-50 animate-pulse">
          Annotations saved (not persisted yet)
        </div>
      )}
    </div>
  );
}
