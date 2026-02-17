import { Camera } from '../types';
import { X, Pencil } from 'lucide-react';

interface ZoomPanelProps {
  camera: Camera;
  onClose: () => void;
  onEdit: () => void;
}

export default function ZoomPanel({ camera, onClose, onEdit }: ZoomPanelProps) {
  return (
    <div className="mt-6 bg-gray-800 rounded-lg border border-gray-700 shadow-xl">
      <div className="flex items-center justify-between p-4 border-b border-gray-700">
        <div>
          <h3 className="text-lg font-semibold">{camera.name} - Zoom View</h3>
          <p className="text-sm text-gray-400">Status: Active</p>
        </div>
        <button
          onClick={onClose}
          className="p-2 hover:bg-gray-700 rounded transition-colors"
          title="Close"
        >
          <X size={20} />
        </button>
      </div>

      <div className="p-4">
        <div className="aspect-video bg-gray-700 rounded-lg flex items-center justify-center text-gray-500 mb-4">
          <div className="text-center">
            <div className="text-6xl mb-3">📹</div>
            <div className="text-xl font-medium">{camera.name}</div>
            <div className="text-sm text-gray-600 mt-2">Enlarged Live Feed</div>
          </div>
        </div>

        <div className="flex gap-3">
          <button
            onClick={onEdit}
            className="flex items-center gap-2 px-4 py-2 bg-cyan-600 hover:bg-cyan-700 rounded-lg transition-colors font-medium"
          >
            <Pencil size={18} />
            Edit annotations
          </button>
          <button
            onClick={onClose}
            className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
          >
            Back to overview
          </button>
        </div>
      </div>
    </div>
  );
}
