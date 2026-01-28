import { Camera } from '../types';
import { ZoomIn, Pencil } from 'lucide-react';

interface CctvTileProps {
  camera: Camera;
  isSelected: boolean;
  onZoom: () => void;
  onEdit: () => void;
}

const severityColors = {
  normal: 'border-gray-600',
  low: 'border-green-500',
  medium: 'border-yellow-500',
  high: 'border-red-500'
};

const severityGlow = {
  normal: '',
  low: 'shadow-green-500/50',
  medium: 'shadow-yellow-500/50',
  high: 'shadow-red-500/50'
};

export default function CctvTile({ camera, isSelected, onZoom, onEdit }: CctvTileProps) {
  return (
    <div
      className={`relative bg-gray-800 rounded-lg overflow-hidden border-2 transition-all ${
        severityColors[camera.severity]
      } ${isSelected ? `shadow-lg ${severityGlow[camera.severity]} ring-2 ring-cyan-400` : 'shadow'}`}
    >
      <div className="absolute top-2 left-2 z-10 bg-black/70 px-2 py-1 rounded text-xs font-medium">
        {camera.name}
      </div>

      <div className="absolute top-2 right-2 z-10 flex gap-2">
        <button
          onClick={onZoom}
          className="bg-black/70 hover:bg-cyan-600 p-2 rounded transition-all hover:scale-110"
          title="Zoom"
        >
          <ZoomIn size={18} />
        </button>
        <button
          onClick={onEdit}
          className="bg-black/70 hover:bg-cyan-600 p-2 rounded transition-all hover:scale-110"
          title="Edit annotations"
        >
          <Pencil size={18} />
        </button>
      </div>

      <div className="aspect-video bg-gray-700 flex items-center justify-center text-gray-500">
        <div className="text-center">
          <div className="text-4xl mb-2">📹</div>
          <div className="text-sm">{camera.name}</div>
          <div className="text-xs text-gray-600 mt-1">Live Feed</div>
        </div>
      </div>

      {camera.severity !== 'normal' && (
        <div className={`absolute bottom-2 left-2 px-2 py-1 rounded text-xs font-bold ${
          camera.severity === 'high' ? 'bg-red-500/90' :
          camera.severity === 'medium' ? 'bg-yellow-500/90 text-gray-900' :
          'bg-green-500/90 text-gray-900'
        }`}>
          {camera.severity.toUpperCase()}
        </div>
      )}
    </div>
  );
}
