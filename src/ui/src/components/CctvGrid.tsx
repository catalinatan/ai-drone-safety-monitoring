import { Camera } from '../types';
import CctvTile from './CctvTile';

interface CctvGridProps {
  cameras: Camera[];
  selectedCameraId: string | null;
  onZoom: (cameraId: string) => void;
  onEdit: (cameraId: string) => void;
}

export default function CctvGrid({ cameras, selectedCameraId, onZoom, onEdit }: CctvGridProps) {
  return (
    <div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        {cameras.map((camera) => (
          <CctvTile
            key={camera.id}
            camera={camera}
            isSelected={selectedCameraId === camera.id}
            onZoom={() => onZoom(camera.id)}
            onEdit={() => onEdit(camera.id)}
          />
        ))}
      </div>

      <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <div className="text-sm font-medium mb-2">Severity Legend:</div>
        <div className="flex flex-wrap gap-4 text-xs">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 border-2 border-gray-600 rounded"></div>
            <span>Normal</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 border-2 border-green-500 rounded"></div>
            <span>Low Hazard</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 border-2 border-yellow-500 rounded"></div>
            <span>Medium Hazard</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 border-2 border-red-500 rounded"></div>
            <span>High Hazard</span>
          </div>
        </div>
      </div>
    </div>
  );
}
