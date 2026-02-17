import { useState } from 'react';
import { PanelMode } from '../types';
import { mockCameras } from '../mockData';
import CctvGrid from './CctvGrid';
import ZoomPanel from './ZoomPanel';
import EditAnnotationsPanel from './EditAnnotationsPanel';

export default function MainControlTab() {
  const [selectedCameraId, setSelectedCameraId] = useState<string | null>(null);
  const [panelMode, setPanelMode] = useState<PanelMode>('none');

  const selectedCamera = mockCameras.find((cam) => cam.id === selectedCameraId);

  const handleZoom = (cameraId: string) => {
    setSelectedCameraId(cameraId);
    setPanelMode('zoom');
  };

  const handleEdit = (cameraId: string) => {
    setSelectedCameraId(cameraId);
    setPanelMode('edit');
  };

  const handleClosePanel = () => {
    setPanelMode('none');
    setSelectedCameraId(null);
  };

  const handleSaveAnnotations = () => {
    setPanelMode('zoom');
  };

  return (
    <div>
      <CctvGrid
        cameras={mockCameras}
        selectedCameraId={selectedCameraId}
        onZoom={handleZoom}
        onEdit={handleEdit}
      />

      {panelMode === 'none' && (
        <div className="mt-6 bg-gray-800 rounded-lg p-8 text-center border border-gray-700">
          <p className="text-gray-400">Select a camera to zoom or edit annotations</p>
        </div>
      )}

      {panelMode === 'zoom' && selectedCamera && (
        <ZoomPanel
          camera={selectedCamera}
          onClose={handleClosePanel}
          onEdit={() => handleEdit(selectedCamera.id)}
        />
      )}

      {panelMode === 'edit' && selectedCamera && (
        <EditAnnotationsPanel
          camera={selectedCamera}
          onSave={handleSaveAnnotations}
          onCancel={handleClosePanel}
        />
      )}
    </div>
  );
}
