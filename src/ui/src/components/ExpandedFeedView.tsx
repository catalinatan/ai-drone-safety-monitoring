import { useState } from 'react';
import { ArrowLeft, Pencil, Navigation, Loader2, CheckCircle, AlertCircle } from 'lucide-react';
import { PolygonCanvas } from './PolygonCanvas';
import type { Feed, NEDCoordinate } from '../types';

interface ExpandedFeedViewProps {
  feed: Feed;
  onBack: () => void;
  onEdit: () => void;
}

type DeployStatus = 'idle' | 'deploying' | 'success' | 'error';

export function ExpandedFeedView({ feed, onBack, onEdit }: ExpandedFeedViewProps) {
  const [coordinates, setCoordinates] = useState<NEDCoordinate>({
    x: 0,
    y: 0,
    z: -10,
  });
  const [deployStatus, setDeployStatus] = useState<DeployStatus>('idle');
  const [errorMessage, setErrorMessage] = useState<string>('');

  const handleDeploy = async () => {
    setDeployStatus('deploying');
    setErrorMessage('');

    try {
      // Stubbed API call - will integrate with drone.py later
      // POST to http://localhost:8000/goto
      // First need to set mode to automatic, then send goto
      console.log('[DEPLOY] Sending drone to:', coordinates);

      // Simulate API call
      await new Promise((resolve) => setTimeout(resolve, 1500));

      // Uncomment when integrating:
      // await fetch('http://localhost:8000/mode', {
      //   method: 'POST',
      //   headers: { 'Content-Type': 'application/json' },
      //   body: JSON.stringify({ mode: 'automatic' }),
      // });
      //
      // const response = await fetch('http://localhost:8000/goto', {
      //   method: 'POST',
      //   headers: { 'Content-Type': 'application/json' },
      //   body: JSON.stringify({
      //     x: coordinates.x,
      //     y: coordinates.y,
      //     z: coordinates.z,
      //   }),
      // });
      //
      // if (!response.ok) throw new Error('Failed to deploy drone');

      setDeployStatus('success');
      setTimeout(() => setDeployStatus('idle'), 3000);
    } catch (error) {
      console.error('[DEPLOY] Error:', error);
      setDeployStatus('error');
      setErrorMessage(error instanceof Error ? error.message : 'Deployment failed');
      setTimeout(() => setDeployStatus('idle'), 5000);
    }
  };

  const handleInputChange = (axis: keyof NEDCoordinate, value: string) => {
    const numValue = parseFloat(value) || 0;
    setCoordinates((prev) => ({ ...prev, [axis]: numValue }));
  };

  return (
    <div className="h-full flex flex-col bg-[var(--bg-primary)] tactical-grid">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-4 border-b border-[var(--border-dim)] bg-[var(--bg-secondary)]/80 backdrop-blur-sm">
        <div className="flex items-center gap-4">
          <button
            onClick={onBack}
            className="flex items-center gap-2 text-[var(--text-secondary)] hover:text-[var(--accent-cyan)] transition-colors"
          >
            <ArrowLeft size={18} />
            <span className="text-sm font-medium">Back</span>
          </button>
          <div className="h-4 w-px bg-[var(--border-dim)]" />
          <div>
            <h1 className="text-lg font-bold tracking-wide uppercase text-glow-cyan">
              {feed.name} — Expanded View
            </h1>
            <span className="text-xs font-mono text-[var(--text-muted)]">{feed.location}</span>
          </div>
        </div>

        <button
          onClick={onEdit}
          className="flex items-center gap-2 px-3 py-2 rounded border border-[var(--border-dim)] bg-[var(--bg-tertiary)] text-[var(--text-secondary)] hover:text-[var(--accent-cyan)] hover:border-[var(--accent-cyan-dim)] transition-all"
        >
          <Pencil size={14} />
          <span className="text-sm font-medium">Edit Zones</span>
        </button>
      </header>

      {/* Main Content */}
      <main className="flex-1 p-6 flex flex-col gap-4 overflow-hidden">
        {/* Feed Canvas - Takes most of the space */}
        <div className="flex-1 relative rounded-lg overflow-hidden border border-[var(--border-dim)] corner-brackets min-h-0">
          <div className="absolute inset-0 flex items-center justify-center bg-[var(--bg-card)]">
            <PolygonCanvas
              imageSrc={feed.imageSrc}
              zones={feed.zones}
              onZonesChange={() => {}}
              activeTool={null}
              readOnly
            />
          </div>

          {/* Live indicator */}
          <div className="absolute top-3 left-3 flex items-center gap-2 px-2 py-1 rounded bg-[var(--bg-primary)]/80 border border-[var(--border-dim)]">
            <div className="w-2 h-2 rounded-full bg-[var(--zone-red)] status-live" />
            <span className="text-[10px] font-mono text-[var(--text-secondary)] tracking-wider">
              LIVE FEED
            </span>
          </div>

          {/* Zone count */}
          {feed.zones.length > 0 && (
            <div className="absolute top-3 right-3 px-2 py-1 rounded bg-[var(--bg-primary)]/80 border border-[var(--border-dim)]">
              <span className="text-[10px] font-mono text-[var(--text-secondary)]">
                ZONES ACTIVE:{' '}
                <span className="text-[var(--accent-cyan)]">{feed.zones.length}</span>
              </span>
            </div>
          )}
        </div>

        {/* Deploy Control Bar */}
        <div className="flex items-center justify-between px-4 py-3 rounded-lg border border-[var(--border-dim)] bg-[var(--bg-secondary)]">
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-2">
              <Navigation size={18} className="text-[var(--accent-cyan)]" />
              <span className="text-sm font-semibold text-[var(--text-secondary)] uppercase tracking-wider">
                Deploy Drone
              </span>
            </div>

            {/* Coordinate Inputs */}
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <label className="text-xs font-mono text-[var(--text-muted)]">X (North)</label>
                <input
                  type="number"
                  value={coordinates.x}
                  onChange={(e) => handleInputChange('x', e.target.value)}
                  className="input-tactical w-20 text-center"
                  step="0.5"
                />
                <span className="text-xs text-[var(--text-muted)]">m</span>
              </div>

              <div className="flex items-center gap-2">
                <label className="text-xs font-mono text-[var(--text-muted)]">Y (East)</label>
                <input
                  type="number"
                  value={coordinates.y}
                  onChange={(e) => handleInputChange('y', e.target.value)}
                  className="input-tactical w-20 text-center"
                  step="0.5"
                />
                <span className="text-xs text-[var(--text-muted)]">m</span>
              </div>

              <div className="flex items-center gap-2">
                <label className="text-xs font-mono text-[var(--text-muted)]">Z (Down)</label>
                <input
                  type="number"
                  value={coordinates.z}
                  onChange={(e) => handleInputChange('z', e.target.value)}
                  className="input-tactical w-20 text-center"
                  step="0.5"
                />
                <span className="text-xs text-[var(--text-muted)]">m</span>
              </div>
            </div>

            <span className="text-[10px] font-mono text-[var(--text-muted)]">
              NED: negative Z = above ground
            </span>
          </div>

          {/* Deploy Button */}
          <div className="flex items-center gap-3">
            {deployStatus === 'error' && (
              <div className="flex items-center gap-2 text-[var(--zone-red)]">
                <AlertCircle size={16} />
                <span className="text-xs font-mono">{errorMessage || 'Deploy failed'}</span>
              </div>
            )}

            {deployStatus === 'success' && (
              <div className="flex items-center gap-2 text-[var(--zone-green)]">
                <CheckCircle size={16} />
                <span className="text-xs font-mono">Deployed!</span>
              </div>
            )}

            <button
              onClick={handleDeploy}
              disabled={deployStatus === 'deploying'}
              className="btn-tactical-filled flex items-center gap-2"
            >
              {deployStatus === 'deploying' ? (
                <>
                  <Loader2 size={16} className="animate-spin" />
                  <span>Deploying...</span>
                </>
              ) : (
                <>
                  <Navigation size={16} />
                  <span>Deploy</span>
                </>
              )}
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}
