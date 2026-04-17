import { useState, useCallback } from 'react';
import { ArrowLeft, Pencil, Navigation, Loader2, CheckCircle, AlertCircle, AlertTriangle, Users } from 'lucide-react';
import { PolygonCanvas } from './PolygonCanvas';
import type { Feed, NEDCoordinate } from '../types';
import { useDetectionStatus } from '../hooks/useDetectionStatus';

// Drone control API URL
const DRONE_API_URL = 'http://localhost:8000';

interface ExpandedFeedViewProps {
  feed: Feed;
  onBack: () => void;
  onEdit: () => void;
}

type DeployStatus = 'idle' | 'deploying' | 'success' | 'error';

export function ExpandedFeedView({ feed, onBack, onEdit }: ExpandedFeedViewProps) {
  const detectionStatus = useDetectionStatus(feed.id, feed.isLive);
  const [deployStatus, setDeployStatus] = useState<DeployStatus>('idle');
  const [errorMessage, setErrorMessage] = useState<string>('');

  // Get target coordinates from detection status (only for RED zone alarms)
  const targetCoordinates: NEDCoordinate | null = detectionStatus?.target_coordinates || null;

  const handleDeploy = useCallback(async () => {
    if (!targetCoordinates) {
      setErrorMessage('No target coordinates available');
      setDeployStatus('error');
      setTimeout(() => setDeployStatus('idle'), 3000);
      return;
    }

    setDeployStatus('deploying');
    setErrorMessage('');

    try {
      console.log('[DEPLOY] Setting drone to automatic mode...');

      // Step 1: Set mode to automatic
      const modeResponse = await fetch(`${DRONE_API_URL}/mode`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: 'automatic' }),
      });

      if (!modeResponse.ok) {
        throw new Error('Failed to set automatic mode');
      }

      console.log('[DEPLOY] Sending drone to:', targetCoordinates);

      // Step 2: Send goto command
      const gotoResponse = await fetch(`${DRONE_API_URL}/goto`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          x: targetCoordinates.x,
          y: targetCoordinates.y,
          z: targetCoordinates.z,
        }),
      });

      if (!gotoResponse.ok) {
        const errorData = await gotoResponse.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Failed to deploy drone');
      }

      setDeployStatus('success');
      setTimeout(() => setDeployStatus('idle'), 3000);
    } catch (error) {
      console.error('[DEPLOY] Error:', error);
      setDeployStatus('error');
      setErrorMessage(error instanceof Error ? error.message : 'Deployment failed');
      setTimeout(() => setDeployStatus('idle'), 5000);
    }
  }, [targetCoordinates]);

  const isAlarmActive = detectionStatus?.alarm_active || false;
  const isCautionActive = detectionStatus?.caution_active || false;
  const hasTarget = targetCoordinates !== null;

  // Determine colors based on state (Alarm takes priority over Caution)
  const getBorderColor = () => {
    if (isAlarmActive) return 'border-[var(--zone-red)]';
    if (isCautionActive) return 'border-[var(--zone-yellow)]';
    return 'border-[var(--border-dim)]';
  };

  const getStatusText = () => {
    if (isAlarmActive) return 'ALARM ACTIVE';
    if (isCautionActive) return 'CAUTION';
    return 'LIVE FEED';
  };

  const getStatusColor = () => {
    if (isAlarmActive) return 'text-[var(--zone-red)]';
    if (isCautionActive) return 'text-[var(--zone-yellow)]';
    return 'text-[var(--text-secondary)]';
  };

  const getIndicatorColor = () => {
    if (isAlarmActive) return 'bg-[var(--zone-red)]';
    if (isCautionActive) return 'bg-[var(--zone-yellow)]';
    return 'bg-[var(--accent-cyan)]';
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

        <div className="flex items-center gap-4">
          {/* Detection Stats */}
          {detectionStatus && (
            <div className="flex items-center gap-4 px-3 py-1.5 rounded border border-[var(--border-dim)] bg-[var(--bg-tertiary)]">
              <div className="flex items-center gap-2">
                <Users size={14} className="text-[var(--accent-cyan)]" />
                <span className="text-xs font-mono text-[var(--text-secondary)]">
                  {detectionStatus.people_count}
                </span>
              </div>
              {isAlarmActive && (
                <div className="flex items-center gap-2">
                  <AlertTriangle size={14} className="text-[var(--zone-red)]" />
                  <span className="text-xs font-mono text-[var(--zone-red)]">
                    {detectionStatus.danger_count} IN RED ZONE
                  </span>
                </div>
              )}
              {isCautionActive && !isAlarmActive && (
                <div className="flex items-center gap-2">
                  <AlertCircle size={14} className="text-[var(--zone-yellow)]" />
                  <span className="text-xs font-mono text-[var(--zone-yellow)]">
                    {detectionStatus.caution_count} IN YELLOW ZONE
                  </span>
                </div>
              )}
            </div>
          )}

          <button
            onClick={onEdit}
            className="flex items-center gap-2 px-3 py-2 rounded border border-[var(--border-dim)] bg-[var(--bg-tertiary)] text-[var(--text-secondary)] hover:text-[var(--accent-cyan)] hover:border-[var(--accent-cyan-dim)] transition-all"
          >
            <Pencil size={14} />
            <span className="text-sm font-medium">Edit Zones</span>
          </button>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 p-6 flex flex-col gap-4 overflow-hidden">
        {/* Feed Canvas - Takes most of the space */}
        <div className={`flex-1 relative rounded-lg overflow-hidden border ${getBorderColor()} corner-brackets min-h-0`}>
          <div className="absolute inset-0 flex items-center justify-center bg-[var(--bg-card)]">
            <PolygonCanvas
              imageSrc={feed.imageSrc}
              zones={feed.zones}
              onZonesChange={() => {}}
              activeTool={null}
              readOnly
            />
          </div>

          {/* Alarm overlay (RED) */}
          {isAlarmActive && (
            <div className="absolute inset-0 pointer-events-none border-4 border-[var(--zone-red)] animate-pulse" />
          )}

          {/* Caution overlay (YELLOW) */}
          {isCautionActive && !isAlarmActive && (
            <div className="absolute inset-0 pointer-events-none border-4 border-[var(--zone-yellow)]" />
          )}

          {/* Live indicator */}
          <div className={`absolute top-3 left-3 flex items-center gap-2 px-2 py-1 rounded bg-[var(--bg-primary)]/80 border ${getBorderColor()}`}>
            <div className={`w-2 h-2 rounded-full ${getIndicatorColor()} ${isAlarmActive ? 'animate-pulse' : 'status-live'}`} />
            <span className={`text-[10px] font-mono tracking-wider ${getStatusColor()}`}>
              {getStatusText()}
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

          {/* Target indicator (only for RED zone alarms) */}
          {hasTarget && (
            <div className="absolute bottom-3 left-3 px-3 py-2 rounded bg-[var(--bg-primary)]/90 border border-[var(--zone-red)]">
              <div className="flex items-center gap-2">
                <Navigation size={14} className="text-[var(--zone-red)]" />
                <span className="text-xs font-mono text-[var(--zone-red)]">
                  TARGET: ({targetCoordinates!.x.toFixed(1)}, {targetCoordinates!.y.toFixed(1)}, {targetCoordinates!.z.toFixed(1)})
                </span>
              </div>
            </div>
          )}

          {/* Caution notice (yellow zone - no drone) */}
          {isCautionActive && !isAlarmActive && (
            <div className="absolute bottom-3 left-3 px-3 py-2 rounded bg-[var(--bg-primary)]/90 border border-[var(--zone-yellow)]">
              <div className="flex items-center gap-2">
                <AlertCircle size={14} className="text-[var(--zone-yellow)]" />
                <span className="text-xs font-mono text-[var(--zone-yellow)]">
                  CAUTION ZONE - Monitor only (no drone deployment)
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Deploy Control Bar */}
        <div className={`flex items-center justify-between px-4 py-3 rounded-lg border ${getBorderColor()} bg-[var(--bg-secondary)]`}>
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-2">
              <Navigation size={18} className={isAlarmActive ? 'text-[var(--zone-red)]' : isCautionActive ? 'text-[var(--zone-yellow)]' : 'text-[var(--accent-cyan)]'} />
              <span className="text-sm font-semibold text-[var(--text-secondary)] uppercase tracking-wider">
                Deploy Drone
              </span>
            </div>

            {/* Auto-calculated Coordinates Display (only for RED zone alarms) */}
            <div className="flex items-center gap-4">
              {hasTarget ? (
                <>
                  <div className="flex items-center gap-2">
                    <label className="text-xs font-mono text-[var(--text-muted)]">X (North)</label>
                    <div className="px-2 py-1 rounded bg-[var(--bg-tertiary)] border border-[var(--border-dim)]">
                      <span className="text-sm font-mono text-[var(--zone-red)]">
                        {targetCoordinates!.x.toFixed(2)}
                      </span>
                    </div>
                    <span className="text-xs text-[var(--text-muted)]">m</span>
                  </div>

                  <div className="flex items-center gap-2">
                    <label className="text-xs font-mono text-[var(--text-muted)]">Y (East)</label>
                    <div className="px-2 py-1 rounded bg-[var(--bg-tertiary)] border border-[var(--border-dim)]">
                      <span className="text-sm font-mono text-[var(--zone-red)]">
                        {targetCoordinates!.y.toFixed(2)}
                      </span>
                    </div>
                    <span className="text-xs text-[var(--text-muted)]">m</span>
                  </div>

                  <div className="flex items-center gap-2">
                    <label className="text-xs font-mono text-[var(--text-muted)]">Z (Alt)</label>
                    <div className="px-2 py-1 rounded bg-[var(--bg-tertiary)] border border-[var(--border-dim)]">
                      <span className="text-sm font-mono text-[var(--zone-red)]">
                        {targetCoordinates!.z.toFixed(2)}
                      </span>
                    </div>
                    <span className="text-xs text-[var(--text-muted)]">m</span>
                  </div>
                </>
              ) : isCautionActive ? (
                <span className="text-xs font-mono text-[var(--zone-yellow)]">
                  Yellow zone - drone deployment not available
                </span>
              ) : (
                <span className="text-xs font-mono text-[var(--text-muted)]">
                  {isAlarmActive ? 'Calculating target...' : 'No target detected'}
                </span>
              )}
            </div>

            <span className="text-[10px] font-mono text-[var(--text-muted)]">
              {hasTarget
                ? 'Auto-calculated from RED zone detection'
                : isAlarmActive
                  ? 'Calculating coordinates...'
                  : 'Requires active RED zone alarm to deploy'}
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
              disabled={deployStatus === 'deploying' || !hasTarget}
              className={`btn-tactical-filled flex items-center gap-2 ${
                !hasTarget ? 'opacity-50 cursor-not-allowed' : ''
              } ${isAlarmActive && hasTarget ? 'bg-[var(--zone-red)] border-[var(--zone-red)] animate-pulse' : ''}`}
              title={!hasTarget ? 'Deploy is only available when a person is detected in a RED zone' : 'Deploy drone to detected coordinates'}
            >
              {deployStatus === 'deploying' ? (
                <>
                  <Loader2 size={16} className="animate-spin" />
                  <span>Deploying...</span>
                </>
              ) : (
                <>
                  <Navigation size={16} />
                  <span>{!hasTarget ? 'No Target' : 'Deploy'}</span>
                </>
              )}
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}
