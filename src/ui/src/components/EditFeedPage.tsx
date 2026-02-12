import { useState } from 'react';
import { ArrowLeft, Circle, Trash2, Scan, Loader2 } from 'lucide-react';
import { PolygonCanvas } from './PolygonCanvas';
import type { Feed, Zone, ZoneLevel } from '../types';

interface EditFeedPageProps {
  feed: Feed;
  onSave: (zones: Zone[]) => void;
  onCancel: () => void;
  onAutoSegment?: () => Promise<Zone[] | null>;
}

type ToolType = ZoneLevel | 'delete' | null;

export function EditFeedPage({ feed, onSave, onCancel, onAutoSegment }: EditFeedPageProps) {
  const [zones, setZones] = useState<Zone[]>(feed.zones);
  const [activeTool, setActiveTool] = useState<ToolType>(null);
  const [isAutoSegmenting, setIsAutoSegmenting] = useState(false);
  const [autoSegFailed, setAutoSegFailed] = useState(false);

  const handleAutoSegment = async () => {
    if (!onAutoSegment) return;
    setIsAutoSegmenting(true);
    setAutoSegFailed(false);
    try {
      const newZones = await onAutoSegment();
      if (newZones) {
        setZones(newZones);
      } else {
        setAutoSegFailed(true);
        setTimeout(() => setAutoSegFailed(false), 2000);
      }
    } finally {
      setIsAutoSegmenting(false);
    }
  };

  const handleToolClick = (tool: ToolType) => {
    setActiveTool((current) => (current === tool ? null : tool));
  };

  const handleSave = () => {
    onSave(zones);
  };

  const zoneStats = {
    red: zones.filter((z) => z.level === 'red').length,
    yellow: zones.filter((z) => z.level === 'yellow').length,
    green: zones.filter((z) => z.level === 'green').length,
  };

  return (
    <div className="h-full flex flex-col bg-[var(--bg-primary)] tactical-grid">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-4 border-b border-[var(--border-dim)] bg-[var(--bg-secondary)]/80 backdrop-blur-sm">
        <div className="flex items-center gap-4">
          <button
            onClick={onCancel}
            className="flex items-center gap-2 text-[var(--text-secondary)] hover:text-[var(--accent-cyan)] transition-colors"
          >
            <ArrowLeft size={18} />
            <span className="text-sm font-medium">Back</span>
          </button>
          <div className="h-4 w-px bg-[var(--border-dim)]" />
          <div>
            <h1 className="text-lg font-bold tracking-wide uppercase text-glow-cyan">
              Edit Zones
            </h1>
            <span className="text-xs font-mono text-[var(--text-muted)]">
              {feed.name} — {feed.location}
            </span>
          </div>
        </div>

        {/* Zone Stats */}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-3 px-3 py-1.5 rounded border border-[var(--border-dim)] bg-[var(--bg-tertiary)]">
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 rounded-full bg-[var(--zone-red)]" />
              <span className="text-xs font-mono text-[var(--text-secondary)]">{zoneStats.red}</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 rounded-full bg-[var(--zone-yellow)]" />
              <span className="text-xs font-mono text-[var(--text-secondary)]">{zoneStats.yellow}</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 rounded-full bg-[var(--zone-green)]" />
              <span className="text-xs font-mono text-[var(--text-secondary)]">{zoneStats.green}</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 p-6 flex flex-col gap-4 overflow-hidden">
        {/* Auto-segmentation info banner */}
        {feed.sceneType === 'ship' && (
          <div className="flex items-center gap-2 px-4 py-2 rounded border border-[var(--accent-cyan-dim)] bg-[var(--accent-cyan-glow)]">
            <div className="w-1.5 h-1.5 rounded-full bg-[var(--accent-cyan)] animate-pulse" />
            <span className="text-xs font-mono text-[var(--accent-cyan)]">
              Ship feed — Zones auto-update every 60s. Manual edits will be overwritten.
            </span>
          </div>
        )}
        {(feed.sceneType === 'railway' || feed.sceneType === 'bridge') && (
          <div className="flex items-center gap-2 px-4 py-2 rounded border border-[var(--border-dim)] bg-[var(--bg-tertiary)]">
            <span className="text-xs font-mono text-[var(--text-secondary)]">
              {feed.sceneType.charAt(0).toUpperCase() + feed.sceneType.slice(1)} feed — Auto-segmented on startup. Manual edits will persist.
            </span>
          </div>
        )}

        {/* Canvas Container */}
        <div className="flex-1 min-h-0 rounded-lg overflow-hidden border border-[var(--border-dim)] corner-brackets bg-[var(--bg-card)]">
          <PolygonCanvas
            imageSrc={feed.imageSrc}
            zones={zones}
            onZonesChange={setZones}
            activeTool={activeTool}
          />
        </div>

        {/* Tools Bar */}
        <div className="flex items-center justify-between px-4 py-3 rounded-lg border border-[var(--border-dim)] bg-[var(--bg-secondary)]">
          <div className="flex items-center gap-6">
            <span className="text-sm font-semibold text-[var(--text-secondary)] uppercase tracking-wider">
              Tools
            </span>

            <div className="flex items-center gap-3">
              {/* Red Zone Tool */}
              <button
                onClick={() => handleToolClick('red')}
                className={`zone-tool zone-tool-red ${activeTool === 'red' ? 'active' : ''}`}
                title="Restricted Zone (No Entry)"
              >
                <Circle size={18} fill={activeTool === 'red' ? 'currentColor' : 'none'} />
              </button>

              {/* Yellow Zone Tool */}
              <button
                onClick={() => handleToolClick('yellow')}
                className={`zone-tool zone-tool-yellow ${activeTool === 'yellow' ? 'active' : ''}`}
                title="Caution Zone (Limited Access)"
              >
                <Circle size={18} fill={activeTool === 'yellow' ? 'currentColor' : 'none'} />
              </button>

              {/* Green Zone Tool */}
              <button
                onClick={() => handleToolClick('green')}
                className={`zone-tool zone-tool-green ${activeTool === 'green' ? 'active' : ''}`}
                title="Safe Zone (Normal Access)"
              >
                <Circle size={18} fill={activeTool === 'green' ? 'currentColor' : 'none'} />
              </button>

              <div className="h-6 w-px bg-[var(--border-dim)] mx-2" />

              {/* Delete Tool */}
              <button
                onClick={() => handleToolClick('delete')}
                className={`zone-tool border-[var(--zone-red)] text-[var(--zone-red)] ${
                  activeTool === 'delete' ? 'active bg-[var(--zone-red-fill)]' : ''
                } hover:bg-[var(--zone-red-fill)]`}
                title="Delete Zone"
              >
                <Trash2 size={18} />
              </button>

              {/* Auto Segment Button */}
              {feed.sceneType && onAutoSegment && (
                <>
                  <div className="h-6 w-px bg-[var(--border-dim)] mx-2" />
                  <button
                    onClick={handleAutoSegment}
                    disabled={isAutoSegmenting}
                    className="flex items-center gap-2 px-3 py-1.5 rounded border border-[var(--accent-cyan-dim)] text-[var(--accent-cyan)] hover:bg-[var(--accent-cyan-glow)] transition-all disabled:opacity-50"
                    title={`Auto-detect ${feed.sceneType} hazard zones`}
                  >
                    {isAutoSegmenting ? (
                      <>
                        <Loader2 size={16} className="animate-spin" />
                        <span className="text-xs font-mono">Segmenting...</span>
                      </>
                    ) : autoSegFailed ? (
                      <>
                        <Scan size={16} />
                        <span className="text-xs font-mono text-[var(--zone-red)]">Failed</span>
                      </>
                    ) : (
                      <>
                        <Scan size={16} />
                        <span className="text-xs font-mono">Auto Segment</span>
                      </>
                    )}
                  </button>
                </>
              )}
            </div>

            {/* Tool descriptions */}
            <div className="text-xs text-[var(--text-muted)] font-mono">
              {activeTool === 'red' && 'RESTRICTED — No entry allowed'}
              {activeTool === 'yellow' && 'CAUTION — Limited access'}
              {activeTool === 'green' && 'SAFE — Normal access'}
              {activeTool === 'delete' && 'DELETE — Click on a zone to remove'}
              {!activeTool && 'Select a tool to begin drawing zones'}
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex items-center gap-3">
            <button onClick={onCancel} className="btn-cancel">
              Cancel
            </button>
            <button onClick={handleSave} className="btn-tactical-filled">
              Save Zones
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}
