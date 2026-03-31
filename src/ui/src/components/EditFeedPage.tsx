import { useState, useCallback } from 'react';
import { ArrowLeft, Circle, Trash2, Scissors, Maximize } from 'lucide-react';
import polygonClipping from 'polygon-clipping';
import { PolygonCanvas } from './PolygonCanvas';
import type { Feed, Zone, ZoneLevel, Point } from '../types';

interface EditFeedPageProps {
  feed: Feed;
  onSave: (zones: Zone[]) => void;
  onCancel: () => void;
}

export type ToolType = ZoneLevel | 'delete' | 'cut' | null;

// Convert Zone points to polygon-clipping format
function zoneToPoly(zone: Zone): [number, number][] {
  return zone.points.map((p) => [p.x, p.y]);
}

// Convert polygon-clipping result ring to Zone points
function ringToPoints(ring: [number, number][]): Point[] {
  return ring.map(([x, y]) => ({ x, y }));
}

export function EditFeedPage({ feed, onSave, onCancel }: EditFeedPageProps) {
  const [zones, setZones] = useState<Zone[]>(feed.zones);
  const [activeTool, setActiveTool] = useState<ToolType>(null);

  const handleToolClick = (tool: ToolType) => {
    setActiveTool((current) => (current === tool ? null : tool));
  };

  const handleSave = () => {
    onSave(zones);
  };

  // Fill Remaining: create a zone covering all uncovered area
  const handleFillRemaining = useCallback(() => {
    // Need a drawing level selected (red/yellow/green)
    const level = activeTool === 'red' || activeTool === 'yellow' || activeTool === 'green'
      ? activeTool : 'green';

    const fullRect: [number, number][] = [[0, 0], [100, 0], [100, 100], [0, 100]];

    if (zones.length === 0) {
      // No zones exist — fill the entire frame
      setZones([...zones, {
        id: `zone-${Date.now()}`,
        level,
        points: ringToPoints(fullRect),
        source: 'manual',
      }]);
      return;
    }

    // Subtract all existing zones from the full rectangle
    const existingPolys = zones.map((z) => [zoneToPoly(z)] as [number, number][][]);
    try {
      const result = polygonClipping.difference([fullRect], ...existingPolys);
      if (!result || result.length === 0) return; // fully covered

      const newZones: Zone[] = result.map((poly, i) => ({
        id: `zone-fill-${Date.now()}-${i}`,
        level,
        points: ringToPoints(poly[0]), // outer ring
        source: 'manual' as const,
      }));
      setZones([...zones, ...newZones]);
    } catch {
      // Fallback: just create full rectangle
      setZones([...zones, {
        id: `zone-${Date.now()}`,
        level,
        points: ringToPoints(fullRect),
        source: 'manual',
      }]);
    }
  }, [zones, activeTool]);

  // Cut handler: subtract a drawn polygon from the zone it overlaps
  const handleCutComplete = useCallback((cutPoints: Point[]) => {
    if (cutPoints.length < 3) return;
    const cutPoly: [number, number][] = cutPoints.map((p) => [p.x, p.y]);

    // Find the zone whose polygon contains the centroid of the cut shape
    const cx = cutPoints.reduce((s, p) => s + p.x, 0) / cutPoints.length;
    const cy = cutPoints.reduce((s, p) => s + p.y, 0) / cutPoints.length;

    const targetIdx = zones.findIndex((z) => isPointInPolygon({ x: cx, y: cy }, z.points));
    if (targetIdx === -1) return;

    const target = zones[targetIdx];
    const targetPoly: [number, number][] = zoneToPoly(target);

    try {
      const result = polygonClipping.difference([targetPoly], [cutPoly]);
      if (!result || result.length === 0) {
        // Zone fully erased
        setZones(zones.filter((_, i) => i !== targetIdx));
        return;
      }
      // Replace target with result polygon(s)
      const replacements: Zone[] = result.map((poly, i) => ({
        id: i === 0 ? target.id : `zone-cut-${Date.now()}-${i}`,
        level: target.level,
        points: ringToPoints(poly[0]),
        source: target.source,
      }));
      const updated = [...zones];
      updated.splice(targetIdx, 1, ...replacements);
      setZones(updated);
    } catch {
      // polygon-clipping failure — ignore
    }
  }, [zones]);

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
        {feed.sceneType && (
          <div className="flex items-center gap-2 px-4 py-2 rounded border border-[var(--border-dim)] bg-[var(--bg-tertiary)]">
            <span className="text-xs font-mono text-[var(--text-secondary)]">
              {feed.sceneType.charAt(0).toUpperCase() + feed.sceneType.slice(1)} feed — Drag zone vertices to adjust. Saved edits pause auto-segmentation.
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
            onCutComplete={handleCutComplete}
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

              {/* Cut Tool */}
              <button
                onClick={() => handleToolClick('cut')}
                className={`zone-tool border-[var(--accent-cyan)] text-[var(--accent-cyan)] ${
                  activeTool === 'cut' ? 'active bg-[var(--accent-cyan)]/20' : ''
                } hover:bg-[var(--accent-cyan)]/10`}
                title="Cut — Draw shape to remove from a zone"
              >
                <Scissors size={18} />
              </button>

              <div className="h-6 w-px bg-[var(--border-dim)] mx-2" />

              {/* Fill Remaining */}
              <button
                onClick={handleFillRemaining}
                className="zone-tool border-[var(--accent-cyan)] text-[var(--accent-cyan)] hover:bg-[var(--accent-cyan)]/10"
                title="Fill all uncovered area with selected zone level"
              >
                <Maximize size={18} />
              </button>
            </div>

            {/* Tool descriptions */}
            <div className="text-xs text-[var(--text-muted)] font-mono">
              {activeTool === 'red' && 'RESTRICTED — No entry allowed'}
              {activeTool === 'yellow' && 'CAUTION — Limited access'}
              {activeTool === 'green' && 'SAFE — Normal access'}
              {activeTool === 'delete' && 'DELETE — Click on a zone to remove'}
              {activeTool === 'cut' && 'CUT — Draw shape to subtract from a zone'}
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

// Ray casting point-in-polygon check
function isPointInPolygon(point: Point, polygon: Point[]): boolean {
  let inside = false;
  const n = polygon.length;
  for (let i = 0, j = n - 1; i < n; j = i++) {
    const xi = polygon[i].x, yi = polygon[i].y;
    const xj = polygon[j].x, yj = polygon[j].y;
    if (yi > point.y !== yj > point.y && point.x < ((xj - xi) * (point.y - yi)) / (yj - yi) + xi) {
      inside = !inside;
    }
  }
  return inside;
}
