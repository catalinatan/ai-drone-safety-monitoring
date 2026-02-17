import { useRef, useState, useEffect, useCallback } from 'react';
import type { Zone, ZoneLevel, Point } from '../types';

interface PolygonCanvasProps {
  imageSrc: string;
  zones: Zone[];
  onZonesChange: (zones: Zone[]) => void;
  activeTool: ZoneLevel | 'delete' | null;
  readOnly?: boolean;
}

const ZONE_COLORS: Record<ZoneLevel, { stroke: string; fill: string }> = {
  red: { stroke: '#ff3b5c', fill: 'rgba(255, 59, 92, 0.25)' },
  yellow: { stroke: '#ffc107', fill: 'rgba(255, 193, 7, 0.25)' },
  green: { stroke: '#00e676', fill: 'rgba(0, 230, 118, 0.25)' },
};

export function PolygonCanvas({
  imageSrc,
  zones,
  onZonesChange,
  activeTool,
  readOnly = false,
}: PolygonCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const [currentPoints, setCurrentPoints] = useState<Point[]>([]);
  const [selectedZoneId, setSelectedZoneId] = useState<string | null>(null);
  const [imageLoaded, setImageLoaded] = useState(false);

  // Handle image load to trigger re-render
  const handleImageLoad = useCallback(() => {
    setImageLoaded(true);
  }, []);

  // Get relative coordinates from mouse event using the image's bounding rect
  const getRelativeCoords = useCallback(
    (e: React.MouseEvent): Point | null => {
      const img = imageRef.current;
      if (!img || !imageLoaded) return null;

      // Use the image's bounding rect directly for accurate coordinates
      const rect = img.getBoundingClientRect();
      const x = ((e.clientX - rect.left) / rect.width) * 100;
      const y = ((e.clientY - rect.top) / rect.height) * 100;

      return { x: Math.max(0, Math.min(100, x)), y: Math.max(0, Math.min(100, y)) };
    },
    [imageLoaded]
  );

  // Handle canvas click
  const handleClick = useCallback(
    (e: React.MouseEvent) => {
      if (readOnly) return;

      const coords = getRelativeCoords(e);
      if (!coords) return;

      // Delete tool: check if clicking on a zone
      if (activeTool === 'delete') {
        const clickedZone = zones.find((zone) => isPointInPolygon(coords, zone.points));
        if (clickedZone) {
          onZonesChange(zones.filter((z) => z.id !== clickedZone.id));
          setSelectedZoneId(null);
        }
        return;
      }

      // No drawing tool selected
      if (!activeTool) {
        // Check if clicking on existing zone to select it
        const clickedZone = zones.find((zone) => isPointInPolygon(coords, zone.points));
        setSelectedZoneId(clickedZone?.id ?? null);
        return;
      }

      // Drawing mode: add point
      const newPoints = [...currentPoints, coords];

      // Check if closing the polygon (clicking near first point)
      if (newPoints.length >= 3) {
        const firstPoint = newPoints[0];
        const distance = Math.sqrt(
          Math.pow(coords.x - firstPoint.x, 2) + Math.pow(coords.y - firstPoint.y, 2)
        );

        if (distance < 3) {
          // Close polygon - remove the last point (which is near the first)
          const closedPoints = newPoints.slice(0, -1);
          const newZone: Zone = {
            id: `zone-${Date.now()}`,
            level: activeTool,
            points: closedPoints,
          };
          onZonesChange([...zones, newZone]);
          setCurrentPoints([]);
          return;
        }
      }

      setCurrentPoints(newPoints);
    },
    [activeTool, currentPoints, zones, onZonesChange, readOnly, getRelativeCoords]
  );

  // Handle double-click to close polygon
  const handleDoubleClick = useCallback(() => {
    if (readOnly || !activeTool || activeTool === 'delete' || currentPoints.length < 3) return;

    const newZone: Zone = {
      id: `zone-${Date.now()}`,
      level: activeTool,
      points: currentPoints,
    };
    onZonesChange([...zones, newZone]);
    setCurrentPoints([]);
  }, [activeTool, currentPoints, zones, onZonesChange, readOnly]);

  // Handle escape key to cancel current drawing
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setCurrentPoints([]);
        setSelectedZoneId(null);
      }
      if (e.key === 'Delete' || e.key === 'Backspace') {
        if (selectedZoneId) {
          onZonesChange(zones.filter((z) => z.id !== selectedZoneId));
          setSelectedZoneId(null);
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selectedZoneId, zones, onZonesChange]);

  // Reset current points when tool changes
  useEffect(() => {
    setCurrentPoints([]);
  }, [activeTool]);

  return (
    <div
      ref={containerRef}
      className="relative overflow-hidden cursor-crosshair w-full h-full flex items-center justify-center bg-[var(--bg-primary)]"
      onClick={handleClick}
      onDoubleClick={handleDoubleClick}
    >
      {/* Inner wrapper sizes to the image so SVG overlay aligns exactly */}
      <div className="relative max-w-full max-h-full inline-block">
        {/* Background Image */}
        <img
          ref={imageRef}
          src={imageSrc}
          alt="Feed"
          className="block max-w-full max-h-full"
          onLoad={handleImageLoad}
          draggable={false}
        />

        {/* SVG Overlay for zones */}
        {imageLoaded && (
          <svg
            className="absolute inset-0 w-full h-full pointer-events-none"
            viewBox="0 0 100 100"
            preserveAspectRatio="none"
          >
          {/* Completed zones */}
          {zones.map((zone) => {
            const colors = ZONE_COLORS[zone.level];
            const isSelected = zone.id === selectedZoneId;
            const pointsString = zone.points.map((p) => `${p.x},${p.y}`).join(' ');

            return (
              <g key={zone.id}>
                <polygon
                  points={pointsString}
                  fill={colors.fill}
                  stroke={colors.stroke}
                  strokeWidth={isSelected ? 0.5 : 0.3}
                  strokeDasharray={isSelected ? '1,0.5' : 'none'}
                  className="pointer-events-auto cursor-pointer"
                  style={{
                    filter: isSelected ? `drop-shadow(0 0 3px ${colors.stroke})` : 'none',
                  }}
                />
                {/* Zone vertices */}
                {!readOnly &&
                  zone.points.map((point, idx) => (
                    <circle
                      key={idx}
                      cx={point.x}
                      cy={point.y}
                      r={0.6}
                      fill={colors.stroke}
                      className="pointer-events-none"
                    />
                  ))}
              </g>
            );
          })}

          {/* Current drawing polygon */}
          {currentPoints.length > 0 && activeTool && activeTool !== 'delete' && (
            <g>
              {/* Lines */}
              <polyline
                points={currentPoints.map((p) => `${p.x},${p.y}`).join(' ')}
                fill="none"
                stroke={ZONE_COLORS[activeTool].stroke}
                strokeWidth={0.3}
                strokeDasharray="1,0.5"
              />
              {/* Preview line to first point if we have 3+ points */}
              {currentPoints.length >= 3 && (
                <line
                  x1={currentPoints[currentPoints.length - 1].x}
                  y1={currentPoints[currentPoints.length - 1].y}
                  x2={currentPoints[0].x}
                  y2={currentPoints[0].y}
                  stroke={ZONE_COLORS[activeTool].stroke}
                  strokeWidth={0.2}
                  strokeDasharray="0.5,0.5"
                  opacity={0.5}
                />
              )}
              {/* Points */}
              {currentPoints.map((point, idx) => (
                <circle
                  key={idx}
                  cx={point.x}
                  cy={point.y}
                  r={idx === 0 ? 1 : 0.6}
                  fill={ZONE_COLORS[activeTool].stroke}
                  className={idx === 0 ? 'cursor-pointer' : ''}
                />
              ))}
            </g>
          )}
        </svg>
      )}

        {/* Scanline overlay */}
        <div className="absolute inset-0 scanlines pointer-events-none" />

        {/* Drawing instructions */}
        {!readOnly && activeTool && activeTool !== 'delete' && currentPoints.length === 0 && (
          <div className="absolute bottom-3 left-3 px-2 py-1 rounded text-xs font-mono bg-[var(--bg-primary)]/80 border border-[var(--border-dim)] text-[var(--text-secondary)]">
            Click to place points • Double-click or click first point to close
          </div>
        )}

        {currentPoints.length > 0 && (
          <div className="absolute bottom-3 left-3 px-2 py-1 rounded text-xs font-mono bg-[var(--bg-primary)]/80 border border-[var(--border-dim)] text-[var(--accent-cyan)]">
            Points: {currentPoints.length} • ESC to cancel
          </div>
        )}
      </div>
    </div>
  );
}

// Helper function to check if point is inside polygon (ray casting algorithm)
function isPointInPolygon(point: Point, polygon: Point[]): boolean {
  let inside = false;
  const n = polygon.length;

  for (let i = 0, j = n - 1; i < n; j = i++) {
    const xi = polygon[i].x,
      yi = polygon[i].y;
    const xj = polygon[j].x,
      yj = polygon[j].y;

    if (yi > point.y !== yj > point.y && point.x < ((xj - xi) * (point.y - yi)) / (yj - yi) + xi) {
      inside = !inside;
    }
  }

  return inside;
}
