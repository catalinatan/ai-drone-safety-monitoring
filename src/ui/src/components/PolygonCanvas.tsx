import { useRef, useState, useEffect, useCallback } from 'react';
import type { Zone, ZoneLevel, Point } from '../types';
import type { ToolType } from './EditFeedPage';

interface PolygonCanvasProps {
  imageSrc: string;
  zones: Zone[];
  onZonesChange: (zones: Zone[]) => void;
  activeTool: ToolType;
  readOnly?: boolean;
  onCutComplete?: (cutPoints: Point[]) => void;
}

interface ImgRect {
  top: number;
  left: number;
  width: number;
  height: number;
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
  onCutComplete,
}: PolygonCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const [currentPoints, setCurrentPoints] = useState<Point[]>([]);
  const [selectedZoneId, setSelectedZoneId] = useState<string | null>(null);
  const [imageLoaded, setImageLoaded] = useState(false);
  const [imgRect, setImgRect] = useState<ImgRect | null>(null);
  const [streamAttempt, setStreamAttempt] = useState(0);
  const [, setRetryCount] = useState(0);

  // Vertex drag state
  const [draggingVertex, setDraggingVertex] = useState<{ zoneId: string; pointIdx: number } | null>(null);
  const draggingRef = useRef(draggingVertex);
  draggingRef.current = draggingVertex;

  // Measure the rendered image rect relative to the container
  const updateImgRect = useCallback(() => {
    const img = imageRef.current;
    const container = containerRef.current;
    if (!img || !container) return;

    const containerBounds = container.getBoundingClientRect();
    const containerWidth = containerBounds.width;
    const containerHeight = containerBounds.height;
    const naturalWidth = img.naturalWidth;
    const naturalHeight = img.naturalHeight;

    if (!naturalWidth || !naturalHeight || !containerWidth || !containerHeight) return;

    // Compute the actual rendered image box for object-contain (excluding letterboxing).
    const imageAspect = naturalWidth / naturalHeight;
    const containerAspect = containerWidth / containerHeight;

    let width = containerWidth;
    let height = containerHeight;
    let left = 0;
    let top = 0;

    if (imageAspect > containerAspect) {
      width = containerWidth;
      height = containerWidth / imageAspect;
      top = (containerHeight - height) / 2;
    } else {
      height = containerHeight;
      width = containerHeight * imageAspect;
      left = (containerWidth - width) / 2;
    }

    setImgRect({
      top,
      left,
      width,
      height,
    });
  }, []);

  // Re-measure on window resize
  useEffect(() => {
    if (!imageLoaded) return;
    window.addEventListener('resize', updateImgRect);
    return () => window.removeEventListener('resize', updateImgRect);
  }, [imageLoaded, updateImgRect]);

  // Keep overlay aligned if container size changes without a window resize.
  useEffect(() => {
    if (!imageLoaded || !containerRef.current) return;

    const observer = new ResizeObserver(() => updateImgRect());
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, [imageLoaded, updateImgRect]);

  // Get relative coordinates from mouse event using the rendered image rect.
  const getRelativeCoords = useCallback(
    (e: React.MouseEvent): Point | null => {
      const img = imageRef.current;
      const container = containerRef.current;
      if (!img || !container || !imageLoaded || !imgRect) return null;

      const containerBounds = container.getBoundingClientRect();
      const xPx = e.clientX - containerBounds.left;
      const yPx = e.clientY - containerBounds.top;

      // Ignore clicks in letterbox bars outside the actual rendered frame area.
      if (
        xPx < imgRect.left ||
        xPx > imgRect.left + imgRect.width ||
        yPx < imgRect.top ||
        yPx > imgRect.top + imgRect.height
      ) {
        return null;
      }

      const x = ((xPx - imgRect.left) / imgRect.width) * 100;
      const y = ((yPx - imgRect.top) / imgRect.height) * 100;

      return { x: Math.max(0, Math.min(100, x)), y: Math.max(0, Math.min(100, y)) };
    },
    [imageLoaded, imgRect]
  );

  // Vertex drag: mousedown on a vertex circle
  const handleVertexMouseDown = useCallback(
    (e: React.MouseEvent, zoneId: string, pointIdx: number) => {
      if (readOnly) return;
      e.stopPropagation();
      e.preventDefault();
      setDraggingVertex({ zoneId, pointIdx });
      setSelectedZoneId(zoneId);
    },
    [readOnly]
  );

  // Vertex drag: mousemove
  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (!draggingRef.current) return;
      const coords = getRelativeCoords(e);
      if (!coords) return;

      const { zoneId, pointIdx } = draggingRef.current;
      const updatedZones = zones.map((zone) => {
        if (zone.id !== zoneId) return zone;
        const newPoints = [...zone.points];
        newPoints[pointIdx] = coords;
        return { ...zone, points: newPoints };
      });
      onZonesChange(updatedZones);
    },
    [zones, onZonesChange, getRelativeCoords]
  );

  // Vertex drag: mouseup
  const handleMouseUp = useCallback(() => {
    setDraggingVertex(null);
  }, []);

  // Handle canvas click
  const handleClick = useCallback(
    (e: React.MouseEvent) => {
      if (readOnly) return;
      // Skip click if we were dragging a vertex
      if (draggingRef.current) return;

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

      // Cut tool or drawing mode: add point
      const newPoints = [...currentPoints, coords];

      // Check if closing the polygon (clicking near first point)
      if (newPoints.length >= 3) {
        const firstPoint = newPoints[0];
        const distance = Math.sqrt(
          Math.pow(coords.x - firstPoint.x, 2) + Math.pow(coords.y - firstPoint.y, 2)
        );

        if (distance < 3) {
          const closedPoints = newPoints.slice(0, -1);

          if (activeTool === 'cut') {
            // Cut tool: subtract this polygon from overlapping zone
            onCutComplete?.(closedPoints);
            setCurrentPoints([]);
            return;
          }

          // Normal drawing: create new zone
          const newZone: Zone = {
            id: `zone-${Date.now()}`,
            level: activeTool,
            points: closedPoints,
            source: 'manual',
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

    if (activeTool === 'cut') {
      onCutComplete?.(currentPoints);
      setCurrentPoints([]);
      return;
    }

    const newZone: Zone = {
      id: `zone-${Date.now()}`,
      level: activeTool,
      points: currentPoints,
      source: 'manual',
    };
    onZonesChange([...zones, newZone]);
    setCurrentPoints([]);
  }, [activeTool, currentPoints, zones, onZonesChange, readOnly, onCutComplete]);

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

  // Reset stream state when feed source changes.
  useEffect(() => {
    setImageLoaded(false);
    setImgRect(null);
    setStreamAttempt(0);
    setRetryCount(0);
  }, [imageSrc]);

  const streamUrl = `${imageSrc}${imageSrc.includes('?') ? '&' : '?'}attempt=${streamAttempt}`;

  return (
    <div
      ref={containerRef}
      className={`relative overflow-hidden w-full h-full flex items-center justify-center bg-[var(--bg-primary)] ${draggingVertex ? 'cursor-grabbing' : 'cursor-crosshair'}`}
      onClick={handleClick}
      onDoubleClick={handleDoubleClick}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    >
      {/* Background Image */}
      <img
        ref={imageRef}
        src={streamUrl}
        alt="Feed"
        className="w-full h-full object-contain"
        onLoad={() => {
          setImageLoaded(true);
          updateImgRect();
        }}
        onError={() => {
          setImageLoaded(false);
          setRetryCount((current) => {
            const nextRetry = current + 1;
            const delayMs = Math.min(1000 * Math.pow(2, Math.max(0, nextRetry - 1)), 2000);
            setTimeout(() => setStreamAttempt((n) => n + 1), delayMs);
            return nextRetry;
          });
        }}
        draggable={false}
      />

      {!imageLoaded && (
        <div className="absolute inset-0 flex items-center justify-center bg-[var(--bg-card)]">
          <span className="text-xs font-mono text-[var(--text-muted)]">
            Loading feed...
          </span>
        </div>
      )}

      {/* SVG Overlay for zones — positioned to match the rendered image bounds */}
      {imageLoaded && imgRect && (
        <svg
          className="absolute pointer-events-none"
          style={{
            top: imgRect.top,
            left: imgRect.left,
            width: imgRect.width,
            height: imgRect.height,
          }}
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
                {/* Zone vertices — draggable when not readOnly */}
                {!readOnly &&
                  zone.points.map((point, idx) => (
                    <circle
                      key={idx}
                      cx={point.x}
                      cy={point.y}
                      r={isSelected ? 1 : 0.6}
                      fill={colors.stroke}
                      stroke={isSelected ? '#fff' : 'none'}
                      strokeWidth={0.2}
                      className="pointer-events-auto cursor-grab"
                      onMouseDown={(e) => handleVertexMouseDown(e, zone.id, idx)}
                    />
                  ))}
              </g>
            );
          })}

          {/* Current drawing polygon */}
          {currentPoints.length > 0 && activeTool && activeTool !== 'delete' && (() => {
            const isCut = activeTool === 'cut';
            const strokeColor = isCut ? '#00d4ff' : ZONE_COLORS[activeTool as ZoneLevel].stroke;
            return (
              <g>
                {/* Lines */}
                <polyline
                  points={currentPoints.map((p) => `${p.x},${p.y}`).join(' ')}
                  fill="none"
                  stroke={strokeColor}
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
                    stroke={strokeColor}
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
                    fill={strokeColor}
                    className={idx === 0 ? 'cursor-pointer' : ''}
                  />
                ))}
              </g>
            );
          })()}
        </svg>
      )}

      {/* Scanline overlay */}
      <div className="absolute inset-0 scanlines pointer-events-none" />

      {/* Drawing instructions */}
      {!readOnly && activeTool && activeTool !== 'delete' && currentPoints.length === 0 && (
        <div className="absolute bottom-3 left-3 px-2 py-1 rounded text-xs font-mono bg-[var(--bg-primary)]/80 border border-[var(--border-dim)] text-[var(--text-secondary)]">
          {activeTool === 'cut'
            ? 'Draw shape over a zone to cut • Double-click or click first point to close'
            : 'Click to place points • Double-click or click first point to close • Drag vertices to adjust'}
        </div>
      )}
      {!readOnly && !activeTool && currentPoints.length === 0 && !draggingVertex && (
        <div className="absolute bottom-3 left-3 px-2 py-1 rounded text-xs font-mono bg-[var(--bg-primary)]/80 border border-[var(--border-dim)] text-[var(--text-muted)]">
          Drag zone vertices to adjust • Select a tool to draw new zones
        </div>
      )}

      {currentPoints.length > 0 && (
        <div className="absolute bottom-3 left-3 px-2 py-1 rounded text-xs font-mono bg-[var(--bg-primary)]/80 border border-[var(--border-dim)] text-[var(--accent-cyan)]">
          Points: {currentPoints.length} • ESC to cancel
        </div>
      )}
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
