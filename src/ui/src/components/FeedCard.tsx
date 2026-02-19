import { useState, useEffect, useRef, useCallback } from 'react';
import { Pencil, Maximize2, AlertTriangle, AlertCircle, Users, VideoOff } from 'lucide-react';
import type { Feed } from '../types';
import { useDetectionStatus } from '../hooks/useDetectionStatus';

const ZONE_COLORS: Record<string, { stroke: string; fill: string }> = {
  red: { stroke: '#ff3b5c', fill: 'rgba(255, 59, 92, 0.25)' },
  yellow: { stroke: '#ffc107', fill: 'rgba(255, 193, 7, 0.25)' },
  green: { stroke: '#00e676', fill: 'rgba(0, 230, 118, 0.25)' },
};

interface FeedCardProps {
  feed: Feed;
  onEdit: () => void;
  onExpand: () => void;
  showZones?: boolean;
}

export function FeedCard({ feed, onEdit, onExpand, showZones = false }: FeedCardProps) {
  const detectionStatus = useDetectionStatus(feed.id, feed.isLive);
  const [hasRenderedFrame, setHasRenderedFrame] = useState(false);
  const [streamAttempt, setStreamAttempt] = useState(0);
  const [retryCount, setRetryCount] = useState(0);
  const [isStreamBroken, setIsStreamBroken] = useState(false);
  const imgRef = useRef<HTMLImageElement>(null);
  const [imgRect, setImgRect] = useState<{ left: number; top: number; width: number; height: number } | null>(null);

  // Calculate the rendered image area within the object-contain container
  const updateImgRect = useCallback(() => {
    const img = imgRef.current;
    if (!img || !img.naturalWidth || !img.naturalHeight) return;
    const containerW = img.clientWidth;
    const containerH = img.clientHeight;
    const naturalW = img.naturalWidth;
    const naturalH = img.naturalHeight;
    const scale = Math.min(containerW / naturalW, containerH / naturalH);
    const renderedW = naturalW * scale;
    const renderedH = naturalH * scale;
    const left = (containerW - renderedW) / 2;
    const top = (containerH - renderedH) / 2;
    setImgRect({ left, top, width: renderedW, height: renderedH });
  }, []);

  // Update on resize
  useEffect(() => {
    const img = imgRef.current;
    if (!img) return;
    const observer = new ResizeObserver(updateImgRect);
    observer.observe(img);
    return () => observer.disconnect();
  }, [updateImgRect]);

  useEffect(() => {
    setHasRenderedFrame(false);
    setStreamAttempt(0);
    setRetryCount(0);
    setIsStreamBroken(false);
  }, [feed.id, feed.imageSrc]);

  // Watchdog: if the stream is broken, force a retry every 3s instead of
  // waiting for the exponential backoff to climb.  Also catches silently
  // stalled MJPEG connections that never fire onError.
  useEffect(() => {
    if (!feed.isLive || !feed.imageSrc || !isStreamBroken) return;

    const watchdog = setInterval(() => {
      setStreamAttempt((n) => n + 1);
    }, 3000);

    return () => clearInterval(watchdog);
  }, [feed.isLive, feed.imageSrc, isStreamBroken]);

  const isAlarmActive = hasRenderedFrame && (detectionStatus?.alarm_active || false);
  const isCautionActive = hasRenderedFrame && (detectionStatus?.caution_active || false);
  const isPlaceholder = !feed.isLive || !feed.imageSrc;

  // Determine border and header colors based on state
  // Priority: Alarm (red) > Caution (yellow) > Normal
  const getBorderClass = () => {
    if (isAlarmActive) return 'ring-2 ring-[var(--zone-red)] animate-pulse';
    if (isCautionActive) return 'ring-2 ring-[var(--zone-yellow)]';
    return '';
  };

  const getHeaderClass = () => {
    if (isAlarmActive) return 'border-[var(--zone-red)] bg-[var(--zone-red)]/10';
    if (isCautionActive) return 'border-[var(--zone-yellow)] bg-[var(--zone-yellow)]/10';
    return 'border-[var(--border-dim)] bg-[var(--bg-secondary)]';
  };

  const getFooterClass = () => {
    if (isAlarmActive) return 'border-[var(--zone-red)] bg-[var(--zone-red)]/10';
    if (isCautionActive) return 'border-[var(--zone-yellow)] bg-[var(--zone-yellow)]/10';
    return 'border-[var(--border-dim)] bg-[var(--bg-secondary)]';
  };

  const getTextColor = () => {
    if (isAlarmActive) return 'text-[var(--zone-red)]';
    if (isCautionActive) return 'text-[var(--zone-yellow)]';
    return 'text-[var(--text-primary)]';
  };

  return (
    <div className={`feed-card corner-brackets rounded-lg group flex flex-col h-full ${getBorderClass()}`}>
      {/* Header */}
      <div className={`flex items-center justify-between px-3 py-2 border-b ${getHeaderClass()}`}>
        <div className="flex items-center gap-2">
          {isAlarmActive ? (
            <AlertTriangle size={14} className="text-[var(--zone-red)] animate-pulse" />
          ) : isCautionActive ? (
            <AlertCircle size={14} className="text-[var(--zone-yellow)]" />
          ) : (
            <div className={`w-2 h-2 rounded-full ${feed.isLive ? 'bg-[var(--accent-cyan)] status-live' : 'bg-[var(--text-muted)]'}`} />
          )}
          <span className={`text-sm font-semibold tracking-wide ${getTextColor()}`}>
            {feed.name}
          </span>
          <span className="text-xs text-[var(--text-muted)] font-mono">{feed.location}</span>
        </div>

        {/* Action Icons */}
        <div className="flex items-center gap-1">
          {/* People count */}
          {detectionStatus && detectionStatus.people_count > 0 && (
            <div className="flex items-center gap-1 px-2 py-0.5 rounded bg-[var(--bg-tertiary)] mr-2">
              <Users size={12} className={isAlarmActive ? 'text-[var(--zone-red)]' : isCautionActive ? 'text-[var(--zone-yellow)]' : 'text-[var(--accent-cyan)]'} />
              <span className={`text-xs font-mono ${isAlarmActive ? 'text-[var(--zone-red)]' : isCautionActive ? 'text-[var(--zone-yellow)]' : 'text-[var(--text-secondary)]'}`}>
                {detectionStatus.people_count}
              </span>
            </div>
          )}

          <button
            onClick={(e) => {
              e.stopPropagation();
              onEdit();
            }}
            className="p-1.5 rounded transition-all duration-200 text-[var(--text-secondary)] hover:text-[var(--accent-cyan)] hover:bg-[var(--accent-cyan-glow)]"
            title="Edit zones"
            disabled={isPlaceholder}
          >
            <Pencil size={14} />
          </button>
          <button
            onClick={(e) => {
              e.stopPropagation();
              onExpand();
            }}
            className="p-1.5 rounded transition-all duration-200 text-[var(--text-secondary)] hover:text-[var(--accent-cyan)] hover:bg-[var(--accent-cyan-glow)]"
            title="Expand view"
            disabled={isPlaceholder}
          >
            <Maximize2 size={14} />
          </button>
        </div>
      </div>

      {/* Feed Image */}
      <div className={`relative flex-1 min-h-0 overflow-hidden bg-gray-900 ${isAlarmActive ? 'border-2 border-[var(--zone-red)]' : isCautionActive ? 'border-2 border-[var(--zone-yellow)]' : ''}`}>
        {isPlaceholder ? (
          // Placeholder for disconnected feeds
          <div className="w-full h-full flex flex-col items-center justify-center bg-[var(--bg-tertiary)]">
            <VideoOff size={32} className="text-[var(--text-muted)] mb-2" />
            <span className="text-xs font-mono text-[var(--text-muted)]">NO SIGNAL</span>
          </div>
        ) : feed.isLive ? (
          <img
            ref={imgRef}
            src={`${feed.imageSrc}?attempt=${streamAttempt}`}
            alt={`${feed.name} feed`}
            className="w-full h-full object-contain"
            onLoad={() => {
              setHasRenderedFrame(true);
              setIsStreamBroken(false);
              setRetryCount(0);
              updateImgRect();
            }}
            onError={() => {
              setIsStreamBroken(true);
              setRetryCount((current) => {
                const nextRetry = current + 1;
                const delayMs = Math.min(1000 * Math.pow(2, Math.max(0, nextRetry - 1)), 2000);
                setTimeout(() => setStreamAttempt((n) => n + 1), delayMs);
                return nextRetry;
              });
            }}
          />
        ) : (
          <img
            src={feed.imageSrc}
            alt={`${feed.name} feed`}
            className="w-full h-full object-contain transition-transform duration-500 group-hover:scale-105"
          />
        )}

        {/* Reconnecting overlay — shown instead of black screen during stream errors */}
        {isStreamBroken && hasRenderedFrame && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-[var(--bg-primary)]/80 z-10">
            <div className="w-4 h-4 border-2 border-[var(--accent-cyan)] border-t-transparent rounded-full animate-spin mb-2" />
            <span className="text-[10px] font-mono text-[var(--text-muted)] tracking-wider">RECONNECTING</span>
          </div>
        )}

        {/* Alarm overlay (RED - highest priority) */}
        {isAlarmActive && (
          <div className="absolute top-2 left-2 px-2 py-1 rounded bg-[var(--zone-red)]/90 animate-pulse">
            <span className="text-xs font-bold text-white tracking-wider">ALARM</span>
          </div>
        )}

        {/* Caution overlay (YELLOW - lower priority than red) */}
        {isCautionActive && !isAlarmActive && (
          <div className="absolute top-2 left-2 px-2 py-1 rounded bg-[var(--zone-yellow)]/90">
            <span className="text-xs font-bold text-black tracking-wider">CAUTION</span>
          </div>
        )}

        {/* Zone count indicator */}
        {feed.zones.length > 0 && !isPlaceholder && (
          <div className="absolute bottom-2 left-2 px-2 py-0.5 rounded text-xs font-mono bg-[var(--bg-primary)]/80 border border-[var(--border-dim)]">
            <span className="text-[var(--text-secondary)]">ZONES:</span>{' '}
            <span className="text-[var(--accent-cyan)]">{feed.zones.length}</span>
          </div>
        )}

        {/* Danger count indicator (RED zone) */}
        {isAlarmActive && detectionStatus && detectionStatus.danger_count > 0 && (
          <div className="absolute bottom-2 right-2 px-2 py-0.5 rounded text-xs font-mono bg-[var(--zone-red)]/80 border border-[var(--zone-red)]">
            <span className="text-white">{detectionStatus.danger_count} IN RED ZONE</span>
          </div>
        )}

        {/* Caution count indicator (YELLOW zone) */}
        {isCautionActive && !isAlarmActive && detectionStatus && detectionStatus.caution_count > 0 && (
          <div className="absolute bottom-2 right-2 px-2 py-0.5 rounded text-xs font-mono bg-[var(--zone-yellow)]/80 border border-[var(--zone-yellow)]">
            <span className="text-black">{detectionStatus.caution_count} IN YELLOW ZONE</span>
          </div>
        )}

        {/* Zone overlay — positioned to match the object-contain image area */}
        {showZones && feed.zones.length > 0 && !isPlaceholder && imgRect && (
          <svg
            className="absolute pointer-events-none z-[5]"
            style={{ left: imgRect.left, top: imgRect.top, width: imgRect.width, height: imgRect.height }}
            viewBox="0 0 100 100"
            preserveAspectRatio="none"
          >
            {feed.zones.map((zone) => {
              const colors = ZONE_COLORS[zone.level] || ZONE_COLORS.green;
              const pointsStr = zone.points.map((p) => `${p.x},${p.y}`).join(' ');
              return (
                <polygon
                  key={zone.id}
                  points={pointsStr}
                  fill={colors.fill}
                  stroke={colors.stroke}
                  strokeWidth="0.5"
                />
              );
            })}
          </svg>
        )}

        {/* Scanline overlay */}
        {!isPlaceholder && <div className="absolute inset-0 scanlines pointer-events-none" />}

        {/* Vignette effect */}
        {!isPlaceholder && (
          <div
            className="absolute inset-0 pointer-events-none"
            style={{
              background: 'radial-gradient(ellipse at center, transparent 40%, rgba(10, 14, 20, 0.6) 100%)'
            }}
          />
        )}
      </div>

      {/* Timestamp + Coordinates */}
      <div className={`flex items-center justify-between px-3 py-1.5 border-t ${getFooterClass()}`}>
        <span className="text-[10px] font-mono tracking-wider text-white">
          {new Date().toLocaleTimeString('en-US', { hour12: false })} UTC
        </span>
        {detectionStatus?.position && (
          <span className="text-[10px] font-mono tracking-wider text-white">
            N:{detectionStatus.position.x.toFixed(1)} E:{detectionStatus.position.y.toFixed(1)} D:{detectionStatus.position.z.toFixed(1)}
          </span>
        )}
      </div>
    </div>
  );
}
