import { useState, useEffect } from 'react';
import { Pencil, Maximize2, AlertTriangle, AlertCircle, Users, VideoOff } from 'lucide-react';
import type { Feed, DetectionStatus } from '../types';
import { BACKEND_URL } from '../data/mockFeeds';

interface FeedCardProps {
  feed: Feed;
  onEdit: () => void;
  onExpand: () => void;
}

export function FeedCard({ feed, onEdit, onExpand }: FeedCardProps) {
  const [detectionStatus, setDetectionStatus] = useState<DetectionStatus | null>(null);

  // Poll for detection status (only for live feeds)
  useEffect(() => {
    if (!feed.isLive) return;

    const fetchStatus = async () => {
      try {
        const response = await fetch(`${BACKEND_URL}/feeds/${feed.id}/status`);
        if (response.ok) {
          const status = await response.json();
          setDetectionStatus(status);
        }
      } catch (error) {
        // Silently fail - backend might not be running
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 1000);

    return () => clearInterval(interval);
  }, [feed.id, feed.isLive]);

  const isAlarmActive = detectionStatus?.alarm_active || false;
  const isCautionActive = detectionStatus?.caution_active || false;
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

  const getTimestampColor = () => {
    if (isAlarmActive) return 'text-[var(--zone-red)]';
    if (isCautionActive) return 'text-[var(--zone-yellow)]';
    return 'text-[var(--text-muted)]';
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
          <span className="text-xs text-[var(--text-muted)] font-mono">
            {feed.location}
          </span>
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
            src={feed.imageSrc}
            alt={`${feed.name} feed`}
            className="w-full h-full object-contain"
          />
        ) : (
          <img
            src={feed.imageSrc}
            alt={`${feed.name} feed`}
            className="w-full h-full object-contain transition-transform duration-500 group-hover:scale-105"
          />
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

      {/* Timestamp */}
      <div className={`px-3 py-1.5 border-t ${getFooterClass()}`}>
        <span className={`text-[10px] font-mono tracking-wider ${getTimestampColor()}`}>
          {new Date().toLocaleTimeString('en-US', { hour12: false })} UTC
        </span>
      </div>
    </div>
  );
}
