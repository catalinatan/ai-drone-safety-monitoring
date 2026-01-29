import { Pencil, Maximize2 } from 'lucide-react';
import type { Feed } from '../types';

interface FeedCardProps {
  feed: Feed;
  onEdit: () => void;
  onExpand: () => void;
}

export function FeedCard({ feed, onEdit, onExpand }: FeedCardProps) {
  return (
    <div className="feed-card corner-brackets rounded-lg group">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-[var(--border-dim)] bg-[var(--bg-secondary)]">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-[var(--accent-cyan)] status-live" />
          <span className="text-sm font-semibold tracking-wide text-[var(--text-primary)]">
            {feed.name}
          </span>
          <span className="text-xs text-[var(--text-muted)] font-mono">
            {feed.location}
          </span>
        </div>

        {/* Action Icons */}
        <div className="flex items-center gap-1">
          <button
            onClick={(e) => {
              e.stopPropagation();
              onEdit();
            }}
            className="p-1.5 rounded transition-all duration-200 text-[var(--text-secondary)] hover:text-[var(--accent-cyan)] hover:bg-[var(--accent-cyan-glow)]"
            title="Edit zones"
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
          >
            <Maximize2 size={14} />
          </button>
        </div>
      </div>

      {/* Feed Image */}
      <div className="relative aspect-video overflow-hidden bg-gray-900">
        {feed.isLive ? (
          <img
            src={feed.imageSrc}
            alt={`${feed.name} feed`}
            className="w-full h-full object-contain"
          />
        ) : (
          <img
            src={feed.imageSrc}
            alt={`${feed.name} feed`}
            className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
          />
        )}

        {/* Zone count indicator */}
        {feed.zones.length > 0 && (
          <div className="absolute bottom-2 left-2 px-2 py-0.5 rounded text-xs font-mono bg-[var(--bg-primary)]/80 border border-[var(--border-dim)]">
            <span className="text-[var(--text-secondary)]">ZONES:</span>{' '}
            <span className="text-[var(--accent-cyan)]">{feed.zones.length}</span>
          </div>
        )}

        {/* Scanline overlay */}
        <div className="absolute inset-0 scanlines pointer-events-none" />

        {/* Vignette effect */}
        <div
          className="absolute inset-0 pointer-events-none"
          style={{
            background: 'radial-gradient(ellipse at center, transparent 40%, rgba(10, 14, 20, 0.6) 100%)'
          }}
        />
      </div>

      {/* Timestamp */}
      <div className="px-3 py-1.5 border-t border-[var(--border-dim)] bg-[var(--bg-secondary)]">
        <span className="text-[10px] font-mono text-[var(--text-muted)] tracking-wider">
          {new Date().toLocaleTimeString('en-US', { hour12: false })} UTC
        </span>
      </div>
    </div>
  );
}
