import { useState } from 'react';
import { Shield, Radio, Eye, EyeOff, Scan, Loader2, Check, AlertTriangle, LayoutGrid, Wrench } from 'lucide-react';
import { FeedCard } from './FeedCard';
import type { Feed } from '../types';

interface CommandPanelProps {
  feeds: Feed[];
  onEditFeed: (feedId: string) => void;
  onExpandFeed: (feedId: string) => void;
  onToggleDetection: (feedId: string, enabled: boolean) => void;
  onAutoSegmentAll: () => Promise<boolean>;
  onOpenAdmin?: () => void;
}

export function CommandPanel({ feeds, onEditFeed, onExpandFeed, onToggleDetection, onAutoSegmentAll, onOpenAdmin }: CommandPanelProps) {
  const [showZones, setShowZones] = useState(true);
  const [segState, setSegState] = useState<'idle' | 'loading' | 'success' | 'empty' | 'failed'>('idle');
  const [gridSize, setGridSize] = useState<number | null>(null); // null = show all

  // Compute visible feeds and grid layout
  const visibleCount = gridSize ?? feeds.length;
  const visibleFeeds = feeds.slice(0, visibleCount);

  const getGridClass = (count: number) => {
    if (count <= 1) return 'grid-cols-1';
    if (count <= 2) return 'grid-cols-2 grid-rows-1';
    if (count <= 4) return 'grid-cols-2 grid-rows-2';
    if (count <= 6) return 'grid-cols-3 grid-rows-2';
    return 'grid-cols-3 grid-rows-3';
  };

  // Grid size options based on total feed count
  const gridOptions = [1, 2, 4, 6, 9].filter((n) => n <= feeds.length);

  const handleAutoSegmentAll = async () => {
    setSegState('loading');
    try {
      const hasZones = await onAutoSegmentAll();
      setSegState(hasZones ? 'success' : 'empty');
    } catch {
      setSegState('failed');
    }
    setTimeout(() => setSegState('idle'), 2500);
  };

  return (
    <div className="h-full flex flex-col bg-[var(--bg-primary)] tactical-grid">
      {/* Header */}
      <header className="flex items-center justify-between px-4 py-3 border-b border-[var(--border-dim)] bg-[var(--bg-secondary)]/80 backdrop-blur-sm">
        <div className="flex items-center gap-3">
          <Shield className="w-5 h-5 text-[var(--accent-cyan)]" />
          <h1 className="text-sm font-bold tracking-wider uppercase text-glow-cyan">
            Command Center
          </h1>
          <div className="h-4 w-px bg-[var(--border-dim)]" />
          <div className="flex items-center gap-2 px-2 py-1 rounded border border-[var(--border-dim)] bg-[var(--bg-tertiary)]">
            <Radio className="w-3 h-3 text-[var(--zone-green)] status-live" />
            <span className="text-[10px] font-mono text-[var(--zone-green)]">ONLINE</span>
          </div>
          {onOpenAdmin && (
            <button
              onClick={onOpenAdmin}
              className="flex items-center gap-1.5 px-2 py-1 rounded border border-[var(--border-dim)] bg-[var(--bg-tertiary)] text-[var(--text-muted)] hover:text-[var(--accent-cyan)] hover:border-[var(--accent-cyan)] transition-all duration-200"
              title="System Configuration"
            >
              <Wrench size={12} />
              <span className="text-[10px] font-bold font-mono uppercase tracking-wider">Admin</span>
            </button>
          )}
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={handleAutoSegmentAll}
            disabled={segState === 'loading'}
            className={`
              flex items-center gap-1.5 px-2 py-1 rounded border transition-all duration-200
              ${segState === 'loading'
                ? 'border-[var(--accent-cyan)] bg-[var(--accent-cyan)]/10 text-[var(--accent-cyan)] cursor-wait'
                : segState === 'success'
                ? 'border-[var(--zone-green)] bg-[var(--zone-green)]/10 text-[var(--zone-green)]'
                : segState === 'failed' || segState === 'empty'
                ? 'border-[var(--zone-red)] bg-[var(--zone-red)]/10 text-[var(--zone-red)]'
                : 'border-[var(--border-dim)] bg-[var(--bg-tertiary)] text-[var(--text-muted)] hover:text-[var(--accent-cyan)] hover:border-[var(--accent-cyan)]'
              }
            `}
            title="Auto-segment zones for all feeds"
          >
            {segState === 'loading' ? <Loader2 size={12} className="animate-spin" /> :
             segState === 'success' ? <Check size={12} /> :
             segState === 'failed' || segState === 'empty' ? <AlertTriangle size={12} /> :
             <Scan size={12} />}
            <span className="text-[10px] font-bold font-mono uppercase tracking-wider">
              {segState === 'loading' ? 'Segmenting...' :
               segState === 'success' ? 'Done' :
               segState === 'empty' ? 'No Zones' :
               segState === 'failed' ? 'Failed' :
               'Auto Segment'}
            </span>
          </button>
          <button
            onClick={() => setShowZones((v) => !v)}
            className={`
              flex items-center gap-1.5 px-2 py-1 rounded border transition-all duration-200
              ${showZones
                ? 'border-[var(--accent-cyan)] bg-[var(--accent-cyan)]/10 text-[var(--accent-cyan)]'
                : 'border-[var(--border-dim)] bg-[var(--bg-tertiary)] text-[var(--text-muted)] hover:text-[var(--text-secondary)]'
              }
            `}
            title={showZones ? 'Hide zones' : 'Show zones'}
          >
            {showZones ? <Eye size={12} /> : <EyeOff size={12} />}
            <span className="text-[10px] font-bold font-mono uppercase tracking-wider">Zones</span>
          </button>
          <div className="flex items-center gap-2 px-2 py-1 rounded border border-[var(--border-dim)] bg-[var(--bg-tertiary)]">
            <span className="text-[10px] text-[var(--text-muted)]">FEEDS:</span>
            <span className="text-xs font-mono text-[var(--accent-cyan)]">
              {feeds.filter(f => f.isLive).length}/{feeds.length}
            </span>
          </div>
          {/* Grid size selector */}
          <div className="flex items-center gap-1 px-1.5 py-0.5 rounded border border-[var(--border-dim)] bg-[var(--bg-tertiary)]">
            <LayoutGrid size={12} className="text-[var(--text-muted)]" />
            {gridOptions.map((n) => (
              <button
                key={n}
                onClick={() => setGridSize(n === feeds.length ? null : n)}
                className={`
                  w-6 h-6 rounded text-[10px] font-bold font-mono transition-all duration-150
                  ${(gridSize === n || (gridSize === null && n === feeds.length))
                    ? 'bg-[var(--accent-cyan)]/20 text-[var(--accent-cyan)] border border-[var(--accent-cyan)]/50'
                    : 'text-[var(--text-muted)] hover:text-[var(--accent-cyan)] hover:bg-[var(--accent-cyan)]/10'
                  }
                `}
                title={`Show ${n} feed${n > 1 ? 's' : ''}`}
              >
                {n}
              </button>
            ))}
            {feeds.length > 1 && (
              <button
                onClick={() => setGridSize(null)}
                className={`
                  px-1.5 h-6 rounded text-[10px] font-bold font-mono transition-all duration-150
                  ${gridSize === null
                    ? 'bg-[var(--accent-cyan)]/20 text-[var(--accent-cyan)] border border-[var(--accent-cyan)]/50'
                    : 'text-[var(--text-muted)] hover:text-[var(--accent-cyan)] hover:bg-[var(--accent-cyan)]/10'
                  }
                `}
                title="Show all feeds"
              >
                ALL
              </button>
            )}
          </div>
        </div>
      </header>

      {/* Main Content - Dynamic Grid */}
      <main className="flex-1 min-h-0 p-2 relative overflow-hidden">
        {/* Grid container */}
        <div className={`h-full grid ${getGridClass(visibleFeeds.length)} gap-2`}>
          {visibleFeeds.map((feed) => (
            <FeedCard
              key={feed.id}
              feed={feed}
              onEdit={() => onEditFeed(feed.id)}
              onExpand={() => onExpandFeed(feed.id)}
              onToggleDetection={onToggleDetection}
              showZones={showZones}
            />
          ))}
        </div>

        {/* Center overlay - decorative element */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 pointer-events-none">
          <div className="relative">
            {/* Crosshair */}
            <div className="w-24 h-24 border border-[var(--accent-cyan-dim)] rounded-full opacity-20" />
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-12 h-12 border border-[var(--accent-cyan-dim)] rounded-full opacity-30" />
            <div className="absolute top-1/2 left-0 w-full h-px bg-[var(--accent-cyan-dim)] opacity-20" />
            <div className="absolute top-0 left-1/2 w-px h-full bg-[var(--accent-cyan-dim)] opacity-20" />
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="px-4 py-2 border-t border-[var(--border-dim)] bg-[var(--bg-secondary)]/80">
        <div className="flex items-center justify-between text-[10px] font-mono text-[var(--text-muted)]">
          <span>{new Date().toLocaleDateString('en-US', {
            weekday: 'short',
            year: 'numeric',
            month: 'short',
            day: 'numeric'
          }).toUpperCase()}</span>
          <span>LAT: 51.4988° N | LON: 0.1749° W</span>
        </div>
      </footer>

    </div>
  );
}
