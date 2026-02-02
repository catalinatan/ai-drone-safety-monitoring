import { Shield, Radio } from 'lucide-react';
import { FeedCard } from './FeedCard';
import type { Feed } from '../types';

interface CommandPanelProps {
  feeds: Feed[];
  onEditFeed: (feedId: string) => void;
  onExpandFeed: (feedId: string) => void;
}

export function CommandPanel({ feeds, onEditFeed, onExpandFeed }: CommandPanelProps) {
  return (
    <div className="h-full flex flex-col bg-[var(--bg-primary)] tactical-grid pt-14">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-4 border-b border-[var(--border-dim)] bg-[var(--bg-secondary)]/80 backdrop-blur-sm">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <Shield className="w-6 h-6 text-[var(--accent-cyan)]" />
            <h1 className="text-xl font-bold tracking-wider uppercase text-glow-cyan">
              Safety Command Center
            </h1>
          </div>
          <div className="h-4 w-px bg-[var(--border-dim)]" />
          <span className="text-xs font-mono text-[var(--text-muted)] tracking-wide">
            AI MONITORING SYSTEM v1.0
          </span>
        </div>

        <div className="flex items-center gap-4">
          {/* System Status */}
          <div className="flex items-center gap-2 px-3 py-1.5 rounded border border-[var(--border-dim)] bg-[var(--bg-tertiary)]">
            <Radio className="w-3 h-3 text-[var(--zone-green)] status-live" />
            <span className="text-xs font-mono text-[var(--zone-green)]">
              SYSTEM ONLINE
            </span>
          </div>

          {/* Active feeds counter */}
          <div className="flex items-center gap-2 px-3 py-1.5 rounded border border-[var(--border-dim)] bg-[var(--bg-tertiary)]">
            <span className="text-xs text-[var(--text-muted)]">FEEDS:</span>
            <span className="text-sm font-mono text-[var(--accent-cyan)]">
              {feeds.filter(f => f.isLive).length}/4
            </span>
          </div>
        </div>
      </header>

      {/* Main Content - 4 Corner Grid */}
      <main className="flex-1 min-h-0 p-6 relative overflow-hidden">
        {/* Grid container */}
        <div className="h-full grid grid-cols-2 grid-rows-2 gap-6">
          {feeds.slice(0, 4).map((feed) => (
            <FeedCard
              key={feed.id}
              feed={feed}
              onEdit={() => onEditFeed(feed.id)}
              onExpand={() => onExpandFeed(feed.id)}
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
      <footer className="px-6 py-3 border-t border-[var(--border-dim)] bg-[var(--bg-secondary)]/80">
        <div className="flex items-center justify-between text-xs font-mono text-[var(--text-muted)]">
          <span>DRONE CONTROL: localhost:8000</span>
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
