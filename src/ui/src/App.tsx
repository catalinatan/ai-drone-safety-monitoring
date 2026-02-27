import { useState, useCallback, useEffect, useRef } from 'react';
import { Plane, ChevronLeft, ChevronRight } from 'lucide-react';
import { CommandPanel } from './components/CommandPanel';
import { EditFeedPage } from './components/EditFeedPage';
import { ExpandedFeedView } from './components/ExpandedFeedView';
import { DroneControlPanel } from './components/DroneControlPanel';
import { mockFeeds, BACKEND_URL } from './data/mockFeeds';
import type { Feed, ViewState, Zone } from './types';

function App() {
  // Command Center specific state
  const [feeds, setFeeds] = useState<Feed[]>([]);
  const [commandViewState, setCommandViewState] = useState<ViewState>({ type: 'command' });
  const [sceneType, setSceneType] = useState('bridge');
  const [autoRefresh, setAutoRefresh] = useState(false);

  // Drone panel expand/collapse state
  const [droneExpanded, setDroneExpanded] = useState(false);
  const [droneActive, setDroneActive] = useState(false);
  const userCollapsedRef = useRef(false); // track if user manually collapsed

  // Poll drone status to detect activity for auto-expand
  useEffect(() => {
    const DRONE_API = 'http://localhost:8000';
    const pollDrone = async () => {
      try {
        const res = await fetch(`${DRONE_API}/status`);
        if (res.ok) {
          const data = await res.json();
          const active = data.is_navigating || data.returning_home;
          setDroneActive(active);
          // Auto-expand when drone becomes active (unless user manually collapsed)
          if (active && !userCollapsedRef.current) {
            setDroneExpanded(true);
          }
        }
      } catch {
        // Drone API not available
      }
    };
    pollDrone();
    const interval = setInterval(pollDrone, 2000);
    return () => clearInterval(interval);
  }, []);

  const toggleDronePanel = useCallback(() => {
    setDroneExpanded((prev) => {
      const next = !prev;
      // If user is collapsing, mark it so auto-expand doesn't override
      userCollapsedRef.current = !next;
      return next;
    });
  }, []);

  // Load feeds from backend on startup (fallback to mockFeeds if unavailable)
  useEffect(() => {
    const loadFeeds = async () => {
      try {
        const response = await fetch(`${BACKEND_URL}/feeds`);
        if (response.ok) {
          const data = await response.json();
          if (Array.isArray(data.feeds) && data.feeds.length > 0) {
            const backendFeeds: Feed[] = data.feeds.map((f: Feed) => ({
              id: f.id,
              name: f.name || f.id.toUpperCase().replace('-', ' '),
              location: f.location || 'Aerial Overview',
              imageSrc: f.imageSrc || `${BACKEND_URL}/video_feed/${f.id}`,
              zones: Array.isArray(f.zones) ? f.zones : [],
              isLive: f.isLive ?? true,
              sceneType: f.sceneType || null,
              autoSegActive: f.autoSegActive ?? false,
            }));
            console.log(`[INIT] Loaded ${backendFeeds.length} feeds from backend`);
            setFeeds(backendFeeds);
          } else {
            console.log('[INIT] No feeds from backend, using defaults');
            setFeeds(mockFeeds);
          }
          // Load global settings
          if (data.globalSceneType) setSceneType(data.globalSceneType);
          if (data.autoRefresh != null) setAutoRefresh(data.autoRefresh);
        } else {
          setFeeds(mockFeeds);
        }
      } catch (error) {
        console.log('[INIT] Backend not available, using default feeds');
        setFeeds(mockFeeds);
      }
    };

    loadFeeds();
  }, []);

  // Poll for zone updates from backend auto-segmentation (every 30s)
  // Skip polling while the user is editing zones to avoid overwriting their work
  const commandViewStateRef = useRef(commandViewState);
  commandViewStateRef.current = commandViewState;

  useEffect(() => {
    const pollZones = async () => {
      // Don't overwrite zones while user is editing
      if (commandViewStateRef.current.type === 'edit') return;
      try {
        const response = await fetch(`${BACKEND_URL}/feeds`);
        if (!response.ok) return;
        const data = await response.json();
        setFeeds((prevFeeds) =>
          prevFeeds.map((feed) => {
            const backendFeed = data.feeds?.find((f: Feed) => f.id === feed.id);
            if (backendFeed && Array.isArray(backendFeed.zones)) {
              return { ...feed, zones: backendFeed.zones };
            }
            return feed;
          })
        );
      } catch {
        // Backend unavailable, skip this poll
      }
    };

    const interval = setInterval(pollZones, 30_000);
    return () => clearInterval(interval);
  }, []);

  const getCurrentFeed = useCallback(
    (feedId: string): Feed | undefined => {
      return feeds.find((f) => f.id === feedId);
    },
    [feeds]
  );

  const handleEditFeed = useCallback((feedId: string) => {
    setCommandViewState({ type: 'edit', feedId });
  }, []);

  const handleExpandFeed = useCallback((feedId: string) => {
    setCommandViewState({ type: 'expanded', feedId });
  }, []);

  const handleBackToCommand = useCallback(() => {
    setCommandViewState({ type: 'command' });
  }, []);

  const handleSaveZones = useCallback(async (feedId: string, zones: Zone[]) => {
    // Update local state
    setFeeds((prev) =>
      prev.map((feed) => (feed.id === feedId ? { ...feed, zones } : feed))
    );

    // Save to backend API
    try {
      const response = await fetch(`${BACKEND_URL}/feeds/${feedId}/zones`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ zones }),
      });

      if (response.ok) {
        console.log(`[API] Zones saved for ${feedId}:`, zones.length, 'zones');
      } else {
        console.error(`[API] Failed to save zones for ${feedId}`);
      }
    } catch (error) {
      console.error(`[API] Error saving zones for ${feedId}:`, error);
    }

    setCommandViewState({ type: 'command' });
  }, []);

  const handleAutoSegmentAll = useCallback(async (): Promise<boolean> => {
    const feedIds = feeds.map((f) => f.id);
    try {
      const results = await Promise.all(
        feedIds.map((feedId) =>
          fetch(`${BACKEND_URL}/feeds/${feedId}/auto-segment`, { method: 'POST' })
            .then((res) => (res.ok ? res.json() : null))
            .catch(() => null)
        )
      );

      let totalZones = 0;
      setFeeds((prev) =>
        prev.map((feed, i) => {
          const data = results[i];
          if (data?.zones) {
            totalZones += data.zones.length;
            return { ...feed, zones: data.zones };
          }
          return feed;
        })
      );

      console.log(`[AUTO-SEG] All feeds segmented: ${totalZones} total zones`);
      return totalZones > 0;
    } catch (error) {
      console.error('[AUTO-SEG] Error segmenting all feeds:', error);
      return false;
    }
  }, [feeds]);

  const handleSaveSettings = useCallback(async (newSceneType: string, newAutoRefresh: boolean) => {
    const response = await fetch(`${BACKEND_URL}/settings`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sceneType: newSceneType, autoRefresh: newAutoRefresh }),
    });
    if (response.ok) {
      setSceneType(newSceneType);
      setAutoRefresh(newAutoRefresh);
      // Update scene type on all local feeds
      setFeeds((prev) => prev.map((f) => ({ ...f, sceneType: newSceneType as Feed['sceneType'] })));
      console.log(`[SETTINGS] Saved: sceneType=${newSceneType}, autoRefresh=${newAutoRefresh}`);
    }
  }, []);

  // Render Command Center views
  const renderCommandCenter = () => {
    switch (commandViewState.type) {
      case 'edit': {
        const feed = getCurrentFeed(commandViewState.feedId);
        if (!feed) {
          setCommandViewState({ type: 'command' });
          return null;
        }
        return (
          <EditFeedPage
            feed={feed}
            onSave={(zones) => handleSaveZones(feed.id, zones)}
            onCancel={handleBackToCommand}
          />
        );
      }

      case 'expanded': {
        const feed = getCurrentFeed(commandViewState.feedId);
        if (!feed) {
          setCommandViewState({ type: 'command' });
          return null;
        }
        return (
          <ExpandedFeedView
            feed={feed}
            onBack={handleBackToCommand}
            onEdit={() => handleEditFeed(feed.id)}
          />
        );
      }

      case 'command':
      default:
        return (
          <CommandPanel
            feeds={feeds}
            onEditFeed={handleEditFeed}
            onExpandFeed={handleExpandFeed}
            onAutoSegmentAll={handleAutoSegmentAll}
            sceneType={sceneType}
            autoRefresh={autoRefresh}
            onSaveSettings={handleSaveSettings}
          />
        );
    }
  };

  // In sub-views (edit/expanded), take full screen for the command center content
  const isInSubView = commandViewState.type !== 'command';

  if (isInSubView) {
    return (
      <div className="h-screen flex flex-col bg-[var(--bg-primary)]">
        {renderCommandCenter()}
      </div>
    );
  }

  return (
    <div className="h-screen flex bg-[var(--bg-primary)]">
      {/* Left: Command Center — takes full width when drone panel collapsed */}
      <div className="flex-1 min-w-0 h-full border-r border-[var(--border-dim)]">
        {renderCommandCenter()}
      </div>

      {/* Right: Drone Control — collapsible */}
      {droneExpanded ? (
        <div className="w-[40%] min-w-0 h-full flex flex-col relative">
          {/* Collapse button */}
          <button
            onClick={toggleDronePanel}
            className="absolute top-3 left-0 -translate-x-1/2 z-10 w-6 h-10 flex items-center justify-center rounded-full border border-[var(--border-dim)] bg-[var(--bg-secondary)] text-[var(--text-muted)] hover:text-[var(--accent-cyan)] hover:border-[var(--accent-cyan)] transition-all"
            title="Collapse drone panel"
          >
            <ChevronRight size={14} />
          </button>
          <DroneControlPanel />
        </div>
      ) : (
        /* Collapsed tab */
        <button
          onClick={toggleDronePanel}
          className={`
            h-full w-12 flex flex-col items-center justify-center gap-3 border-l transition-all duration-300 cursor-pointer
            ${droneActive
              ? 'border-[var(--accent-cyan)] bg-[var(--accent-cyan)]/10 text-[var(--accent-cyan)]'
              : 'border-[var(--border-dim)] bg-[var(--bg-secondary)] text-[var(--text-muted)] hover:text-[var(--accent-cyan)] hover:border-[var(--accent-cyan)]'
            }
          `}
          title="Expand drone panel"
        >
          <ChevronLeft size={14} />
          <Plane size={18} className={droneActive ? 'animate-pulse' : ''} />
          <span className="text-[9px] font-bold font-mono uppercase tracking-widest" style={{ writingMode: 'vertical-rl' }}>
            Drone
          </span>
          {droneActive && (
            <div className="w-2 h-2 rounded-full bg-[var(--zone-green)] status-live" />
          )}
        </button>
      )}
    </div>
  );
}

export default App;
