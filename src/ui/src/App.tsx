import { useState, useCallback, useEffect, useRef } from 'react';
import { CommandPanel } from './components/CommandPanel';
import { EditFeedPage } from './components/EditFeedPage';
import { ExpandedFeedView } from './components/ExpandedFeedView';
import { DroneControlPanel } from './components/DroneControlPanel';
import { mockFeeds, BACKEND_URL } from './data/mockFeeds';
import type { Feed, ViewState, Zone } from './types';

function App() {
  // Command Center specific state
  const [feeds, setFeeds] = useState<Feed[]>(mockFeeds);
  const [commandViewState, setCommandViewState] = useState<ViewState>({ type: 'command' });

  // Load saved zones from backend on startup
  useEffect(() => {
    const loadZones = async () => {
      try {
        const response = await fetch(`${BACKEND_URL}/feeds`);
        if (response.ok) {
          const data = await response.json();
          // Merge backend zones with local feed config
          setFeeds((prevFeeds) =>
            prevFeeds.map((feed) => {
              const backendFeed = data.feeds?.find((f: Feed) => f.id === feed.id);
              if (backendFeed) {
                const updated = { ...feed };
                if (Array.isArray(backendFeed.zones)) {
                  console.log(`[INIT] Loaded ${backendFeed.zones.length} zones for ${feed.id}`);
                  updated.zones = backendFeed.zones;
                }
                if (backendFeed.sceneType) updated.sceneType = backendFeed.sceneType;
                if (backendFeed.autoSegActive != null) updated.autoSegActive = backendFeed.autoSegActive;
                return updated;
              }
              return feed;
            })
          );
        }
      } catch (error) {
        console.log('[INIT] Backend not available, using default feeds');
      }
    };

    loadZones();
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

  const handleAutoSegment = useCallback(async (feedId: string): Promise<Zone[] | null> => {
    try {
      const response = await fetch(`${BACKEND_URL}/feeds/${feedId}/auto-segment`, {
        method: 'POST',
      });

      if (response.ok) {
        const data = await response.json();
        console.log(`[AUTO-SEG] Generated ${data.zones_count} zones for ${feedId}`);

        if (data.zones) {
          setFeeds((prev) =>
            prev.map((feed) => (feed.id === feedId ? { ...feed, zones: data.zones } : feed))
          );
          return data.zones;
        }
      } else {
        console.error(`[AUTO-SEG] Failed for ${feedId}`);
      }
    } catch (error) {
      console.error(`[AUTO-SEG] Error for ${feedId}:`, error);
    }
    return null;
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
            onAutoSegment={() => handleAutoSegment(feed.id)}
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
      {/* Left: Command Center */}
      <div className="flex-[6] min-w-0 h-full border-r border-[var(--border-dim)]">
        {renderCommandCenter()}
      </div>

      {/* Right: Drone Control */}
      <div className="flex-[4] min-w-0 h-full">
        <DroneControlPanel />
      </div>
    </div>
  );
}

export default App;
