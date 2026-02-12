import { useState, useCallback, useEffect } from 'react';
import { AppNav, type AppView } from './components/AppNav';
import { CommandPanel } from './components/CommandPanel';
import { EditFeedPage } from './components/EditFeedPage';
import { ExpandedFeedView } from './components/ExpandedFeedView';
import { DroneControlPanel } from './components/DroneControlPanel';
import { mockFeeds, BACKEND_URL } from './data/mockFeeds';
import type { Feed, ViewState, Zone } from './types';

function App() {
  // Top-level app view: Command Center vs Drone Control
  const [appView, setAppView] = useState<AppView>('command-center');

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
                if (backendFeed.zones?.length > 0) {
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

  // Check if we're in a sub-view that shouldn't show nav
  const isInSubView = commandViewState.type !== 'command';

  return (
    <div className="h-screen flex flex-col bg-[var(--bg-primary)]">
      {/* Navigation - hidden when in edit/expanded views */}
      {!isInSubView && (
        <div className="absolute top-4 left-1/2 -translate-x-1/2 z-50">
          <AppNav currentView={appView} onViewChange={setAppView} />
        </div>
      )}

      {/* Main Content */}
      <div className="flex-1">
        {appView === 'command-center' ? renderCommandCenter() : <DroneControlPanel />}
      </div>
    </div>
  );
}

export default App;
