import { useState, useCallback } from 'react';
import { AppNav, type AppView } from './components/AppNav';
import { CommandPanel } from './components/CommandPanel';
import { EditFeedPage } from './components/EditFeedPage';
import { ExpandedFeedView } from './components/ExpandedFeedView';
import { DroneControlPanel } from './components/DroneControlPanel';
import { mockFeeds } from './data/mockFeeds';
import type { Feed, ViewState, Zone } from './types';

function App() {
  // Top-level app view: Command Center vs Drone Control
  const [appView, setAppView] = useState<AppView>('command-center');

  // Command Center specific state
  const [feeds, setFeeds] = useState<Feed[]>(mockFeeds);
  const [commandViewState, setCommandViewState] = useState<ViewState>({ type: 'command' });

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

  const handleSaveZones = useCallback((feedId: string, zones: Zone[]) => {
    setFeeds((prev) =>
      prev.map((feed) => (feed.id === feedId ? { ...feed, zones } : feed))
    );

    // TODO: Integrate with backend API
    // POST /feeds/{feedId}/zones with zones data
    console.log(`[API STUB] Saving zones for ${feedId}:`, zones);

    setCommandViewState({ type: 'command' });
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
