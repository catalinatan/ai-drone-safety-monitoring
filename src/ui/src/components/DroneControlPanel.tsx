import { useState, useEffect, useCallback, useRef } from 'react';
import {
  Home,
  ChevronUp,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  ArrowUp,
  ArrowDown,
  Package,
  Radio,
  AlertTriangle,
  Plane,
  Hand,
  Camera,
  Battery,
  Wifi,
  Clock,
  Play,
  Pause,
  SkipBack,
  Crosshair,
  Check,
} from 'lucide-react';
import { BACKEND_URL } from '../data/mockFeeds';

interface DroneStatus {
  mode: 'manual' | 'automatic';
  connected: boolean;
  is_navigating: boolean;
  returning_home: boolean;
  grounded: boolean;
  target_position: [number, number, number] | null;
  pose: { x: number; y: number; z: number } | null;
}

interface TriggerMeta {
  id: number;
  feed_id: string;
  timestamp: string;
  deployed: boolean;
  replay_frame_count: number;
  replay_trigger_index: number;
  coords: [number, number, number];
}

const DRONE_API_BASE = 'http://localhost:8000';

export function DroneControlPanel() {
  const [status, setStatus] = useState<DroneStatus>({
    mode: 'manual',
    connected: false,
    is_navigating: false,
    returning_home: false,
    grounded: true,
    target_position: null,
    pose: null,
  });
  const [isConnected, setIsConnected] = useState(false);
  const [activeControls, setActiveControls] = useState<Set<string>>(new Set());
  const [isReturningHome, setIsReturningHome] = useState(false);
  const [equipmentDeployed, setEquipmentDeployed] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  // Multi-trigger state
  const [triggers, setTriggers] = useState<TriggerMeta[]>([]);
  const [selectedTriggerId, setSelectedTriggerId] = useState<number | null>(null);
  const [replayFps, setReplayFps] = useState(10);
  const lastTriggerCountRef = useRef(0);
  // Replay player state
  const [replayFrame, setReplayFrame] = useState(0);
  const [replayPlaying, setReplayPlaying] = useState(false);
  const replayIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const videoRef = useRef<HTMLImageElement>(null);
  const lastVelocityRef = useRef({ vx: 0, vy: 0, vz: 0 });
  const [airborneStart, setAirborneStart] = useState<number | null>(null);
  const wasAirborneRef = useRef(false);
  const [flightTime, setFlightTime] = useState('--:--');

  // Fetch drone status periodically
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const response = await fetch(`${DRONE_API_BASE}/status`);
        if (response.ok) {
          const data = await response.json();
          setStatus(data);
          setIsConnected(true);

          // Detect airborne transitions (connected + not grounded = in the air)
          const airborne = data.connected && !data.grounded;
          if (!wasAirborneRef.current && airborne) {
            setAirborneStart(Date.now());
          } else if (wasAirborneRef.current && !airborne) {
            setAirborneStart(null);
          }
          wasAirborneRef.current = airborne;
        } else {
          setIsConnected(false);
        }
      } catch {
        setIsConnected(false);
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 1000);
    return () => clearInterval(interval);
  }, []);

  // Flight time ticker (airborne time)
  useEffect(() => {
    if (!airborneStart) { setFlightTime('--:--'); return; }
    const tick = setInterval(() => {
      const elapsed = Math.floor((Date.now() - airborneStart) / 1000);
      const mm = String(Math.floor(elapsed / 60)).padStart(2, '0');
      const ss = String(elapsed % 60).padStart(2, '0');
      setFlightTime(`${mm}:${ss}`);
    }, 1000);
    return () => clearInterval(tick);
  }, [airborneStart]);

  // Poll triggers from backend
  useEffect(() => {
    const fetchTriggers = async () => {
      try {
        const response = await fetch(`${BACKEND_URL}/triggers`);
        if (response.ok) {
          const data = await response.json();
          const newTriggers: TriggerMeta[] = data.triggers || [];
          setTriggers(newTriggers);
          setReplayFps(data.replay_fps || 10);

          // Auto-select latest trigger when new ones arrive
          if (newTriggers.length > lastTriggerCountRef.current && newTriggers.length > 0) {
            const latest = newTriggers[newTriggers.length - 1];
            setSelectedTriggerId(latest.id);
            setReplayFrame(0);
            if (latest.replay_frame_count > 0) {
              setReplayPlaying(true);
            }
          }
          lastTriggerCountRef.current = newTriggers.length;
        }
      } catch {
        // Backend unavailable
      }
    };

    fetchTriggers();
    const interval = setInterval(fetchTriggers, 2000);
    return () => clearInterval(interval);
  }, []);

  // Get selected trigger metadata
  const selectedTrigger = triggers.find((t) => t.id === selectedTriggerId) || null;

  // Replay playback — advance frames at replay FPS
  useEffect(() => {
    if (replayIntervalRef.current) {
      clearInterval(replayIntervalRef.current);
      replayIntervalRef.current = null;
    }
    if (!replayPlaying || !selectedTrigger || selectedTrigger.replay_frame_count <= 0) return;

    const frameCount = selectedTrigger.replay_frame_count;
    replayIntervalRef.current = setInterval(() => {
      setReplayFrame((prev) => {
        const next = prev + 1;
        if (next >= frameCount) {
          setReplayPlaying(false);
          return prev;
        }
        return next;
      });
    }, 1000 / replayFps);

    return () => {
      if (replayIntervalRef.current) {
        clearInterval(replayIntervalRef.current);
        replayIntervalRef.current = null;
      }
    };
  }, [replayPlaying, selectedTrigger?.id, selectedTrigger?.replay_frame_count, replayFps]);

  // Show error with auto-dismiss
  const showError = useCallback((message: string) => {
    setErrorMessage(message);
    setTimeout(() => setErrorMessage(null), 4000);
  }, []);

  // Deploy drone to selected trigger
  const handleDeployToTrigger = useCallback(async () => {
    if (!selectedTriggerId) return;
    try {
      const response = await fetch(`${BACKEND_URL}/triggers/${selectedTriggerId}/deploy`, {
        method: 'POST',
      });
      if (response.ok) {
        setTriggers((prev) =>
          prev.map((t) => (t.id === selectedTriggerId ? { ...t, deployed: true } : t))
        );
        setStatus((prev) => ({ ...prev, is_navigating: true }));
      } else {
        const data = await response.json();
        showError(data.detail || 'Failed to deploy');
      }
    } catch {
      showError('Connection error: Failed to deploy');
    }
  }, [selectedTriggerId, showError]);

  // Handle mode switch (manual override)
  const handleModeSwitch = useCallback(async () => {
    const newMode = status.mode === 'manual' ? 'automatic' : 'manual';
    try {
      const response = await fetch(`${DRONE_API_BASE}/mode`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: newMode }),
      });
      if (response.ok) {
        setStatus((prev) => ({ ...prev, mode: newMode }));
      } else {
        const data = await response.json();
        showError(data.detail || 'Failed to switch mode');
      }
    } catch (error) {
      showError('Connection error: Failed to switch mode');
      console.error('Failed to switch mode:', error);
    }
  }, [status.mode, showError]);

  // Handle return home
  const handleReturnHome = useCallback(async () => {
    try {
      // /return_home switches to automatic mode internally
      const response = await fetch(`${DRONE_API_BASE}/return_home`, {
        method: 'POST',
      });

      if (response.ok) {
        setIsReturningHome(true);
        setStatus((prev) => ({ ...prev, mode: 'automatic', is_navigating: true, returning_home: true }));
      } else {
        const data = await response.json();
        showError(data.detail || 'Failed to return home');
      }
    } catch (error) {
      showError('Connection error: Failed to return home');
      console.error('Failed to return home:', error);
    }
  }, [status.mode, showError]);

  // Handle deploy equipment (placeholder)
  const handleDeployEquipment = useCallback(() => {
    setEquipmentDeployed(true);
    console.log('[EQUIPMENT] Deploy command sent (placeholder)');
    // TODO: Integrate with actual equipment deployment API
    setTimeout(() => setEquipmentDeployed(false), 3000);
  }, []);

  // Send move command to drone API
  const sendMoveCommand = useCallback(async (vx: number, vy: number, vz: number) => {
    if (status.mode !== 'manual' || !isConnected) return;
    try {
      await fetch(`${DRONE_API_BASE}/move`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ vx, vy, vz }),
      });
    } catch (error) {
      console.error('Failed to send move command:', error);
    }
  }, [status.mode, isConnected]);

  // Continuously send move commands based on active controls
  useEffect(() => {
    if (status.mode !== 'manual' || !isConnected) return;

    const interval = setInterval(() => {
      // Calculate velocity from active controls
      let vx = 0, vy = 0, vz = 0;

      // Forward/backward (North/South)
      if (activeControls.has('forward')) vx = 3;
      else if (activeControls.has('backward')) vx = -3;

      // Left/right (West/East)
      if (activeControls.has('left')) vy = -3;
      else if (activeControls.has('right')) vy = 3;

      // Up/down (NED: negative = up)
      if (activeControls.has('up')) vz = -2;
      else if (activeControls.has('down')) vz = 2;

      // Only send if velocity changed (avoids redundant zero commands)
      const last = lastVelocityRef.current;
      if (vx !== last.vx || vy !== last.vy || vz !== last.vz) {
        lastVelocityRef.current = { vx, vy, vz };
        sendMoveCommand(vx, vy, vz);
      }
    }, 100); // Send every 100ms

    return () => clearInterval(interval);
  }, [status.mode, isConnected, activeControls, sendMoveCommand]);

  // Control button press/release handlers
  const handleControlPress = useCallback(
    (control: string) => {
      if (status.mode !== 'manual') return;
      setActiveControls((prev) => new Set(prev).add(control));
    },
    [status.mode]
  );

  const handleControlRelease = useCallback((control: string) => {
    setActiveControls((prev) => {
      const next = new Set(prev);
      next.delete(control);
      return next;
    });
  }, []);

  // Keyboard event handlers for actual drone control
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (status.mode !== 'manual') return;

      const keyMap: Record<string, string> = {
        w: 'forward',
        s: 'backward',
        a: 'left',
        d: 'right',
        z: 'up',
        x: 'down',
      };

      const control = keyMap[e.key.toLowerCase()];
      if (control) {
        setActiveControls((prev) => new Set(prev).add(control));
      }
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      const keyMap: Record<string, string> = {
        w: 'forward',
        s: 'backward',
        a: 'left',
        d: 'right',
        z: 'up',
        x: 'down',
      };

      const control = keyMap[e.key.toLowerCase()];
      if (control) {
        setActiveControls((prev) => {
          const next = new Set(prev);
          next.delete(control);
          return next;
        });
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, [status.mode]);

  // Sync returning home state with backend
  useEffect(() => {
    setIsReturningHome(status.returning_home);
  }, [status.returning_home]);

  const isManual = status.mode === 'manual';
  const isAutomatic = status.mode === 'automatic';
  const isDroneFlying = isConnected && !status.grounded;

  return (
    <div className="h-full flex flex-col bg-[var(--bg-primary)] tactical-grid relative overflow-hidden">
      {/* Header Bar */}
      <header className="flex items-center justify-between px-4 py-3 border-b border-[var(--border-dim)] bg-[var(--bg-secondary)]/80 backdrop-blur-sm flex-shrink-0">
        <div className="flex items-center gap-3">
          <Plane className="w-5 h-5 text-[var(--accent-cyan)]" />
          <h1 className="text-sm font-bold tracking-wider uppercase text-glow-cyan">
            Drone Control
          </h1>
          <div className="h-4 w-px bg-[var(--border-dim)]" />
          {/* Connection Status */}
          <div
            className={`
            flex items-center gap-2 px-2 py-1 rounded border
            ${
              isConnected
                ? 'border-[var(--zone-green)]/50 bg-[var(--zone-green-fill)]'
                : 'border-[var(--zone-red)]/50 bg-[var(--zone-red-fill)]'
            }
          `}
          >
            <Radio
              size={12}
              className={isConnected ? 'text-[var(--zone-green)] status-live' : 'text-[var(--zone-red)]'}
            />
            <span
              className={`text-[10px] font-mono ${isConnected ? 'text-[var(--zone-green)]' : 'text-[var(--zone-red)]'}`}
            >
              {isConnected ? 'CONNECTED' : 'DISCONNECTED'}
            </span>
          </div>
        </div>

        {/* Navigation Status */}
        {isAutomatic && status.is_navigating && (
          <div className={`flex items-center gap-2 px-2 py-1 rounded border ${
            isReturningHome
              ? 'border-[var(--accent-cyan)]/50 bg-[var(--accent-cyan)]/10'
              : 'border-[var(--zone-yellow)]/50 bg-[var(--zone-yellow-fill)]'
          }`}>
            <Home size={12} className={isReturningHome
              ? 'text-[var(--accent-cyan)] animate-pulse'
              : 'hidden'
            } />
            <Plane size={12} className={isReturningHome
              ? 'hidden'
              : 'text-[var(--zone-yellow)] animate-pulse'
            } />
            <span className={`text-[10px] font-mono ${
              isReturningHome ? 'text-[var(--accent-cyan)]' : 'text-[var(--zone-yellow)]'
            }`}>
              {isReturningHome ? 'RETURNING HOME' : 'NAVIGATING'}
            </span>
          </div>
        )}
      </header>

      {/* Telemetry Bar */}
      {isConnected && (
        <div className="flex items-center justify-between px-4 py-1.5 border-b border-[var(--border-dim)] bg-[var(--bg-secondary)]/50 backdrop-blur-sm flex-shrink-0">
          <div className="flex items-center gap-1.5">
            <Battery size={12} className="text-[var(--zone-green)]" />
            <span className="text-[10px] font-bold font-mono uppercase tracking-wider text-[var(--zone-green)]">87%</span>
          </div>
          <div className="flex items-center gap-1.5">
            <ArrowUp size={12} className="text-[var(--accent-cyan)]" />
            <span className="text-[10px] font-bold font-mono uppercase tracking-wider text-[var(--accent-cyan)]">
              ALT {status.pose ? Math.abs(status.pose.z).toFixed(1) : '--'}M
            </span>
          </div>
          <div className="flex items-center gap-1.5">
            <Wifi size={12} className="text-[var(--zone-green)]" />
            <span className="text-[10px] font-bold font-mono uppercase tracking-wider text-[var(--zone-green)]">STRONG</span>
          </div>
          <div className="flex items-center gap-1.5">
            <Clock size={12} className="text-[var(--text-secondary)]" />
            <span className="text-[10px] font-bold font-mono uppercase tracking-wider text-[var(--text-secondary)]">{flightTime}</span>
          </div>
        </div>
      )}

      {/* Main Content - CCTV Trigger + Drone Feeds */}
      <main className="flex-1 flex flex-col items-center px-4 py-3 gap-3 min-h-0 overflow-hidden">
        {/* Multi-Trigger Section */}
        <div className="w-full max-w-sm flex-shrink-0">
          {/* Trigger thumbnail list — horizontal scroll */}
          {triggers.length > 0 && (
            <div className="mb-2 flex items-center gap-1.5 overflow-x-auto pb-1 scrollbar-thin">
              {triggers.map((t) => (
                <button
                  key={t.id}
                  onClick={() => {
                    setSelectedTriggerId(t.id);
                    setReplayFrame(0);
                    setReplayPlaying(t.replay_frame_count > 0);
                  }}
                  className={`
                    flex-shrink-0 relative w-16 h-10 rounded border-2 overflow-hidden transition-all duration-200
                    ${selectedTriggerId === t.id
                      ? 'border-[var(--accent-cyan)] shadow-[0_0_8px_var(--accent-cyan-dim)]'
                      : 'border-[var(--border-dim)] hover:border-[var(--text-muted)]'
                    }
                  `}
                  title={`Trigger #${t.id} — ${t.feed_id.toUpperCase()} — ${new Date(t.timestamp).toLocaleTimeString('en-US', { hour12: false })}`}
                >
                  <img
                    src={`${BACKEND_URL}/triggers/${t.id}/snapshot`}
                    alt={`Trigger ${t.id}`}
                    className="w-full h-full object-cover"
                  />
                  {/* Deployed indicator */}
                  <div className={`absolute top-0.5 right-0.5 w-3 h-3 rounded-full flex items-center justify-center ${
                    t.deployed ? 'bg-[var(--zone-green)]' : 'bg-[var(--zone-red)]'
                  }`}>
                    {t.deployed ? <Check size={8} className="text-white" /> : <Crosshair size={8} className="text-white" />}
                  </div>
                  {/* Trigger ID */}
                  <div className="absolute bottom-0 left-0 right-0 bg-black/60 text-center">
                    <span className="text-[8px] font-mono text-white">#{t.id}</span>
                  </div>
                </button>
              ))}
            </div>
          )}

          {/* Replay viewer */}
          <div className={`relative aspect-video rounded-lg overflow-hidden border-2 ${selectedTrigger ? 'border-[var(--zone-red)]' : 'border-dashed border-[var(--border-dim)]'} corner-brackets`}>
            {selectedTrigger && selectedTrigger.replay_frame_count > 0 ? (
              <img
                src={`${BACKEND_URL}/triggers/${selectedTrigger.id}/replay/${replayFrame}`}
                alt="Trigger Replay"
                className="w-full h-full object-cover"
              />
            ) : selectedTrigger ? (
              <img
                src={`${BACKEND_URL}/triggers/${selectedTrigger.id}/snapshot`}
                alt="CCTV Trigger"
                className="w-full h-full object-cover"
              />
            ) : (
              <div className="w-full h-full flex flex-col items-center justify-center bg-[var(--bg-card)] gap-2">
                <Camera size={28} className="text-[var(--text-muted)]" />
                <span className="text-xs font-mono text-[var(--text-secondary)] tracking-wider">CCTV TRIGGER FEED</span>
                <span className="text-[10px] font-mono text-[var(--text-muted)]">Awaiting hazard detection...</span>
              </div>
            )}
            {/* Label overlay */}
            <div className="absolute top-2 left-2 flex items-center gap-2 px-2 py-1 rounded bg-[var(--bg-primary)]/80 border border-[var(--border-dim)]">
              <div className={`w-2 h-2 rounded-full ${selectedTrigger ? 'bg-[var(--zone-red)]' : 'bg-[var(--text-muted)]'}`} />
              <span className="text-[10px] font-mono text-[var(--text-secondary)] tracking-wider">
                {selectedTrigger ? selectedTrigger.feed_id.toUpperCase().replace('-', ' ') + ' TRIGGER #' + selectedTrigger.id : 'TRIGGER CAM'}
              </span>
            </div>
            {/* Timestamp */}
            {selectedTrigger && (
              <div className="absolute bottom-2 right-2 px-2 py-0.5 rounded bg-[var(--bg-primary)]/80 border border-[var(--border-dim)]">
                <span className="text-[10px] font-mono text-[var(--text-muted)] tracking-wider">
                  {new Date(selectedTrigger.timestamp).toLocaleTimeString('en-US', { hour12: false })}
                </span>
              </div>
            )}
            <div className="absolute inset-0 scanlines pointer-events-none opacity-50" />
          </div>

          {/* Replay controls + Deploy button */}
          {selectedTrigger && selectedTrigger.replay_frame_count > 0 && (
            <div className="mt-1.5 flex items-center gap-2">
              {/* Play/Pause */}
              <button
                onClick={() => {
                  if (!replayPlaying && replayFrame >= selectedTrigger.replay_frame_count - 1) {
                    setReplayFrame(0);
                  }
                  setReplayPlaying((p) => !p);
                }}
                className="p-1 rounded border border-[var(--border-dim)] bg-[var(--bg-secondary)] text-[var(--text-secondary)] hover:text-[var(--accent-cyan)] hover:border-[var(--accent-cyan)] transition-all"
              >
                {replayPlaying ? <Pause size={12} /> : <Play size={12} />}
              </button>
              {/* Restart */}
              <button
                onClick={() => { setReplayFrame(0); setReplayPlaying(true); }}
                className="p-1 rounded border border-[var(--border-dim)] bg-[var(--bg-secondary)] text-[var(--text-secondary)] hover:text-[var(--accent-cyan)] hover:border-[var(--accent-cyan)] transition-all"
                title="Restart"
              >
                <SkipBack size={12} />
              </button>
              {/* Progress bar */}
              <div
                className="flex-1 h-3 rounded bg-[var(--bg-tertiary)] border border-[var(--border-dim)] relative cursor-pointer overflow-hidden"
                onClick={(e) => {
                  const rect = e.currentTarget.getBoundingClientRect();
                  const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
                  setReplayFrame(Math.round(pct * (selectedTrigger.replay_frame_count - 1)));
                }}
              >
                {/* Playback position */}
                <div
                  className="absolute top-0 left-0 h-full bg-[var(--accent-cyan)]/40 transition-[width] duration-75"
                  style={{ width: `${(replayFrame / Math.max(1, selectedTrigger.replay_frame_count - 1)) * 100}%` }}
                />
                {/* Trigger marker */}
                {selectedTrigger.replay_trigger_index > 0 && (
                  <div
                    className="absolute top-0 h-full w-0.5 bg-[var(--zone-red)]"
                    style={{ left: `${(selectedTrigger.replay_trigger_index / Math.max(1, selectedTrigger.replay_frame_count - 1)) * 100}%` }}
                    title="Trigger point"
                  />
                )}
              </div>
              {/* Frame counter */}
              <span className="text-[9px] font-mono text-[var(--text-muted)] whitespace-nowrap">
                {String(replayFrame + 1).padStart(2, '0')}/{selectedTrigger.replay_frame_count}
              </span>
            </div>
          )}

          {/* Deploy to trigger button */}
          {selectedTrigger && (
            <button
              onClick={handleDeployToTrigger}
              disabled={selectedTrigger.deployed || status.is_navigating}
              className={`
                mt-2 w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg border-2 transition-all duration-300 font-mono
                ${selectedTrigger.deployed
                  ? 'border-[var(--zone-green)]/50 bg-[var(--zone-green)]/10 text-[var(--zone-green)] cursor-default'
                  : status.is_navigating
                    ? 'border-[var(--border-dim)] bg-[var(--bg-tertiary)] text-[var(--text-muted)] cursor-not-allowed'
                    : 'border-[var(--zone-red)]/80 bg-[var(--zone-red)]/15 text-[var(--zone-red)] hover:border-[var(--zone-red)] hover:bg-[var(--zone-red)]/25'
                }
              `}
            >
              {selectedTrigger.deployed ? (
                <>
                  <Check size={14} />
                  <span className="text-[10px] font-bold uppercase tracking-wider">DEPLOYED</span>
                </>
              ) : (
                <>
                  <Crosshair size={14} />
                  <span className="text-[10px] font-bold uppercase tracking-wider">
                    {status.is_navigating ? 'DRONE BUSY' : 'DEPLOY TO TRIGGER'}
                  </span>
                </>
              )}
            </button>
          )}
        </div>

        {/* Drone Feeds - Side by Side */}
        <div className="flex items-center justify-center gap-3 w-full flex-1 min-h-0">
          {/* Forward Camera */}
          <div className="relative flex-1 h-full rounded-lg overflow-hidden border-2 border-[var(--border-dim)] corner-brackets">
            {isConnected ? (
              <img
                ref={videoRef}
                src={`${DRONE_API_BASE}/video_feed/down`}
                alt="Drone Forward Camera"
                className="w-full h-full object-cover"
              />
            ) : (
              <div className="w-full h-full bg-[var(--bg-card)] flex flex-col items-center justify-center gap-2">
                <AlertTriangle size={32} className="text-[var(--zone-yellow)]" />
                <span className="text-sm font-mono text-[var(--text-secondary)]">NO SIGNAL</span>
              </div>
            )}
            <div className="absolute inset-0 scanlines pointer-events-none" />
            <div className="absolute inset-0 pointer-events-none">
              <div className="absolute top-1/2 left-0 right-0 h-px bg-[var(--accent-cyan)]/20" />
              <div className="absolute left-1/2 top-0 bottom-0 w-px bg-[var(--accent-cyan)]/20" />
              <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-12 h-12 border border-[var(--accent-cyan)]/30 rounded-full" />
            </div>
            <div className="absolute top-2 left-2 flex items-center gap-1.5 px-1.5 py-0.5 rounded bg-[var(--bg-primary)]/80 border border-[var(--border-dim)]">
              <div className={`w-1.5 h-1.5 rounded-full ${isAutomatic ? 'bg-[var(--zone-yellow)]' : 'bg-[var(--zone-green)]'} status-live`} />
              <span className="text-[9px] font-mono text-[var(--text-secondary)] tracking-wider">FORWARD</span>
            </div>
          </div>

          {/* Down Camera */}
          <div className="relative flex-1 h-full rounded-lg overflow-hidden border-2 border-[var(--border-dim)] corner-brackets">
            {isConnected ? (
              <img
                src={`${DRONE_API_BASE}/video_feed/forward`}
                alt="Drone Down Camera"
                className="w-full h-full object-cover"
              />
            ) : (
              <div className="w-full h-full bg-[var(--bg-card)] flex flex-col items-center justify-center gap-2">
                <AlertTriangle size={32} className="text-[var(--zone-yellow)]" />
                <span className="text-sm font-mono text-[var(--text-secondary)]">NO SIGNAL</span>
              </div>
            )}
            <div className="absolute inset-0 scanlines pointer-events-none" />
            <div className="absolute inset-0 pointer-events-none">
              <div className="absolute top-1/2 left-0 right-0 h-px bg-[var(--accent-cyan)]/20" />
              <div className="absolute left-1/2 top-0 bottom-0 w-px bg-[var(--accent-cyan)]/20" />
              <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-12 h-12 border border-[var(--accent-cyan)]/30 rounded-full" />
            </div>
            <div className="absolute top-2 left-2 flex items-center gap-1.5 px-1.5 py-0.5 rounded bg-[var(--bg-primary)]/80 border border-[var(--border-dim)]">
              <div className={`w-1.5 h-1.5 rounded-full ${isConnected ? 'bg-[var(--zone-green)]' : 'bg-[var(--zone-red)]'} status-live`} />
              <span className="text-[9px] font-mono text-[var(--text-secondary)] tracking-wider">DOWN</span>
            </div>
          </div>
        </div>
      </main>

      {/* Bottom Controls Bar */}
      <footer className="flex-shrink-0 px-3 py-3">
        <div className="relative rounded-xl border border-[var(--accent-cyan)]/30 bg-[var(--bg-primary)]/80 backdrop-blur-xl px-4 py-4 shadow-[0_0_20px_rgba(0,0,0,0.5),inset_0_1px_0_rgba(255,255,255,0.05)]">

          {/* Auto-flight warning */}
          {isAutomatic && (
            <div className="absolute -top-3 left-1/2 -translate-x-1/2 flex items-center gap-1.5 px-2 py-0.5 rounded-md bg-[var(--bg-primary)] border border-[var(--zone-yellow)] animate-pulse whitespace-nowrap">
              <AlertTriangle size={10} className="text-[var(--zone-yellow)]" />
              <span className="text-[10px] font-bold font-mono text-[var(--zone-yellow)]">
                Auto flight active
              </span>
            </div>
          )}

          {/* Controls: Left (Movement + Altitude) | Right (RTH + Mode + Deploy) */}
          <div className="flex items-start justify-evenly">
            {/* Left side: Movement + Altitude */}
            <div className="flex items-start gap-4">
              {/* Movement Controls (WASD) */}
              <div className="flex flex-col items-center gap-1">
                <span className="text-[9px] font-bold font-mono text-white uppercase tracking-wider mb-1">
                  Movement
                </span>
                <ControlButton
                  icon={<ChevronUp size={18} />}
                  isActive={activeControls.has('forward')}
                  isDisabled={!isManual}
                  onPress={() => handleControlPress('forward')}
                  onRelease={() => handleControlRelease('forward')}
                  label="W"
                  labelPosition="above"
                />
                <div className="flex gap-1">
                  <ControlButton
                    icon={<ChevronLeft size={18} />}
                    isActive={activeControls.has('left')}
                    isDisabled={!isManual}
                    onPress={() => handleControlPress('left')}
                    onRelease={() => handleControlRelease('left')}
                    label="A"
                  />
                  <ControlButton
                    icon={<ChevronDown size={18} />}
                    isActive={activeControls.has('backward')}
                    isDisabled={!isManual}
                    onPress={() => handleControlPress('backward')}
                    onRelease={() => handleControlRelease('backward')}
                    label="S"
                  />
                  <ControlButton
                    icon={<ChevronRight size={18} />}
                    isActive={activeControls.has('right')}
                    isDisabled={!isManual}
                    onPress={() => handleControlPress('right')}
                    onRelease={() => handleControlRelease('right')}
                    label="D"
                  />
                </div>
              </div>

              {/* Altitude Controls */}
              <div className="flex flex-col items-center gap-1">
                <span className="text-[9px] font-bold font-mono text-white uppercase tracking-wider mb-1">
                  Altitude
                </span>
                <ControlButton
                  icon={<ArrowUp size={18} />}
                  isActive={activeControls.has('up')}
                  isDisabled={!isManual}
                  onPress={() => handleControlPress('up')}
                  onRelease={() => handleControlRelease('up')}
                  label="Z"
                  variant="altitude"
                  labelPosition="above"
                />
                <ControlButton
                  icon={<ArrowDown size={18} />}
                  isActive={activeControls.has('down')}
                  isDisabled={!isManual}
                  onPress={() => handleControlPress('down')}
                  onRelease={() => handleControlRelease('down')}
                  label="X"
                  variant="altitude"
                />
              </div>
            </div>

            {/* Right side: RTH + Mode Toggle + Deploy */}
            <div className="flex flex-col items-center gap-2">
              {/* Return Home */}
              <button
                onClick={handleReturnHome}
                disabled={isReturningHome}
                className={`
                  flex items-center gap-1.5 px-3 py-2 rounded-lg border-2 transition-all duration-300 w-full justify-center
                  ${
                    isReturningHome
                      ? 'border-[var(--zone-yellow)] bg-[var(--zone-yellow)]/20 text-[var(--zone-yellow)]'
                      : 'border-[var(--accent-cyan)]/60 bg-[var(--accent-cyan)]/10 text-[var(--accent-cyan)] hover:border-[var(--accent-cyan)]'
                  }
                `}
              >
                <Home size={16} className={isReturningHome ? 'animate-pulse' : ''} />
                <span className="text-[9px] font-bold font-mono uppercase tracking-wider">
                  {isReturningHome ? 'Returning...' : 'Return to Base'}
                </span>
              </button>

              {/* Flight Mode Toggle */}
              <div className="flex items-center gap-1 bg-[var(--bg-tertiary)] rounded-lg p-0.5 border border-[var(--border-dim)]">
                <button
                  onClick={() => { if (!isAutomatic) handleModeSwitch(); }}
                  className={`
                    px-3 py-1.5 rounded-md text-[10px] font-bold uppercase tracking-wider transition-all duration-200
                    flex items-center gap-1.5
                    ${
                      isAutomatic
                        ? 'bg-[var(--zone-red)] text-white shadow-lg shadow-[var(--zone-red)]/20'
                        : 'text-[var(--text-muted)] hover:text-[var(--zone-red)] hover:bg-[var(--zone-red)]/10'
                    }
                  `}
                >
                  <Plane size={12} />
                  <span>{isDroneFlying ? 'Auto Ctrl' : 'Auto Deploy'}</span>
                </button>
                <button
                  onClick={() => { if (!isManual) handleModeSwitch(); }}
                  className={`
                    px-3 py-1.5 rounded-md text-[10px] font-bold uppercase tracking-wider transition-all duration-200
                    flex items-center gap-1.5
                    ${
                      isManual
                        ? 'bg-[var(--zone-green)] text-white shadow-lg shadow-[var(--zone-green)]/20'
                        : 'text-[var(--text-muted)] hover:text-[var(--zone-green)] hover:bg-[var(--zone-green)]/10'
                    }
                  `}
                >
                  <Hand size={12} />
                  <span>{isDroneFlying ? 'Manual Ctrl' : 'Manual Deploy'}</span>
                </button>
              </div>

              {/* Deploy Equipment */}
              <button
                onClick={handleDeployEquipment}
                disabled={equipmentDeployed}
                className={`
                  flex items-center gap-1.5 px-3 py-2 rounded-lg border-2 transition-all duration-300 w-full justify-center
                  ${
                    equipmentDeployed
                      ? 'border-[var(--zone-green)] bg-[var(--zone-green)]/20 text-[var(--zone-green)]'
                      : 'border-[var(--zone-red)]/80 bg-[var(--zone-red)]/15 text-[var(--zone-red)] hover:border-[var(--zone-red)]'
                  }
                `}
              >
                <Package size={16} className={equipmentDeployed ? 'animate-bounce' : ''} />
                <span className="text-[9px] font-bold font-mono uppercase tracking-wider">
                  {equipmentDeployed ? 'Deployed' : 'Drop Equipment'}
                </span>
              </button>
            </div>
          </div>
        </div>
      </footer>

      {/* Error Notification */}
      {errorMessage && (
        <div className="absolute top-16 left-1/2 -translate-x-1/2 flex items-center gap-2 px-3 py-1.5 rounded-lg border border-[var(--zone-red)] bg-[var(--zone-red-fill)] backdrop-blur-sm z-50">
          <AlertTriangle size={14} className="text-[var(--zone-red)]" />
          <span className="text-xs font-mono text-[var(--zone-red)]">
            {errorMessage}
          </span>
        </div>
      )}
    </div>
  );
}

// Control Button Component
interface ControlButtonProps {
  icon: React.ReactNode;
  isActive: boolean;
  isDisabled: boolean;
  onPress: () => void;
  onRelease: () => void;
  label: string;
  variant?: 'movement' | 'altitude';
  labelPosition?: 'above' | 'below';
}

function ControlButton({
  icon,
  isActive,
  isDisabled,
  onPress,
  onRelease,
  label,
  variant = 'movement',
  labelPosition = 'below',
}: ControlButtonProps) {
  const baseColor = variant === 'altitude' ? 'var(--zone-green)' : 'var(--accent-cyan)';

  return (
    <button
      onMouseDown={onPress}
      onMouseUp={onRelease}
      onMouseLeave={onRelease}
      onTouchStart={onPress}
      onTouchEnd={onRelease}
      disabled={isDisabled}
      className={`
        relative w-11 h-11 rounded-lg border-2 transition-all duration-150
        flex items-center justify-center
        ${
          isDisabled
            ? 'border-[var(--border-dim)] bg-[var(--bg-tertiary)] text-[var(--text-muted)] cursor-not-allowed opacity-50'
            : isActive
              ? `border-[${baseColor}] bg-[${baseColor}]/30 text-[${baseColor}]`
              : `border-white/40 bg-[var(--bg-secondary)] text-[var(--text-primary)] hover:border-[${baseColor}] hover:text-[${baseColor}]`
        }
        ${isActive && !isDisabled ? 'scale-95 glow-cyan-subtle' : ''}
      `}
      style={
        isActive && !isDisabled
          ? {
              borderColor: baseColor,
              backgroundColor: `color-mix(in srgb, ${baseColor} 20%, transparent)`,
              color: baseColor,
              boxShadow: `0 0 15px color-mix(in srgb, ${baseColor} 30%, transparent)`,
            }
          : {}
      }
    >
      {icon}
      <span className={`absolute text-[9px] font-mono text-white/60 ${labelPosition === 'above' ? '-top-5' : '-bottom-5'}`}>{label}</span>
    </button>
  );
}
