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
} from 'lucide-react';
import { BACKEND_URL } from '../data/mockFeeds';

interface DroneStatus {
  mode: 'manual' | 'automatic';
  connected: boolean;
  is_navigating: boolean;
  returning_home: boolean;
  target_position: [number, number, number] | null;
  pose: { x: number; y: number; z: number } | null;
}

const DRONE_API_BASE = 'http://localhost:8000';

export function DroneControlPanel() {
  const [status, setStatus] = useState<DroneStatus>({
    mode: 'manual',
    connected: false,
    is_navigating: false,
    returning_home: false,
    target_position: null,
    pose: null,
  });
  const [isConnected, setIsConnected] = useState(false);
  const [activeControls, setActiveControls] = useState<Set<string>>(new Set());
  const [isReturningHome, setIsReturningHome] = useState(false);
  const [equipmentDeployed, setEquipmentDeployed] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [triggerInfo, setTriggerInfo] = useState<{ has_snapshot: boolean; feed_id: string | null; timestamp: string | null }>({ has_snapshot: false, feed_id: null, timestamp: null });
  const [triggerImgKey, setTriggerImgKey] = useState(0);
  const videoRef = useRef<HTMLImageElement>(null);
  const lastVelocityRef = useRef({ vx: 0, vy: 0, vz: 0 });
  const [connectedSince, setConnectedSince] = useState<number | null>(null);
  const [flightTime, setFlightTime] = useState('00:00');

  // Fetch drone status periodically
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const response = await fetch(`${DRONE_API_BASE}/status`);
        if (response.ok) {
          const data = await response.json();
          setStatus(data);
          if (!isConnected) setConnectedSince(Date.now());
          setIsConnected(true);
        } else {
          setIsConnected(false);
          setConnectedSince(null);
        }
      } catch {
        setIsConnected(false);
        setConnectedSince(null);
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 1000);
    return () => clearInterval(interval);
  }, []);

  // Flight time ticker
  useEffect(() => {
    if (!connectedSince) { setFlightTime('00:00'); return; }
    const tick = setInterval(() => {
      const elapsed = Math.floor((Date.now() - connectedSince) / 1000);
      const mm = String(Math.floor(elapsed / 60)).padStart(2, '0');
      const ss = String(elapsed % 60).padStart(2, '0');
      setFlightTime(`${mm}:${ss}`);
    }, 1000);
    return () => clearInterval(tick);
  }, [connectedSince]);

  // Poll trigger info from backend
  useEffect(() => {
    const fetchTriggerInfo = async () => {
      try {
        const response = await fetch(`${BACKEND_URL}/trigger-info`);
        if (response.ok) {
          const data = await response.json();
          // If timestamp changed, bump key to reload image
          if (data.has_snapshot && data.timestamp !== triggerInfo.timestamp) {
            setTriggerImgKey((k) => k + 1);
          }
          setTriggerInfo(data);
        }
      } catch {
        // Backend unavailable
      }
    };

    fetchTriggerInfo();
    const interval = setInterval(fetchTriggerInfo, 2000);
    return () => clearInterval(interval);
  }, [triggerInfo.timestamp]);

  // Show error with auto-dismiss
  const showError = useCallback((message: string) => {
    setErrorMessage(message);
    setTimeout(() => setErrorMessage(null), 4000);
  }, []);

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
      // First switch to automatic mode if not already
      if (status.mode !== 'automatic') {
        const modeResponse = await fetch(`${DRONE_API_BASE}/mode`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ mode: 'automatic' }),
        });
        if (!modeResponse.ok) {
          const data = await modeResponse.json();
          showError(data.detail || 'Failed to switch to automatic mode');
          return;
        }
      }

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
        {/* CCTV Trigger Feed - shows the frame that triggered drone deployment */}
        <div className={`relative w-full max-w-sm aspect-video rounded-lg overflow-hidden border-2 ${triggerInfo.has_snapshot ? 'border-[var(--zone-red)]' : 'border-dashed border-[var(--border-dim)]'} corner-brackets flex-shrink-0`}>
          {triggerInfo.has_snapshot ? (
            <img
              key={triggerImgKey}
              src={`${BACKEND_URL}/trigger-snapshot?t=${triggerImgKey}`}
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
          <div className="absolute top-3 left-3 flex items-center gap-2 px-2 py-1 rounded bg-[var(--bg-primary)]/80 border border-[var(--border-dim)]">
            <div className={`w-2 h-2 rounded-full ${triggerInfo.has_snapshot ? 'bg-[var(--zone-red)]' : 'bg-[var(--text-muted)]'}`} />
            <span className="text-[10px] font-mono text-[var(--text-secondary)] tracking-wider">
              {triggerInfo.has_snapshot && triggerInfo.feed_id ? triggerInfo.feed_id.toUpperCase().replace('-', ' ') + ' TRIGGER' : 'TRIGGER CAM'}
            </span>
          </div>
          {triggerInfo.has_snapshot && triggerInfo.timestamp && (
            <div className="absolute bottom-3 right-3 px-2 py-1 rounded bg-[var(--bg-primary)]/80 border border-[var(--border-dim)]">
              <span className="text-[10px] font-mono text-[var(--text-muted)] tracking-wider">
                {new Date(triggerInfo.timestamp).toLocaleTimeString('en-US', { hour12: false })}
              </span>
            </div>
          )}
          <div className="absolute inset-0 scanlines pointer-events-none opacity-50" />
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

          {/* Top row: RTH + Flight Mode Toggle */}
          <div className="flex items-center justify-center gap-4 mb-3">
            {/* Return Home */}
            <button
              onClick={handleReturnHome}
              disabled={isReturningHome}
              className={`
                flex items-center gap-1.5 px-3 py-2 rounded-lg border-2 transition-all duration-300
                ${
                  isReturningHome
                    ? 'border-[var(--zone-yellow)] bg-[var(--zone-yellow)]/20 text-[var(--zone-yellow)]'
                    : 'border-[var(--accent-cyan)]/60 bg-[var(--accent-cyan)]/10 text-[var(--accent-cyan)] hover:border-[var(--accent-cyan)]'
                }
              `}
            >
              <Home size={16} className={isReturningHome ? 'animate-pulse' : ''} />
              <span className="text-[9px] font-bold font-mono uppercase tracking-wider">
                {isReturningHome ? 'RTH...' : 'RTH'}
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
                <span>Auto</span>
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
                <span>Manual</span>
              </button>
            </div>
          </div>

          {/* Bottom row: Movement + Altitude + Deploy Controls */}
          <div className="flex items-center justify-center gap-6">
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

            {/* Deploy Equipment */}
            <div className="flex flex-col items-center gap-1">
              <span className="text-[9px] font-bold font-mono text-white uppercase tracking-wider mb-1">
                Deploy
              </span>
              <button
                onClick={handleDeployEquipment}
                disabled={equipmentDeployed}
                className={`
                  relative w-11 h-11 rounded-lg border-2 transition-all duration-300
                  flex items-center justify-center
                  ${
                    equipmentDeployed
                      ? 'border-[var(--zone-green)] bg-[var(--zone-green)]/20 text-[var(--zone-green)]'
                      : 'border-[var(--zone-red)]/80 bg-[var(--zone-red)]/15 text-[var(--zone-red)] hover:border-[var(--zone-red)]'
                  }
                `}
              >
                <Package size={18} className={equipmentDeployed ? 'animate-bounce' : ''} />
              </button>
              <span className="text-[9px] font-mono text-white/60">
                {equipmentDeployed ? 'SENT' : 'DROP'}
              </span>
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
