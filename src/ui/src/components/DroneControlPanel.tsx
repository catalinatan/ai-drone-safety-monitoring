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
} from 'lucide-react';

interface DroneStatus {
  mode: 'manual' | 'automatic';
  connected: boolean;
  is_navigating: boolean;
  target_position: [number, number, number] | null;
}

const DRONE_API_BASE = 'http://localhost:8000';

export function DroneControlPanel() {
  const [status, setStatus] = useState<DroneStatus>({
    mode: 'manual',
    connected: false,
    is_navigating: false,
    target_position: null,
  });
  const [isConnected, setIsConnected] = useState(false);
  const [activeControls, setActiveControls] = useState<Set<string>>(new Set());
  const [isReturningHome, setIsReturningHome] = useState(false);
  const [equipmentDeployed, setEquipmentDeployed] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const videoRef = useRef<HTMLImageElement>(null);
  const lastVelocityRef = useRef({ vx: 0, vy: 0, vz: 0 });

  // Fetch drone status periodically
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const response = await fetch(`${DRONE_API_BASE}/status`);
        if (response.ok) {
          const data = await response.json();
          setStatus(data);
          setIsConnected(true);
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
        setStatus((prev) => ({ ...prev, mode: 'automatic', is_navigating: true }));
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

  // Reset returning home state when navigation completes
  useEffect(() => {
    if (!status.is_navigating && isReturningHome) {
      setIsReturningHome(false);
    }
  }, [status.is_navigating, isReturningHome]);

  const isManual = status.mode === 'manual';
  const isAutomatic = status.mode === 'automatic';

  return (
    <div className="h-full flex flex-col bg-[var(--bg-primary)] tactical-grid relative overflow-hidden pt-28">
      {/* Top Bar - Connection & Navigation Status */}
      <header className="absolute top-14 left-0 right-0 z-20 flex items-center justify-center px-6 py-3">
        <div className="flex items-center gap-4">
          {/* Connection Status */}
          <div
            className={`
            flex items-center gap-2 px-3 py-2 rounded-lg border backdrop-blur-sm
            ${
              isConnected
                ? 'border-[var(--zone-green)]/50 bg-[var(--zone-green-fill)]'
                : 'border-[var(--zone-red)]/50 bg-[var(--zone-red-fill)]'
            }
          `}
          >
            <Radio
              size={14}
              className={isConnected ? 'text-[var(--zone-green)] status-live' : 'text-[var(--zone-red)]'}
            />
            <span
              className={`text-xs font-mono ${isConnected ? 'text-[var(--zone-green)]' : 'text-[var(--zone-red)]'}`}
            >
              {isConnected ? 'CONNECTED' : 'DISCONNECTED'}
            </span>
          </div>

          {/* Navigation Status */}
          {isAutomatic && status.is_navigating && (
            <div className="flex items-center gap-2 px-3 py-2 rounded-lg border border-[var(--zone-yellow)]/50 bg-[var(--zone-yellow-fill)] backdrop-blur-sm">
              <Plane size={14} className="text-[var(--zone-yellow)] animate-pulse" />
              <span className="text-xs font-mono text-[var(--zone-yellow)]">NAVIGATING TO TARGET</span>
            </div>
          )}
        </div>
      </header>

      {/* Main Content - CCTV Trigger + Drone Feeds */}
      <main className="flex-1 flex flex-col items-center justify-center px-6 pt-4 pb-56 gap-4 min-h-0">
        {/* CCTV Trigger Feed - shows the frame that triggered drone deployment */}
        <div className="relative w-full max-w-sm aspect-video rounded-lg overflow-hidden border-2 border-dashed border-[var(--border-dim)] corner-brackets flex-shrink-0">
          <div className="w-full h-full flex flex-col items-center justify-center bg-[var(--bg-card)] gap-2">
            <Camera size={28} className="text-[var(--text-muted)]" />
            <span className="text-xs font-mono text-[var(--text-secondary)] tracking-wider">CCTV TRIGGER FEED</span>
            <span className="text-[10px] font-mono text-[var(--text-muted)]">Awaiting hazard detection...</span>
          </div>
          <div className="absolute top-3 left-3 flex items-center gap-2 px-2 py-1 rounded bg-[var(--bg-primary)]/80 border border-[var(--border-dim)]">
            <div className="w-2 h-2 rounded-full bg-[var(--text-muted)]" />
            <span className="text-[10px] font-mono text-[var(--text-secondary)] tracking-wider">TRIGGER CAM</span>
          </div>
          <div className="absolute inset-0 scanlines pointer-events-none opacity-50" />
        </div>

        {/* Drone Feeds - Side by Side */}
        <div className="flex items-center justify-center gap-6 w-full flex-1 min-h-0">
        {/* Down Camera (Camera 0) - Ground surveillance */}
        <div className="relative flex-1 max-w-2xl h-full rounded-lg overflow-hidden border-2 border-[var(--border-dim)] corner-brackets">
          {/* Video Feed */}
          {isConnected ? (
            <img
              ref={videoRef}
              src={`${DRONE_API_BASE}/video_feed/down`}
              alt="Drone Down Camera"
              className="w-full h-full object-cover"
            />
          ) : (
            <div className="w-full h-full bg-[var(--bg-card)] flex flex-col items-center justify-center gap-4">
              <AlertTriangle size={48} className="text-[var(--zone-yellow)]" />
              <span className="text-lg font-mono text-[var(--text-secondary)]">NO VIDEO SIGNAL</span>
              <span className="text-sm text-[var(--text-muted)]">Waiting for drone connection...</span>
            </div>
          )}

          {/* Scanline overlay */}
          <div className="absolute inset-0 scanlines pointer-events-none" />

          {/* Crosshair overlay */}
          <div className="absolute inset-0 pointer-events-none">
            <div className="absolute top-1/2 left-0 right-0 h-px bg-[var(--accent-cyan)]/20" />
            <div className="absolute left-1/2 top-0 bottom-0 w-px bg-[var(--accent-cyan)]/20" />
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-16 h-16 border border-[var(--accent-cyan)]/30 rounded-full" />
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 border border-[var(--accent-cyan)]/50 rounded-full" />
          </div>

          {/* Camera label */}
          <div className="absolute top-4 left-4 flex items-center gap-2 px-2 py-1 rounded bg-[var(--bg-primary)]/80 border border-[var(--border-dim)]">
            <div
              className={`w-2 h-2 rounded-full ${isAutomatic ? 'bg-[var(--zone-yellow)]' : 'bg-[var(--zone-green)]'} status-live`}
            />
            <span className="text-[10px] font-mono text-[var(--text-secondary)] tracking-wider">
              FORWARD CAM
            </span>
          </div>

          {/* Timestamp */}
          <div className="absolute bottom-4 right-4 px-2 py-1 rounded bg-[var(--bg-primary)]/80 border border-[var(--border-dim)]">
            <span className="text-[10px] font-mono text-[var(--text-muted)] tracking-wider">
              {new Date().toLocaleTimeString('en-US', { hour12: false })} UTC
            </span>
          </div>
        </div>

        {/* Forward Camera (Camera 3) - Navigation view */}
        <div className="relative flex-1 max-w-2xl aspect-video rounded-lg overflow-hidden border-2 border-[var(--border-dim)] corner-brackets">
          {/* Video Feed */}
          {isConnected ? (
            <img
              src={`${DRONE_API_BASE}/video_feed/forward`}
              alt="Drone Forward Camera"
              className="w-full h-full object-cover"
            />
          ) : (
            <div className="w-full h-full bg-[var(--bg-card)] flex flex-col items-center justify-center gap-4">
              <AlertTriangle size={48} className="text-[var(--zone-yellow)]" />
              <span className="text-lg font-mono text-[var(--text-secondary)]">NO VIDEO SIGNAL</span>
              <span className="text-sm text-[var(--text-muted)]">Waiting for drone connection...</span>
            </div>
          )}

          {/* Scanline overlay */}
          <div className="absolute inset-0 scanlines pointer-events-none" />

          {/* Crosshair overlay */}
          <div className="absolute inset-0 pointer-events-none">
            <div className="absolute top-1/2 left-0 right-0 h-px bg-[var(--accent-cyan)]/20" />
            <div className="absolute left-1/2 top-0 bottom-0 w-px bg-[var(--accent-cyan)]/20" />
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-16 h-16 border border-[var(--accent-cyan)]/30 rounded-full" />
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 border border-[var(--accent-cyan)]/50 rounded-full" />
          </div>

          {/* Camera label */}
          <div className="absolute top-4 left-4 flex items-center gap-2 px-2 py-1 rounded bg-[var(--bg-primary)]/80 border border-[var(--border-dim)]">
            <div
              className={`w-2 h-2 rounded-full ${isConnected ? 'bg-[var(--zone-green)]' : 'bg-[var(--zone-red)]'} status-live`}
            />
            <span className="text-[10px] font-mono text-[var(--text-secondary)] tracking-wider">
              DOWN CAM
            </span>
          </div>

          {/* Timestamp */}
          <div className="absolute bottom-4 right-4 px-2 py-1 rounded bg-[var(--bg-primary)]/80 border border-[var(--border-dim)]">
            <span className="text-[10px] font-mono text-[var(--text-muted)] tracking-wider">
              {new Date().toLocaleTimeString('en-US', { hour12: false })} UTC
            </span>
          </div>
        </div>
        </div>
      </main>

      {/* Bottom Controls Bar */}
      <footer className="absolute bottom-0 left-0 right-0 z-20 px-5 py-3">
        <div className="relative flex items-end justify-between rounded-xl border border-[var(--accent-cyan)]/30 bg-[var(--bg-primary)]/80 backdrop-blur-xl px-6 py-5 shadow-[0_0_20px_rgba(0,0,0,0.5),inset_0_1px_0_rgba(255,255,255,0.05)]">
          {/* Container label */}
          <div className="absolute -top-3 left-6 px-4 py-1 rounded-md bg-[var(--bg-primary)] border border-[var(--accent-cyan)]/40 shadow-[0_0_8px_var(--accent-cyan-glow)]">
            <span className="text-xs font-bold font-mono text-[var(--accent-cyan)] uppercase tracking-widest">Drone Control</span>
          </div>

          {/* Auto-flight warning - inside the control container */}
          {isAutomatic && (
            <div className="absolute -top-3 right-6 flex items-center gap-2 px-3 py-1 rounded-md bg-[var(--bg-primary)] border border-[var(--zone-yellow)] animate-pulse">
              <AlertTriangle size={12} className="text-[var(--zone-yellow)]" />
              <span className="text-xs font-bold font-mono text-[var(--zone-yellow)]">
                Manual controls disabled during auto flight
              </span>
            </div>
          )}

          {/* Left: Return Home + Movement Controls (WASD) */}
          <div className="flex items-end gap-4">
            <button
              onClick={handleReturnHome}
              disabled={isReturningHome}
              className={`
                flex flex-col items-center gap-1.5 px-5 py-3.5 rounded-lg border-2 transition-all duration-300
                ${
                  isReturningHome
                    ? 'border-[var(--zone-yellow)] bg-[var(--zone-yellow)]/20 text-[var(--zone-yellow)] shadow-[0_0_12px_var(--zone-yellow-fill)]'
                    : 'border-[var(--accent-cyan)]/60 bg-[var(--accent-cyan)]/10 text-[var(--accent-cyan)] hover:border-[var(--accent-cyan)] hover:bg-[var(--accent-cyan)]/20 hover:shadow-[0_0_12px_var(--accent-cyan-glow)]'
                }
                backdrop-blur-sm
              `}
            >
              <Home size={22} className={isReturningHome ? 'animate-pulse' : ''} />
              <span className="text-[10px] font-bold font-mono uppercase tracking-wider">
                {isReturningHome ? 'RTH...' : 'Back to Base'}
              </span>
            </button>

            <div className="flex flex-col items-center gap-2">
              <span className="text-[10px] font-bold font-mono text-white uppercase tracking-wider mb-2">
                Movement
              </span>

              <ControlButton
                icon={<ChevronUp size={24} />}
                isActive={activeControls.has('forward')}
                isDisabled={!isManual}
                onPress={() => handleControlPress('forward')}
                onRelease={() => handleControlRelease('forward')}
                label="W"
              />

              <div className="flex gap-2">
                <ControlButton
                  icon={<ChevronLeft size={24} />}
                  isActive={activeControls.has('left')}
                  isDisabled={!isManual}
                  onPress={() => handleControlPress('left')}
                  onRelease={() => handleControlRelease('left')}
                  label="A"
                />
                <ControlButton
                  icon={<ChevronDown size={24} />}
                  isActive={activeControls.has('backward')}
                  isDisabled={!isManual}
                  onPress={() => handleControlPress('backward')}
                  onRelease={() => handleControlRelease('backward')}
                  label="S"
                />
                <ControlButton
                  icon={<ChevronRight size={24} />}
                  isActive={activeControls.has('right')}
                  isDisabled={!isManual}
                  onPress={() => handleControlPress('right')}
                  onRelease={() => handleControlRelease('right')}
                  label="D"
                />
              </div>
            </div>
          </div>

          {/* Center: Auto / Manual Flight Mode Tabs */}
          <div className="absolute left-1/2 -translate-x-1/2 bottom-0 flex flex-col items-center gap-2">
            <span className="text-[10px] font-bold font-mono text-white uppercase tracking-wider">
              Flight Mode
            </span>
            <span className="text-[11px] font-mono text-white/70 max-w-[260px] text-center leading-relaxed">
              Select Manual to override automatic flight. Use WASD to steer, Z/X for altitude.
            </span>

            <div className="flex items-center gap-1 bg-[var(--bg-tertiary)] rounded-lg p-1 border border-[var(--border-dim)]">
              {/* Auto Tab (red) */}
              <button
                onClick={() => { if (!isAutomatic) handleModeSwitch(); }}
                className={`
                  px-5 py-2.5 rounded-md text-sm font-bold uppercase tracking-wider transition-all duration-200
                  flex items-center gap-2
                  ${
                    isAutomatic
                      ? 'bg-[var(--zone-red)] text-white shadow-lg shadow-[var(--zone-red)]/20'
                      : 'text-[var(--text-muted)] hover:text-[var(--zone-red)] hover:bg-[var(--zone-red)]/10'
                  }
                `}
              >
                <Plane size={16} />
                <span>Auto</span>
              </button>

              {/* Manual Tab (green) with tooltip */}
              <div className="relative group">
                <button
                  onClick={() => { if (!isManual) handleModeSwitch(); }}
                  className={`
                    px-5 py-2.5 rounded-md text-sm font-bold uppercase tracking-wider transition-all duration-200
                    flex items-center gap-2
                    ${
                      isManual
                        ? 'bg-[var(--zone-green)] text-white shadow-lg shadow-[var(--zone-green)]/20'
                        : 'text-[var(--text-muted)] hover:text-[var(--zone-green)] hover:bg-[var(--zone-green)]/10'
                    }
                  `}
                >
                  <Hand size={16} />
                  <span>Manual</span>
                </button>

                {/* Tooltip - appears on hover when not already in manual mode */}
                {!isManual && (
                  <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-3 px-3 py-2 rounded-lg bg-[var(--bg-primary)] border border-[var(--border-dim)] shadow-xl opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none whitespace-nowrap z-50">
                    <span className="text-xs font-mono text-[var(--text-secondary)]">Click to override auto flight to manual</span>
                    <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-[var(--border-dim)]" />
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Right: Altitude Controls + Deploy Equipment */}
          <div className="flex items-end gap-4">
            <div className="flex flex-col items-center gap-2">
              <span className="text-[10px] font-bold font-mono text-white uppercase tracking-wider mb-2">
                Altitude
              </span>

              <ControlButton
                icon={<ArrowUp size={24} />}
                isActive={activeControls.has('up')}
                isDisabled={!isManual}
                onPress={() => handleControlPress('up')}
                onRelease={() => handleControlRelease('up')}
                label="Z"
                variant="altitude"
              />

              <ControlButton
                icon={<ArrowDown size={24} />}
                isActive={activeControls.has('down')}
                isDisabled={!isManual}
                onPress={() => handleControlPress('down')}
                onRelease={() => handleControlRelease('down')}
                label="X"
                variant="altitude"
              />
            </div>

            <button
              onClick={handleDeployEquipment}
              disabled={equipmentDeployed}
              className={`
                flex flex-col items-center gap-1.5 px-5 py-3.5 rounded-lg border-2 transition-all duration-300
                ${
                  equipmentDeployed
                    ? 'border-[var(--zone-green)] bg-[var(--zone-green)]/20 text-[var(--zone-green)] shadow-[0_0_12px_var(--zone-green-fill)]'
                    : 'border-[var(--zone-red)]/80 bg-[var(--zone-red)]/15 text-[var(--zone-red)] hover:border-[var(--zone-red)] hover:bg-[var(--zone-red)]/25 hover:shadow-[0_0_12px_var(--zone-red-fill)]'
                }
                backdrop-blur-sm
              `}
            >
              <Package size={22} className={equipmentDeployed ? 'animate-bounce' : ''} />
              <span className="text-[10px] font-bold font-mono uppercase tracking-wider">
                {equipmentDeployed ? 'Sent!' : 'Deploy Equipment'}
              </span>
            </button>
          </div>
        </div>
      </footer>


      {/* Error Notification */}
      {errorMessage && (
        <div className="absolute top-20 left-1/2 -translate-x-1/2 flex items-center gap-2 px-4 py-2 rounded-lg border border-[var(--zone-red)] bg-[var(--zone-red-fill)] backdrop-blur-sm z-50">
          <AlertTriangle size={16} className="text-[var(--zone-red)]" />
          <span className="text-sm font-mono text-[var(--zone-red)]">
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
}

function ControlButton({
  icon,
  isActive,
  isDisabled,
  onPress,
  onRelease,
  label,
  variant = 'movement',
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
        relative w-14 h-14 rounded-lg border-2 transition-all duration-150
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
      <span className="absolute -bottom-5 text-[9px] font-mono text-white/60">{label}</span>
    </button>
  );
}
