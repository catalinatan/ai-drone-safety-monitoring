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
    <div className="h-full flex flex-col bg-[var(--bg-primary)] tactical-grid relative overflow-hidden pt-14">
      {/* Top Bar */}
      <header className="absolute top-0 left-0 right-0 z-20 flex items-center justify-between px-6 py-4">
        {/* Left: Home Button */}
        <button
          onClick={handleReturnHome}
          disabled={isReturningHome}
          className={`
            flex items-center gap-2 px-4 py-3 rounded-lg border transition-all duration-300
            ${
              isReturningHome
                ? 'border-[var(--zone-yellow)] bg-[var(--zone-yellow-fill)] text-[var(--zone-yellow)]'
                : 'border-[var(--border-dim)] bg-[var(--bg-secondary)]/90 text-[var(--text-secondary)] hover:border-[var(--accent-cyan)] hover:text-[var(--accent-cyan)]'
            }
            backdrop-blur-sm
          `}
        >
          <Home size={20} className={isReturningHome ? 'animate-pulse' : ''} />
          <span className="text-sm font-semibold uppercase tracking-wider">
            {isReturningHome ? 'Returning...' : 'Return Home'}
          </span>
        </button>

        {/* Center: Status Display */}
        <div className="absolute left-1/2 -translate-x-1/2 flex items-center gap-4">
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

          {/* Mode Display */}
          <div
            className={`
            flex items-center gap-2 px-4 py-2 rounded-lg border backdrop-blur-sm
            ${
              isAutomatic
                ? 'border-[var(--zone-yellow)]/50 bg-[var(--zone-yellow-fill)]'
                : 'border-[var(--accent-cyan)]/50 bg-[var(--accent-cyan-glow)]'
            }
          `}
          >
            {isAutomatic ? (
              <>
                <Plane size={16} className="text-[var(--zone-yellow)]" />
                <span className="text-sm font-bold text-[var(--zone-yellow)] uppercase tracking-wider">
                  Automatic Flight
                </span>
                {status.is_navigating && (
                  <span className="text-xs text-[var(--zone-yellow)] animate-pulse">● NAVIGATING</span>
                )}
              </>
            ) : (
              <>
                <Hand size={16} className="text-[var(--accent-cyan)]" />
                <span className="text-sm font-bold text-[var(--accent-cyan)] uppercase tracking-wider">
                  Manual Control
                </span>
              </>
            )}
          </div>
        </div>

        {/* Right: Deploy Equipment Button */}
        <button
          onClick={handleDeployEquipment}
          disabled={equipmentDeployed}
          className={`
            flex items-center gap-2 px-4 py-3 rounded-lg border transition-all duration-300
            ${
              equipmentDeployed
                ? 'border-[var(--zone-green)] bg-[var(--zone-green-fill)] text-[var(--zone-green)]'
                : 'border-[var(--zone-red)] bg-[var(--zone-red-fill)] text-[var(--zone-red)] hover:bg-[var(--zone-red)]/30'
            }
            backdrop-blur-sm
          `}
        >
          <Package size={20} className={equipmentDeployed ? 'animate-bounce' : ''} />
          <span className="text-sm font-semibold uppercase tracking-wider">
            {equipmentDeployed ? 'Deployed!' : 'Deploy Equipment'}
          </span>
        </button>
      </header>

      {/* Dual Video Feeds - Side by Side (Equal Size) */}
      <main className="flex-1 flex items-center justify-center p-20 gap-6">
        {/* Down Camera (Camera 0) - Ground surveillance */}
        <div className="relative flex-1 max-w-2xl aspect-video rounded-lg overflow-hidden border-2 border-[var(--border-dim)] corner-brackets">
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
      </main>

      {/* Bottom Controls Bar */}
      <footer className="absolute bottom-0 left-0 right-0 z-20 px-6 py-6">
        <div className="flex items-end justify-between">
          {/* Left: Movement Controls (WASD) */}
          <div className="flex flex-col items-center gap-2">
            <span className="text-[10px] font-mono text-[var(--text-muted)] uppercase tracking-wider mb-2">
              Movement
            </span>

            {/* Up */}
            <ControlButton
              icon={<ChevronUp size={24} />}
              isActive={activeControls.has('forward')}
              isDisabled={!isManual}
              onPress={() => handleControlPress('forward')}
              onRelease={() => handleControlRelease('forward')}
              label="W"
            />

            {/* Left, Down, Right row */}
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

          {/* Center: Manual Override Switch */}
          <div className="flex flex-col items-center gap-3">
            <span className="text-[10px] font-mono text-[var(--text-muted)] uppercase tracking-wider">
              Flight Mode
            </span>

            <button
              onClick={handleModeSwitch}
              className={`
                relative w-32 h-14 rounded-full border-2 transition-all duration-300
                ${
                  isManual
                    ? 'border-[var(--accent-cyan)] bg-[var(--accent-cyan-glow)]'
                    : 'border-[var(--zone-yellow)] bg-[var(--zone-yellow-fill)]'
                }
              `}
            >
              {/* Switch knob */}
              <div
                className={`
                  absolute top-1 w-11 h-11 rounded-full transition-all duration-300 flex items-center justify-center
                  ${
                    isManual
                      ? 'left-1 bg-[var(--accent-cyan)] text-[var(--bg-primary)]'
                      : 'left-[calc(100%-48px)] bg-[var(--zone-yellow)] text-[var(--bg-primary)]'
                  }
                `}
              >
                {isManual ? <Hand size={20} /> : <Plane size={20} />}
              </div>

              {/* Labels */}
              <span
                className={`absolute left-3 top-1/2 -translate-y-1/2 text-[10px] font-bold uppercase ${isManual ? 'opacity-0' : 'text-[var(--zone-yellow)]'}`}
              >
                MAN
              </span>
              <span
                className={`absolute right-3 top-1/2 -translate-y-1/2 text-[10px] font-bold uppercase ${isManual ? 'text-[var(--accent-cyan)]' : 'opacity-0'}`}
              >
                AUTO
              </span>
            </button>

            <span
              className={`text-xs font-semibold uppercase tracking-wider ${isManual ? 'text-[var(--accent-cyan)]' : 'text-[var(--zone-yellow)]'}`}
            >
              {isManual ? 'Manual Override Active' : 'Automatic Flight'}
            </span>
          </div>

          {/* Right: Altitude Controls */}
          <div className="flex flex-col items-center gap-2">
            <span className="text-[10px] font-mono text-[var(--text-muted)] uppercase tracking-wider mb-2">
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
        </div>
      </footer>

      {/* Automatic Flight Warning Overlay */}
      {isAutomatic && (
        <div className="absolute bottom-32 left-1/2 -translate-x-1/2 flex items-center gap-2 px-4 py-2 rounded-lg border border-[var(--zone-yellow)] bg-[var(--zone-yellow-fill)] backdrop-blur-sm animate-pulse">
          <AlertTriangle size={16} className="text-[var(--zone-yellow)]" />
          <span className="text-sm font-mono text-[var(--zone-yellow)]">
            Manual controls disabled during automatic flight
          </span>
        </div>
      )}

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
              : `border-[var(--border-dim)] bg-[var(--bg-secondary)] text-[var(--text-secondary)] hover:border-[${baseColor}] hover:text-[${baseColor}]`
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
      <span className="absolute -bottom-5 text-[9px] font-mono text-[var(--text-muted)]">{label}</span>
    </button>
  );
}
