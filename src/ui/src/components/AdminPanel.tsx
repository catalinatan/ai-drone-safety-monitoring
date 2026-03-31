import { useState, useEffect, useCallback } from 'react';
import { ArrowLeft, Save, Loader2, RefreshCw, AlertTriangle, Check, Activity } from 'lucide-react';
import { BACKEND_URL } from '../data/mockFeeds';

interface AdminPanelProps {
  onBack: () => void;
}

interface AuditEvent {
  type: string;
  timestamp: string;
  feed_id?: string;
  details?: Record<string, unknown>;
  [key: string]: unknown;
}

// Config section definitions — maps YAML structure to form fields
interface FieldDef {
  key: string;
  label: string;
  type: 'number' | 'string' | 'boolean';
  step?: number;
  min?: number;
  max?: number;
  hint?: string;
}

interface SectionDef {
  key: string;
  label: string;
  fields: FieldDef[];
}

const CONFIG_SECTIONS: SectionDef[] = [
  {
    key: 'detection',
    label: 'Human Detection',
    fields: [
      { key: 'confidence_threshold', label: 'Confidence', type: 'number', step: 0.05, min: 0.05, max: 1.0, hint: 'YOLO detection confidence (0-1)' },
      { key: 'inference_imgsz', label: 'Inference Size', type: 'number', step: 32, min: 320, max: 1920, hint: 'Input resolution (px)' },
      { key: 'fps', label: 'Detection FPS', type: 'number', step: 1, min: 1, max: 60, hint: 'Detection loop frequency' },
      { key: 'warmup_frames', label: 'Warmup Frames', type: 'number', step: 5, min: 0, max: 200, hint: 'Ignore alarms for N initial frames' },
    ],
  },
  {
    key: 'zones',
    label: 'Zone Detection',
    fields: [
      { key: 'overlap_threshold', label: 'Overlap Threshold', type: 'number', step: 0.05, min: 0.1, max: 1.0, hint: 'Person-zone overlap ratio to trigger' },
      { key: 'alarm_cooldown_seconds', label: 'Alarm Cooldown (s)', type: 'number', step: 1, min: 0, max: 60, hint: 'Minimum seconds between alarms' },
    ],
  },
  {
    key: 'auto_segmentation',
    label: 'Auto-Segmentation',
    fields: [
      { key: 'interval_seconds', label: 'Interval (s)', type: 'number', step: 10, min: 10, max: 600, hint: 'Re-segment interval for ship scenes' },
      { key: 'confidence', label: 'Confidence', type: 'number', step: 0.05, min: 0.1, max: 1.0, hint: 'Segmentation model confidence' },
      { key: 'simplify_epsilon', label: 'Polygon Simplify', type: 'number', step: 0.5, min: 0.5, max: 10, hint: 'Polygon approximation tolerance (px)' },
      { key: 'min_contour_area', label: 'Min Contour Area', type: 'number', step: 10, min: 10, max: 500, hint: 'Ignore contours smaller than this (px²)' },
    ],
  },
  {
    key: 'streaming',
    label: 'Streaming',
    fields: [
      { key: 'capture_fps', label: 'Capture FPS', type: 'number', step: 5, min: 1, max: 60, hint: 'Frame grab rate from cameras' },
      { key: 'stream_fps', label: 'Stream FPS', type: 'number', step: 5, min: 1, max: 60, hint: 'MJPEG output stream rate' },
    ],
  },
  {
    key: 'drone',
    label: 'Drone',
    fields: [
      { key: 'api_url', label: 'API URL', type: 'string', hint: 'Drone control server URL' },
      { key: 'api_timeout', label: 'API Timeout (s)', type: 'number', step: 1, min: 1, max: 30 },
      { key: 'safe_altitude', label: 'Safe Altitude (NED)', type: 'number', step: 1, min: -50, max: 0, hint: 'Negative = above ground in NED' },
    ],
  },
  {
    key: 'follow_mode',
    label: 'Follow Mode',
    fields: [
      { key: 'target', label: 'Target', type: 'string', hint: '"ship", "railway", "bridge", or empty' },
      { key: 'hover_drones', label: 'Hover Drones', type: 'boolean', hint: 'Keep CCTV drones hovering' },
      { key: 'hover_altitude', label: 'Hover Altitude', type: 'number', step: 1, min: -50, max: 0 },
      { key: 'follow_interval', label: 'Follow Interval (s)', type: 'number', step: 0.01, min: 0.01, max: 1.0, hint: 'Teleport loop tick rate' },
    ],
  },
];

function getNestedValue(obj: Record<string, unknown>, path: string[]): unknown {
  let current: unknown = obj;
  for (const key of path) {
    if (current == null || typeof current !== 'object') return undefined;
    current = (current as Record<string, unknown>)[key];
  }
  return current;
}

function setNestedValue(obj: Record<string, unknown>, path: string[], value: unknown): Record<string, unknown> {
  const result = { ...obj };
  if (path.length === 1) {
    result[path[0]] = value;
    return result;
  }
  const [head, ...tail] = path;
  result[head] = setNestedValue(
    (result[head] as Record<string, unknown>) ?? {},
    tail,
    value,
  );
  return result;
}

export function AdminPanel({ onBack }: AdminPanelProps) {
  const [config, setConfig] = useState<Record<string, unknown> | null>(null);
  const [originalConfig, setOriginalConfig] = useState<Record<string, unknown> | null>(null);
  const [events, setEvents] = useState<AuditEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [saveState, setSaveState] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle');
  const [activeTab, setActiveTab] = useState<'config' | 'events'>('config');

  const fetchConfig = useCallback(async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/config`);
      if (res.ok) {
        const data = await res.json();
        setConfig(data);
        setOriginalConfig(JSON.parse(JSON.stringify(data)));
      }
    } catch {
      // Backend unavailable
    }
  }, []);

  const fetchEvents = useCallback(async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/events?limit=50`);
      if (res.ok) {
        const data = await res.json();
        setEvents(data.events ?? []);
      }
    } catch {
      // Backend unavailable
    }
  }, []);

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      await Promise.all([fetchConfig(), fetchEvents()]);
      setLoading(false);
    };
    load();
  }, [fetchConfig, fetchEvents]);

  const handleFieldChange = (sectionKey: string, fieldKey: string, value: unknown) => {
    if (!config) return;
    setConfig(setNestedValue(config, [sectionKey, fieldKey], value));
  };

  const hasChanges = config && originalConfig
    ? JSON.stringify(config) !== JSON.stringify(originalConfig)
    : false;

  const handleSave = async () => {
    if (!config) return;
    setSaveState('saving');
    try {
      const res = await fetch(`${BACKEND_URL}/config`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      });
      if (res.ok) {
        const data = await res.json();
        setConfig(data.config);
        setOriginalConfig(JSON.parse(JSON.stringify(data.config)));
        setSaveState('saved');
      } else {
        setSaveState('error');
      }
    } catch {
      setSaveState('error');
    }
    setTimeout(() => setSaveState('idle'), 2500);
  };

  const handleReset = () => {
    if (originalConfig) {
      setConfig(JSON.parse(JSON.stringify(originalConfig)));
    }
  };

  const renderField = (section: SectionDef, field: FieldDef) => {
    if (!config) return null;
    const value = getNestedValue(config, [section.key, field.key]);

    if (field.type === 'boolean') {
      const checked = Boolean(value);
      return (
        <div key={field.key} className="flex items-center justify-between py-2">
          <div className="flex-1 min-w-0">
            <span className="text-xs font-mono text-[var(--text-primary)]">{field.label}</span>
            {field.hint && (
              <p className="text-[9px] font-mono text-[var(--text-muted)] mt-0.5">{field.hint}</p>
            )}
          </div>
          <button
            onClick={() => handleFieldChange(section.key, field.key, !checked)}
            className={`
              flex items-center gap-2 px-2 py-1 rounded-lg border transition-all duration-200 ml-3
              ${checked
                ? 'border-[var(--zone-green)] bg-[var(--zone-green)]/10 text-[var(--zone-green)]'
                : 'border-[var(--border-dim)] bg-[var(--bg-tertiary)] text-[var(--text-muted)]'
              }
            `}
          >
            <div className={`
              w-7 h-3.5 rounded-full transition-colors duration-200 relative
              ${checked ? 'bg-[var(--zone-green)]' : 'bg-[var(--bg-primary)]'}
            `}>
              <div className={`
                absolute top-0.5 w-2.5 h-2.5 rounded-full bg-white transition-transform duration-200
                ${checked ? 'translate-x-3.5' : 'translate-x-0.5'}
              `} />
            </div>
            <span className="text-[10px] font-mono">{checked ? 'On' : 'Off'}</span>
          </button>
        </div>
      );
    }

    if (field.type === 'string') {
      return (
        <div key={field.key} className="py-2">
          <label className="block text-xs font-mono text-[var(--text-primary)] mb-1">{field.label}</label>
          <input
            type="text"
            value={String(value ?? '')}
            onChange={(e) => handleFieldChange(section.key, field.key, e.target.value)}
            className="w-full px-2 py-1.5 rounded border border-[var(--border-dim)] bg-[var(--bg-tertiary)] text-xs font-mono text-[var(--text-primary)] focus:outline-none focus:border-[var(--accent-cyan)] transition-colors"
          />
          {field.hint && (
            <p className="text-[9px] font-mono text-[var(--text-muted)] mt-0.5">{field.hint}</p>
          )}
        </div>
      );
    }

    // number
    return (
      <div key={field.key} className="py-2">
        <div className="flex items-center justify-between mb-1">
          <label className="text-xs font-mono text-[var(--text-primary)]">{field.label}</label>
          <span className="text-[10px] font-mono text-[var(--accent-cyan)]">{String(value ?? '')}</span>
        </div>
        <input
          type="number"
          value={value != null ? Number(value) : ''}
          onChange={(e) => {
            const v = e.target.value === '' ? '' : Number(e.target.value);
            handleFieldChange(section.key, field.key, v);
          }}
          step={field.step}
          min={field.min}
          max={field.max}
          className="w-full px-2 py-1.5 rounded border border-[var(--border-dim)] bg-[var(--bg-tertiary)] text-xs font-mono text-[var(--text-primary)] focus:outline-none focus:border-[var(--accent-cyan)] transition-colors"
        />
        {field.hint && (
          <p className="text-[9px] font-mono text-[var(--text-muted)] mt-0.5">{field.hint}</p>
        )}
      </div>
    );
  };

  const formatTimestamp = (ts: string) => {
    try {
      const d = new Date(ts);
      return d.toLocaleTimeString('en-US', { hour12: false }) + '.' + String(d.getMilliseconds()).padStart(3, '0');
    } catch {
      return ts;
    }
  };

  if (loading) {
    return (
      <div className="h-full flex items-center justify-center bg-[var(--bg-primary)] tactical-grid">
        <div className="flex items-center gap-3 text-[var(--accent-cyan)]">
          <Loader2 className="animate-spin" size={20} />
          <span className="text-sm font-mono">Loading configuration...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col bg-[var(--bg-primary)] tactical-grid">
      {/* Header */}
      <header className="flex items-center justify-between px-4 py-3 border-b border-[var(--border-dim)] bg-[var(--bg-secondary)]/80 backdrop-blur-sm">
        <div className="flex items-center gap-3">
          <button
            onClick={onBack}
            className="flex items-center gap-1.5 px-2 py-1 rounded border border-[var(--border-dim)] bg-[var(--bg-tertiary)] text-[var(--text-muted)] hover:text-[var(--accent-cyan)] hover:border-[var(--accent-cyan)] transition-all duration-200"
          >
            <ArrowLeft size={12} />
            <span className="text-[10px] font-bold font-mono uppercase tracking-wider">Back</span>
          </button>
          <div className="h-4 w-px bg-[var(--border-dim)]" />
          <h1 className="text-sm font-bold tracking-wider uppercase text-glow-cyan">
            System Configuration
          </h1>
        </div>

        <div className="flex items-center gap-2">
          {/* Tab switcher */}
          <div className="flex rounded border border-[var(--border-dim)] overflow-hidden">
            <button
              onClick={() => setActiveTab('config')}
              className={`
                px-3 py-1 text-[10px] font-bold font-mono uppercase tracking-wider transition-all
                ${activeTab === 'config'
                  ? 'bg-[var(--accent-cyan)]/10 text-[var(--accent-cyan)] border-r border-[var(--border-dim)]'
                  : 'bg-[var(--bg-tertiary)] text-[var(--text-muted)] hover:text-[var(--text-secondary)] border-r border-[var(--border-dim)]'
                }
              `}
            >
              Config
            </button>
            <button
              onClick={() => { setActiveTab('events'); fetchEvents(); }}
              className={`
                px-3 py-1 text-[10px] font-bold font-mono uppercase tracking-wider transition-all
                ${activeTab === 'events'
                  ? 'bg-[var(--accent-cyan)]/10 text-[var(--accent-cyan)]'
                  : 'bg-[var(--bg-tertiary)] text-[var(--text-muted)] hover:text-[var(--text-secondary)]'
                }
              `}
            >
              Events ({events.length})
            </button>
          </div>

          {activeTab === 'config' && (
            <>
              <button
                onClick={handleReset}
                disabled={!hasChanges}
                className={`
                  flex items-center gap-1.5 px-2 py-1 rounded border text-[10px] font-bold font-mono uppercase tracking-wider transition-all
                  ${hasChanges
                    ? 'border-[var(--zone-yellow)] bg-[var(--zone-yellow)]/10 text-[var(--zone-yellow)] hover:bg-[var(--zone-yellow)]/20'
                    : 'border-[var(--border-dim)] bg-[var(--bg-tertiary)] text-[var(--text-muted)] cursor-not-allowed opacity-50'
                  }
                `}
              >
                <RefreshCw size={10} />
                Reset
              </button>
              <button
                onClick={handleSave}
                disabled={!hasChanges || saveState === 'saving'}
                className={`
                  flex items-center gap-1.5 px-2 py-1 rounded border text-[10px] font-bold font-mono uppercase tracking-wider transition-all
                  ${saveState === 'saved'
                    ? 'border-[var(--zone-green)] bg-[var(--zone-green)]/10 text-[var(--zone-green)]'
                    : saveState === 'error'
                    ? 'border-[var(--zone-red)] bg-[var(--zone-red)]/10 text-[var(--zone-red)]'
                    : hasChanges && saveState !== 'saving'
                    ? 'border-[var(--accent-cyan)] bg-[var(--accent-cyan)]/10 text-[var(--accent-cyan)] hover:bg-[var(--accent-cyan)]/20'
                    : 'border-[var(--border-dim)] bg-[var(--bg-tertiary)] text-[var(--text-muted)] cursor-not-allowed opacity-50'
                  }
                `}
              >
                {saveState === 'saving' ? <Loader2 size={10} className="animate-spin" /> :
                 saveState === 'saved' ? <Check size={10} /> :
                 saveState === 'error' ? <AlertTriangle size={10} /> :
                 <Save size={10} />}
                {saveState === 'saving' ? 'Saving...' :
                 saveState === 'saved' ? 'Saved' :
                 saveState === 'error' ? 'Error' :
                 'Save'}
              </button>
            </>
          )}

          {activeTab === 'events' && (
            <button
              onClick={fetchEvents}
              className="flex items-center gap-1.5 px-2 py-1 rounded border border-[var(--border-dim)] bg-[var(--bg-tertiary)] text-[var(--text-muted)] hover:text-[var(--accent-cyan)] hover:border-[var(--accent-cyan)] transition-all text-[10px] font-bold font-mono uppercase tracking-wider"
            >
              <RefreshCw size={10} />
              Refresh
            </button>
          )}
        </div>
      </header>

      {/* Content */}
      <main className="flex-1 min-h-0 overflow-y-auto p-4">
        {activeTab === 'config' && config && (
          <div className="max-w-3xl mx-auto space-y-4">
            {CONFIG_SECTIONS.map((section) => (
              <div
                key={section.key}
                className="rounded-lg border border-[var(--border-dim)] bg-[var(--bg-secondary)]/60 overflow-hidden"
              >
                <div className="px-4 py-2.5 border-b border-[var(--border-dim)] bg-[var(--bg-tertiary)]/50">
                  <h2 className="text-[10px] font-bold font-mono uppercase tracking-widest text-[var(--accent-cyan)]">
                    {section.label}
                  </h2>
                </div>
                <div className="px-4 py-1 divide-y divide-[var(--border-dim)]/50">
                  {section.fields.map((field) => renderField(section, field))}
                </div>
              </div>
            ))}
          </div>
        )}

        {activeTab === 'events' && (
          <div className="max-w-3xl mx-auto">
            {events.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-16 text-[var(--text-muted)]">
                <Activity size={32} className="mb-3 opacity-30" />
                <span className="text-sm font-mono">No audit events recorded</span>
              </div>
            ) : (
              <div className="space-y-1">
                {events.map((event, i) => (
                  <div
                    key={i}
                    className="flex items-start gap-3 px-3 py-2 rounded border border-[var(--border-dim)]/50 bg-[var(--bg-secondary)]/40 hover:bg-[var(--bg-secondary)]/60 transition-colors"
                  >
                    {/* Timestamp */}
                    <span className="text-[10px] font-mono text-[var(--text-muted)] whitespace-nowrap pt-0.5">
                      {event.timestamp ? formatTimestamp(event.timestamp) : '--:--:--'}
                    </span>

                    {/* Type badge */}
                    <span className={`
                      text-[9px] font-bold font-mono uppercase tracking-wider px-1.5 py-0.5 rounded whitespace-nowrap
                      ${event.type?.includes('ALARM')
                        ? 'bg-[var(--zone-red)]/15 text-[var(--zone-red)] border border-[var(--zone-red)]/30'
                        : event.type?.includes('ZONE')
                        ? 'bg-[var(--zone-yellow)]/15 text-[var(--zone-yellow)] border border-[var(--zone-yellow)]/30'
                        : 'bg-[var(--accent-cyan)]/10 text-[var(--accent-cyan)] border border-[var(--accent-cyan)]/30'
                      }
                    `}>
                      {event.type ?? 'EVENT'}
                    </span>

                    {/* Feed ID */}
                    {event.feed_id && (
                      <span className="text-[10px] font-mono text-[var(--text-secondary)] whitespace-nowrap pt-0.5">
                        {event.feed_id}
                      </span>
                    )}

                    {/* Details */}
                    <span className="text-[10px] font-mono text-[var(--text-muted)] truncate pt-0.5">
                      {event.details
                        ? JSON.stringify(event.details)
                        : Object.entries(event)
                            .filter(([k]) => !['type', 'timestamp', 'feed_id', 'details'].includes(k))
                            .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
                            .join(' ')
                      }
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="px-4 py-2 border-t border-[var(--border-dim)] bg-[var(--bg-secondary)]/80">
        <div className="flex items-center justify-between text-[10px] font-mono text-[var(--text-muted)]">
          <span>RUNTIME CONFIG — CHANGES DO NOT PERSIST TO DISK</span>
          <span>
            {hasChanges
              ? 'UNSAVED CHANGES'
              : 'UP TO DATE'
            }
          </span>
        </div>
      </footer>
    </div>
  );
}
