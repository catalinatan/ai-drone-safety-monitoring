import { useState } from 'react';
import { X, Save } from 'lucide-react';

interface SettingsPanelProps {
  sceneType: string;
  autoRefresh: boolean;
  onSave: (sceneType: string, autoRefresh: boolean) => Promise<void>;
  onClose: () => void;
}

export function SettingsPanel({ sceneType, autoRefresh, onSave, onClose }: SettingsPanelProps) {
  const [localSceneType, setLocalSceneType] = useState(sceneType);
  const [localAutoRefresh, setLocalAutoRefresh] = useState(autoRefresh);
  const [saving, setSaving] = useState(false);

  const handleSave = async () => {
    setSaving(true);
    try {
      await onSave(localSceneType, localAutoRefresh);
    } finally {
      setSaving(false);
    }
  };

  const hasChanges = localSceneType !== sceneType || localAutoRefresh !== autoRefresh;

  return (
    <div className="absolute inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="w-80 rounded-xl border border-[var(--accent-cyan)]/30 bg-[var(--bg-secondary)] shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-[var(--border-dim)]">
          <h2 className="text-sm font-bold tracking-wider uppercase text-[var(--accent-cyan)]">
            Settings
          </h2>
          <button
            onClick={onClose}
            className="p-1 rounded hover:bg-[var(--bg-tertiary)] text-[var(--text-muted)] hover:text-[var(--text-secondary)] transition-colors"
          >
            <X size={14} />
          </button>
        </div>

        {/* Content */}
        <div className="px-4 py-4 space-y-4">
          {/* Scene Type */}
          <div>
            <label className="block text-[10px] font-bold font-mono uppercase tracking-wider text-[var(--text-muted)] mb-1.5">
              Scene Type
            </label>
            <select
              value={localSceneType}
              onChange={(e) => setLocalSceneType(e.target.value)}
              className="w-full px-3 py-2 rounded-lg border border-[var(--border-dim)] bg-[var(--bg-tertiary)] text-sm font-mono text-[var(--text-primary)] focus:outline-none focus:border-[var(--accent-cyan)] transition-colors cursor-pointer"
            >
              <option value="bridge">Bridge</option>
              <option value="railway">Railway</option>
              <option value="ship">Ship</option>
            </select>
            <p className="mt-1 text-[9px] font-mono text-[var(--text-muted)]">
              Applies to all 4 CCTV feeds
            </p>
          </div>

          {/* Auto Refresh */}
          <div>
            <label className="block text-[10px] font-bold font-mono uppercase tracking-wider text-[var(--text-muted)] mb-1.5">
              Auto-Refresh Zones
            </label>
            <button
              onClick={() => setLocalAutoRefresh(!localAutoRefresh)}
              className={`
                flex items-center gap-2 w-full px-3 py-2 rounded-lg border transition-all duration-200
                ${localAutoRefresh
                  ? 'border-[var(--zone-green)] bg-[var(--zone-green)]/10 text-[var(--zone-green)]'
                  : 'border-[var(--border-dim)] bg-[var(--bg-tertiary)] text-[var(--text-muted)]'
                }
              `}
            >
              <div className={`
                w-8 h-4 rounded-full transition-colors duration-200 relative
                ${localAutoRefresh ? 'bg-[var(--zone-green)]' : 'bg-[var(--bg-primary)]'}
              `}>
                <div className={`
                  absolute top-0.5 w-3 h-3 rounded-full bg-white transition-transform duration-200
                  ${localAutoRefresh ? 'translate-x-4' : 'translate-x-0.5'}
                `} />
              </div>
              <span className="text-xs font-mono">
                {localAutoRefresh ? 'Enabled' : 'Disabled'}
              </span>
            </button>
            <p className="mt-1 text-[9px] font-mono text-[var(--text-muted)]">
              Re-run zone segmentation every 60s
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-2 px-4 py-3 border-t border-[var(--border-dim)]">
          <button
            onClick={onClose}
            className="px-3 py-1.5 rounded-lg border border-[var(--border-dim)] bg-[var(--bg-tertiary)] text-xs font-mono text-[var(--text-muted)] hover:text-[var(--text-secondary)] transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            disabled={!hasChanges || saving}
            className={`
              flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-xs font-bold font-mono uppercase tracking-wider transition-all duration-200
              ${hasChanges && !saving
                ? 'border-[var(--accent-cyan)] bg-[var(--accent-cyan)]/10 text-[var(--accent-cyan)] hover:bg-[var(--accent-cyan)]/20'
                : 'border-[var(--border-dim)] bg-[var(--bg-tertiary)] text-[var(--text-muted)] cursor-not-allowed opacity-50'
              }
            `}
          >
            <Save size={12} />
            {saving ? 'Saving...' : 'Save'}
          </button>
        </div>
      </div>
    </div>
  );
}
