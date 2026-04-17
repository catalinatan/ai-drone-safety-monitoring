import { Monitor, Joystick } from 'lucide-react';

export type AppView = 'command-center' | 'drone-control';

interface AppNavProps {
  currentView: AppView;
  onViewChange: (view: AppView) => void;
}

export function AppNav({ currentView, onViewChange }: AppNavProps) {
  return (
    <nav className="flex items-center gap-1 p-1 rounded-lg border border-[var(--border-dim)] bg-[var(--bg-secondary)]/90 backdrop-blur-sm">
      <NavButton
        icon={<Monitor size={16} />}
        label="Command Center"
        isActive={currentView === 'command-center'}
        onClick={() => onViewChange('command-center')}
      />
      <NavButton
        icon={<Joystick size={16} />}
        label="Drone Control"
        isActive={currentView === 'drone-control'}
        onClick={() => onViewChange('drone-control')}
      />
    </nav>
  );
}

interface NavButtonProps {
  icon: React.ReactNode;
  label: string;
  isActive: boolean;
  onClick: () => void;
}

function NavButton({ icon, label, isActive, onClick }: NavButtonProps) {
  return (
    <button
      onClick={onClick}
      className={`
        flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-all duration-200
        ${
          isActive
            ? 'bg-[var(--accent-cyan)] text-[var(--bg-primary)]'
            : 'text-[var(--text-secondary)] hover:text-[var(--accent-cyan)] hover:bg-[var(--accent-cyan-glow)]'
        }
      `}
    >
      {icon}
      <span className="uppercase tracking-wider text-xs">{label}</span>
    </button>
  );
}
