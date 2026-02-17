export type SeverityLevel = 'normal' | 'low' | 'medium' | 'high';

export interface Camera {
  id: string;
  name: string;
  severity: SeverityLevel;
  streamUrl: string;
  annotations?: Annotation[];
}

export interface Annotation {
  id: string;
  type: 'red' | 'yellow' | 'green';
  coordinates: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
}

export type PanelMode = 'none' | 'zoom' | 'edit';

export type DroneMode = 'automatic' | 'manual';
