export type ZoneLevel = 'red' | 'yellow' | 'green';

export interface Point {
  x: number;
  y: number;
}

export interface Zone {
  id: string;
  level: ZoneLevel;
  points: Point[];
}

export interface Feed {
  id: string;
  name: string;
  location: string;
  imageSrc: string;
  zones: Zone[];
  isLive?: boolean;
}

export interface NEDCoordinate {
  x: number; // North (meters)
  y: number; // East (meters)
  z: number; // Down (meters, negative = above ground)
}

export type ViewState =
  | { type: 'command' }
  | { type: 'edit'; feedId: string }
  | { type: 'expanded'; feedId: string };
