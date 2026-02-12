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

export interface NEDCoordinate {
  x: number; // North (meters)
  y: number; // East (meters)
  z: number; // Down (meters, negative = above ground)
}

export interface DetectionStatus {
  feed_id: string;
  alarm_active: boolean;      // RED zone intrusion - drone deployment
  caution_active: boolean;    // YELLOW zone intrusion - highlight only
  people_count: number;
  danger_count: number;       // People in RED zones
  caution_count: number;      // People in YELLOW zones
  target_coordinates: NEDCoordinate | null;
  last_detection_time: string | null;
}

export interface Feed {
  id: string;
  name: string;
  location: string;
  imageSrc: string;
  zones: Zone[];
  isLive?: boolean;
  status?: DetectionStatus;
  sceneType?: 'ship' | 'railway' | 'bridge' | null;
  autoSegActive?: boolean;
}

export type ViewState =
  | { type: 'command' }
  | { type: 'edit'; feedId: string }
  | { type: 'expanded'; feedId: string };
