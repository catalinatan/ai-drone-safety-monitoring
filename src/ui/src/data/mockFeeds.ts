import type { Feed } from '../types';

// Backend API URL
export const BACKEND_URL = 'http://localhost:8001';

// Live feeds from the backend (AirSim cameras)
export const liveFeeds: Feed[] = [
  {
    id: 'cctv-1',
    name: 'CCTV CAM 1',
    location: 'Aerial Overview',
    imageSrc: `${BACKEND_URL}/video_feed/cctv-1`,
    zones: [],
    isLive: true,
  },
  {
    id: 'drone-cam',
    name: 'DRONE CAM',
    location: 'Mobile Unit',
    imageSrc: `${BACKEND_URL}/video_feed/drone-cam`,
    zones: [],
    isLive: true,
  },
];

// Placeholder feeds (shown when there are fewer than 4 live feeds)
export const placeholderFeeds: Feed[] = [
  {
    id: 'placeholder-1',
    name: 'CAM 3',
    location: 'Not Connected',
    imageSrc: '',
    zones: [],
    isLive: false,
  },
  {
    id: 'placeholder-2',
    name: 'CAM 4',
    location: 'Not Connected',
    imageSrc: '',
    zones: [],
    isLive: false,
  },
];

// Combined feeds for the command panel (always 4 feeds)
export const mockFeeds: Feed[] = [
  ...liveFeeds,
  ...placeholderFeeds.slice(0, Math.max(0, 4 - liveFeeds.length)),
];
