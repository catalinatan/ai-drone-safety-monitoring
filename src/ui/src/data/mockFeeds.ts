import type { Feed } from '../types';

// Backend API URL
export const BACKEND_URL = 'http://localhost:8001';

// Drone API URL for the search drone's dual cameras
export const DRONE_API_URL = 'http://localhost:8000';

// Live CCTV feeds from the backend (static aerial cameras on Drone2–Drone5)
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
    id: 'cctv-2',
    name: 'CCTV CAM 2',
    location: 'Aerial Overview',
    imageSrc: `${BACKEND_URL}/video_feed/cctv-2`,
    zones: [],
    isLive: true,
  },
  {
    id: 'cctv-3',
    name: 'CCTV CAM 3',
    location: 'Aerial Overview',
    imageSrc: `${BACKEND_URL}/video_feed/cctv-3`,
    zones: [],
    isLive: true,
  },
  {
    id: 'cctv-4',
    name: 'CCTV CAM 4',
    location: 'Aerial Overview',
    imageSrc: `${BACKEND_URL}/video_feed/cctv-4`,
    zones: [],
    isLive: true,
  },
];

// Combined feeds for the command panel (all 4 live feeds)
export const mockFeeds: Feed[] = [...liveFeeds];
