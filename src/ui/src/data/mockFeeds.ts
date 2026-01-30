import type { Feed } from '../types';

// Backend API URL
export const BACKEND_URL = 'http://localhost:8001';

// Drone API URL for the search drone's dual cameras
export const DRONE_API_URL = 'http://localhost:8000';

// Live CCTV feed from the backend (static aerial camera on Drone2)
export const liveFeeds: Feed[] = [
  {
    id: 'cctv-1',
    name: 'CCTV CAM 1',
    location: 'Aerial Overview',
    imageSrc: `${BACKEND_URL}/video_feed/cctv-1`,
    zones: [],
    isLive: true,
  },
];

// Placeholder feeds (shown when there are fewer than 4 live feeds)
export const placeholderFeeds: Feed[] = [
  {
    id: 'placeholder-1',
    name: 'CAM 2',
    location: 'Not Connected',
    imageSrc: '',
    zones: [],
    isLive: false,
  },
  {
    id: 'placeholder-2',
    name: 'CAM 3',
    location: 'Not Connected',
    imageSrc: '',
    zones: [],
    isLive: false,
  },
  {
    id: 'placeholder-3',
    name: 'CAM 4',
    location: 'Not Connected',
    imageSrc: '',
    zones: [],
    isLive: false,
  },
];

// Combined feeds for the command panel (always 4 feeds: 1 live + 3 placeholders)
export const mockFeeds: Feed[] = [...liveFeeds, ...placeholderFeeds];
