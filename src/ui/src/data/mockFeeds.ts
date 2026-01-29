import type { Feed } from '../types';

export const mockFeeds: Feed[] = [
  {
    id: 'observer-cam-0',
    name: 'OBSERVER CAM SW',
    location: 'Southwest View',
    imageSrc: 'http://localhost:8001/video_feed/0',
    zones: [],
    isLive: true,
  },
  {
    id: 'observer-cam-1',
    name: 'OBSERVER CAM S',
    location: 'South View',
    imageSrc: 'http://localhost:8001/video_feed/1',
    zones: [],
    isLive: true,
  },
  {
    id: 'observer-cam-2',
    name: 'OBSERVER CAM SE',
    location: 'Southeast View',
    imageSrc: 'http://localhost:8001/video_feed/2',
    zones: [],
    isLive: true,
  },
  {
    id: 'observer-cam-3',
    name: 'OBSERVER CAM WIDE',
    location: 'Wide Angle View',
    imageSrc: 'http://localhost:8001/video_feed/3',
    zones: [],
    isLive: true,
  },
];
