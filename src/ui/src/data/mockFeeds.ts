import type { Feed } from '../types';

export const mockFeeds: Feed[] = [
  {
    id: 'observer-cam-0',
    name: 'OBSERVER CAM',
    location: 'Aerial Overview',
    imageSrc: 'http://localhost:8001/video_feed/0',
    zones: [],
    isLive: true,
  },
  {
    id: 'cam-2',
    name: 'CAM 2',
    location: 'East Perimeter',
    imageSrc: 'https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=800&h=600&fit=crop',
    zones: [],
  },
  {
    id: 'cam-3',
    name: 'CAM 3',
    location: 'Parking Zone A',
    imageSrc: 'https://images.unsplash.com/photo-1573348722427-f1d6819fdf98?w=800&h=600&fit=crop',
    zones: [],
  },
  {
    id: 'cam-4',
    name: 'CAM 4',
    location: 'Loading Bay',
    imageSrc: 'https://images.unsplash.com/photo-1586528116311-ad8dd3c8310d?w=800&h=600&fit=crop',
    zones: [],
  },
];
