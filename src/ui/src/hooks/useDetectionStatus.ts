import { useState, useEffect, useRef, useCallback } from 'react';
import type { DetectionStatus } from '../types';
import { BACKEND_URL } from '../data/mockFeeds';

// Derive WebSocket URL from the HTTP backend URL
const WS_URL = BACKEND_URL.replace(/^http/, 'ws') + '/ws/status';

const RECONNECT_DELAY = 2000;

// Shared state across all hook consumers
let sharedStatuses: Record<string, DetectionStatus> = {};
let listeners: Set<() => void> = new Set();
let ws: WebSocket | null = null;
let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

function notify() {
  listeners.forEach((fn) => fn());
}

function connect() {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
    return;
  }

  ws = new WebSocket(WS_URL);

  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);

      // Backend sends { feeds: [...], timestamp } — unwrap the array
      if (msg.feeds && Array.isArray(msg.feeds)) {
        let changed = false;
        for (const status of msg.feeds as DetectionStatus[]) {
          if (status.feed_id) {
            sharedStatuses = { ...sharedStatuses, [status.feed_id]: status };
            changed = true;
          }
        }
        if (changed) notify();
      } else if (msg.feed_id) {
        // Also handle single-status messages for forward compatibility
        sharedStatuses = { ...sharedStatuses, [msg.feed_id]: msg as DetectionStatus };
        notify();
      }
    } catch {
      // ignore malformed messages
    }
  };

  ws.onclose = () => {
    ws = null;
    if (listeners.size > 0) {
      reconnectTimer = setTimeout(connect, RECONNECT_DELAY);
    }
  };

  ws.onerror = () => {
    ws?.close();
  };
}

function disconnect() {
  if (reconnectTimer) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }
  if (ws) {
    ws.close();
    ws = null;
  }
}

/**
 * Hook that subscribes to real-time detection status via WebSocket.
 * All components share a single WebSocket connection.
 * Falls back to an initial REST fetch for immediate state on mount.
 */
export function useDetectionStatus(feedId: string, isLive: boolean): DetectionStatus | null {
  const [, forceUpdate] = useState(0);
  const mounted = useRef(true);

  // Register as a listener to shared WS state
  useEffect(() => {
    mounted.current = true;
    const listener = () => {
      if (mounted.current) forceUpdate((n) => n + 1);
    };
    listeners.add(listener);

    // Start WS connection if this is the first listener
    if (listeners.size === 1) {
      connect();
    }

    return () => {
      mounted.current = false;
      listeners.delete(listener);
      // Disconnect if no more listeners
      if (listeners.size === 0) {
        disconnect();
      }
    };
  }, []);

  // Initial REST fetch so we don't wait for the first WS message
  useEffect(() => {
    if (!isLive) return;

    fetch(`${BACKEND_URL}/feeds/${feedId}/status`)
      .then((res) => (res.ok ? res.json() : null))
      .then((status) => {
        if (status && mounted.current) {
          sharedStatuses = { ...sharedStatuses, [feedId]: status };
          notify();
        }
      })
      .catch(() => {});
  }, [feedId, isLive]);

  return sharedStatuses[feedId] ?? null;
}
