import { useState, useEffect, useCallback, useRef } from 'react';
import { ArrowLeft, Save, Loader2, RefreshCw, AlertTriangle, Check } from 'lucide-react';
import { BACKEND_URL } from '../data/mockFeeds';

interface AdminPanelProps {
  onBack: () => void;
}

// Config section definitions — maps YAML structure to form fields
interface FieldDef {
  key: string;
  label: string;
  type: 'number' | 'string' | 'boolean' | 'select';
  step?: number;
  min?: number;
  max?: number;
  hint?: string;
  options?: { value: string; label: string }[];
  /** When set, this field is disabled unless the sibling field `enabledBy` is truthy */
  enabledBy?: string;
}

interface SectionDef {
  key: string;
  label: string;
  fields: FieldDef[];
}

const CONFIG_SECTIONS: SectionDef[] = [
  {
    key: 'detection',
    label: 'Human Detection',
    fields: [
      { key: 'confidence_threshold', label: 'Confidence', type: 'number', step: 0.05, min: 0.05, max: 1.0, hint: 'YOLO detection confidence (0-1)' },
      { key: 'inference_imgsz', label: 'Inference Size', type: 'number', step: 32, min: 320, max: 1920, hint: 'Input resolution (px)' },
      { key: 'fps', label: 'Detection FPS', type: 'number', step: 1, min: 1, max: 60, hint: 'Detection loop frequency' },
      { key: 'warmup_frames', label: 'Warmup Frames', type: 'number', step: 5, min: 0, max: 200, hint: 'Ignore alarms for N initial frames' },
    ],
  },
  {
    key: 'zones',
    label: 'Zone Detection',
    fields: [
      { key: 'overlap_threshold', label: 'Overlap Threshold', type: 'number', step: 0.05, min: 0.1, max: 1.0, hint: 'Person-zone overlap ratio to trigger' },
      { key: 'alarm_cooldown_seconds', label: 'Alarm Cooldown (s)', type: 'number', step: 1, min: 0, max: 60, hint: 'Minimum seconds between alarms' },
    ],
  },
  {
    key: 'auto_segmentation',
    label: 'Auto-Segmentation',
    fields: [
      { key: 'scene_type', label: 'Scene Type', type: 'select', options: [
        { value: 'bridge', label: 'Bridge' },
        { value: 'railway', label: 'Railway' },
        { value: 'ship', label: 'Ship' },
      ], hint: 'Applies to all CCTV feeds' },
      { key: 'enabled', label: 'Auto-Refresh Zones', type: 'boolean', hint: 'Periodically re-run zone segmentation' },
      { key: 'interval_seconds', label: 'Refresh Interval (s)', type: 'number', step: 10, min: 10, max: 600, hint: 'How often to re-segment zones', enabledBy: 'enabled' },
      { key: 'confidence', label: 'Confidence', type: 'number', step: 0.05, min: 0.1, max: 1.0, hint: 'Segmentation model confidence' },
      { key: 'simplify_epsilon', label: 'Polygon Simplify', type: 'number', step: 0.5, min: 0.5, max: 10, hint: 'Polygon approximation tolerance (px)' },
      { key: 'min_contour_area', label: 'Min Contour Area', type: 'number', step: 10, min: 10, max: 500, hint: 'Ignore contours smaller than this (px²)' },
    ],
  },
  {
    key: 'streaming',
    label: 'Streaming',
    fields: [
      { key: 'capture_fps', label: 'Capture FPS', type: 'number', step: 5, min: 1, max: 60, hint: 'Frame grab rate from cameras' },
      { key: 'stream_fps', label: 'Stream FPS', type: 'number', step: 5, min: 1, max: 60, hint: 'MJPEG output stream rate' },
    ],
  },
  {
    key: 'drone',
    label: 'Drone',
    fields: [
      { key: 'api_url', label: 'API URL', type: 'string', hint: 'Drone control server URL' },
      { key: 'api_timeout', label: 'API Timeout (s)', type: 'number', step: 1, min: 1, max: 30 },
      { key: 'safe_altitude', label: 'Safe Altitude (NED)', type: 'number', step: 1, min: -50, max: 0, hint: 'Negative = above ground in NED' },
    ],
  },
  {
    key: 'equipment',
    label: 'Deploy Equipment',
    fields: [
      { key: 'enabled', label: 'Enable Button', type: 'boolean', hint: 'Show Deploy Equipment button on drone panel' },
      { key: 'label', label: 'Equipment Name', type: 'string', hint: 'e.g. Lifevest, AED, First Aid Kit' },
    ],
  },
];

function getNestedValue(obj: Record<string, unknown>, path: string[]): unknown {
  let current: unknown = obj;
  for (const key of path) {
    if (current == null || typeof current !== 'object') return undefined;
    current = (current as Record<string, unknown>)[key];
  }
  return current;
}

function setNestedValue(obj: Record<string, unknown>, path: string[], value: unknown): Record<string, unknown> {
  const result = { ...obj };
  if (path.length === 1) {
    result[path[0]] = value;
    return result;
  }
  const [head, ...tail] = path;
  result[head] = setNestedValue(
    (result[head] as Record<string, unknown>) ?? {},
    tail,
    value,
  );
  return result;
}

// ---------------------------------------------------------------------------
// Per-feed camera pose section + PnP calibration tool
// ---------------------------------------------------------------------------

interface FeedPoseConfig {
  feed_id: string;
  name: string;
  gps: { latitude: number; longitude: number; altitude: number };
  orientation: { pitch: number; yaw: number; roll: number };
  fov: number;
}

function CalibrationTool({ feedId, onDone }: { feedId: string; onDone: () => void }) {
  const [points, setPoints] = useState<Array<{
    pixel: [number, number];
    world: [number, number, number];
  }>>([]);
  const [status, setStatus] = useState<'idle' | 'solving' | 'done' | 'error'>('idle');
  const [result, setResult] = useState<{ pitch: number; yaw: number; roll: number } | null>(null);
  const imgRef = useRef<HTMLImageElement>(null);

  const handleImageClick = (e: React.MouseEvent<HTMLImageElement>) => {
    const img = imgRef.current;
    if (!img) return;
    const rect = img.getBoundingClientRect();
    const scaleX = img.naturalWidth / rect.width;
    const scaleY = img.naturalHeight / rect.height;
    const px = Math.round((e.clientX - rect.left) * scaleX);
    const py = Math.round((e.clientY - rect.top) * scaleY);

    setPoints(prev => [...prev, {
      pixel: [px, py],
      world: [0, 0, 0],
    }]);
  };

  const updateWorldCoord = (index: number, axis: 0 | 1 | 2, value: number) => {
    setPoints(prev => prev.map((p, i) => {
      if (i !== index) return p;
      const world = [...p.world] as [number, number, number];
      world[axis] = value;
      return { ...p, world };
    }));
  };

  const removePoint = (index: number) => {
    setPoints(prev => prev.filter((_, i) => i !== index));
  };

  const handleCalibrate = async () => {
    if (points.length < 4) return;
    setStatus('solving');
    try {
      const img = imgRef.current;
      const res = await fetch(`${BACKEND_URL}/feeds/${feedId}/calibrate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          pixel_points: points.map(p => p.pixel),
          world_points: points.map(p => p.world),
          frame_w: img?.naturalWidth || 640,
          frame_h: img?.naturalHeight || 480,
        }),
      });
      if (res.ok) {
        const data = await res.json();
        setResult(data.orientation);
        setStatus('done');
      } else {
        setStatus('error');
      }
    } catch {
      setStatus('error');
    }
  };

  return (
    <div className="space-y-3">
      <p className="text-[10px] font-mono text-[var(--text-muted)]">
        Click 4+ points in the camera view, then enter their GPS coordinates (lat, lon, altitude).
      </p>

      {/* Live camera snapshot for clicking */}
      <div className="relative border border-[var(--border-dim)] rounded overflow-hidden">
        <img
          ref={imgRef}
          src={`${BACKEND_URL}/feeds/${feedId}/snapshot`}
          alt="Camera view"
          className="w-full cursor-crosshair"
          onClick={handleImageClick}
          onError={(e) => {
            (e.target as HTMLImageElement).src = `${BACKEND_URL}/video_feed/${feedId}`;
          }}
        />
        {/* Show clicked points as markers */}
        {points.map((p, i) => {
          const img = imgRef.current;
          if (!img) return null;
          const rect = img.getBoundingClientRect();
          const sx = rect.width / (img.naturalWidth || 1);
          const sy = rect.height / (img.naturalHeight || 1);
          return (
            <div key={i} className="absolute w-3 h-3 bg-[var(--accent-cyan)] rounded-full border border-white -translate-x-1/2 -translate-y-1/2 pointer-events-none text-[8px] text-center leading-3 font-bold"
              style={{ left: p.pixel[0] * sx, top: p.pixel[1] * sy }}>
              {i + 1}
            </div>
          );
        })}
      </div>

      {/* Point list with world coordinate inputs */}
      {points.length > 0 && (
        <div className="space-y-1">
          {points.map((p, i) => (
            <div key={i} className="flex items-center gap-2 text-[10px] font-mono">
              <span className="text-[var(--accent-cyan)] w-4">#{i + 1}</span>
              <span className="text-[var(--text-muted)] w-24">px({p.pixel[0]}, {p.pixel[1]})</span>
              <input type="number" step="0.0001" placeholder="Lat" value={p.world[0]}
                onChange={e => updateWorldCoord(i, 0, Number(e.target.value))}
                className="w-20 px-1 py-0.5 rounded border border-[var(--border-dim)] bg-[var(--bg-tertiary)] text-[var(--text-primary)] text-[10px]" />
              <input type="number" step="0.0001" placeholder="Lon" value={p.world[1]}
                onChange={e => updateWorldCoord(i, 1, Number(e.target.value))}
                className="w-20 px-1 py-0.5 rounded border border-[var(--border-dim)] bg-[var(--bg-tertiary)] text-[var(--text-primary)] text-[10px]" />
              <input type="number" step="0.1" placeholder="Alt(m)" value={p.world[2]}
                onChange={e => updateWorldCoord(i, 2, Number(e.target.value))}
                className="w-20 px-1 py-0.5 rounded border border-[var(--border-dim)] bg-[var(--bg-tertiary)] text-[var(--text-primary)] text-[10px]" />
              <button onClick={() => removePoint(i)}
                className="text-[var(--zone-red)] hover:text-red-400">x</button>
            </div>
          ))}
        </div>
      )}

      {/* Status + actions */}
      <div className="flex items-center gap-2">
        <button onClick={handleCalibrate}
          disabled={points.length < 4 || status === 'solving'}
          className={`px-3 py-1 rounded border text-[10px] font-mono font-bold uppercase tracking-wider ${
            points.length >= 4
              ? 'border-[var(--zone-green)] text-[var(--zone-green)] bg-[var(--zone-green)]/10 hover:bg-[var(--zone-green)]/20'
              : 'border-[var(--border-dim)] text-[var(--text-muted)] opacity-50 cursor-not-allowed'
          }`}>
          {status === 'solving' ? 'Solving...' : `Calibrate (${points.length}/4+ pts)`}
        </button>
        <button onClick={onDone}
          className="px-3 py-1 rounded border border-[var(--border-dim)] text-[var(--text-muted)] text-[10px] font-mono font-bold uppercase tracking-wider hover:text-[var(--text-secondary)]">
          Cancel
        </button>
      </div>

      {status === 'done' && result && (
        <div className="text-[10px] font-mono text-[var(--zone-green)] bg-[var(--zone-green)]/10 border border-[var(--zone-green)]/30 rounded p-2">
          Calibration successful — Pitch: {result.pitch.toFixed(1)}° | Yaw: {result.yaw.toFixed(1)}° | Roll: {result.roll.toFixed(1)}°
        </div>
      )}
      {status === 'error' && (
        <div className="text-[10px] font-mono text-[var(--zone-red)] bg-[var(--zone-red)]/10 border border-[var(--zone-red)]/30 rounded p-2">
          Calibration failed — try different point positions
        </div>
      )}
    </div>
  );
}

function HeightCalibrationTool({ feedId, onDone }: { feedId: string; onDone: (height?: number) => void }) {
  const imgRef = useRef<HTMLImageElement>(null);
  const [clickedPixel, setClickedPixel] = useState<[number, number] | null>(null);
  const [gps, setGps] = useState({ latitude: 0, longitude: 0 });
  const [status, setStatus] = useState<'idle' | 'calibrating' | 'success' | 'error'>('idle');
  const [result, setResult] = useState<number | null>(null);
  const [imgDims, setImgDims] = useState<{ w: number; h: number } | null>(null);

  const handleImgClick = (e: React.MouseEvent<HTMLImageElement>) => {
    const img = imgRef.current;
    if (!img) return;
    const rect = img.getBoundingClientRect();
    const scaleX = img.naturalWidth / rect.width;
    const scaleY = img.naturalHeight / rect.height;
    const px = (e.clientX - rect.left) * scaleX;
    const py = (e.clientY - rect.top) * scaleY;
    setClickedPixel([px, py]);
    setImgDims({ w: img.naturalWidth, h: img.naturalHeight });
  };

  const handleCalibrate = async () => {
    if (!clickedPixel || !imgDims) return;
    setStatus('calibrating');
    try {
      const res = await fetch(`${BACKEND_URL}/feeds/${feedId}/calibrate-height`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          pixel_x: clickedPixel[0],
          pixel_y: clickedPixel[1],
          latitude: gps.latitude,
          longitude: gps.longitude,
          frame_w: imgDims.w,
          frame_h: imgDims.h,
        }),
      });
      if (res.ok) {
        const data = await res.json();
        setResult(data.calibrated_height_m);
        setStatus('success');
      } else {
        setStatus('error');
      }
    } catch {
      setStatus('error');
    }
  };

  return (
    <div className="space-y-2 mt-2">
      <p className="text-[9px] font-mono text-[var(--text-muted)]">
        Click a point on the ground where someone could stand, then enter its GPS coordinates.
      </p>
      <div className="relative border border-[var(--border-dim)] rounded overflow-hidden">
        <img
          ref={imgRef}
          src={`${BACKEND_URL}/feeds/${feedId}/snapshot`}
          alt="snapshot"
          className="w-full cursor-crosshair"
          onClick={handleImgClick}
        />
        {clickedPixel && imgRef.current && (() => {
          const img = imgRef.current!;
          const rect = img.getBoundingClientRect();
          const sx = rect.width / (img.naturalWidth || 1);
          const sy = rect.height / (img.naturalHeight || 1);
          return (
            <div className="absolute w-3 h-3 bg-[var(--zone-green)] rounded-full border border-white -translate-x-1/2 -translate-y-1/2 pointer-events-none"
              style={{ left: clickedPixel[0] * sx, top: clickedPixel[1] * sy }} />
          );
        })()}
      </div>
      {clickedPixel && (
        <div className="text-[9px] font-mono text-[var(--text-secondary)]">
          Ground point: ({clickedPixel[0].toFixed(0)}, {clickedPixel[1].toFixed(0)})
        </div>
      )}
      <div className="grid grid-cols-2 gap-2">
        <div>
          <label className="block text-[9px] font-mono text-[var(--text-muted)] mb-0.5">Latitude</label>
          <input type="number" step="0.0001" value={gps.latitude}
            onChange={e => setGps({ ...gps, latitude: Number(e.target.value) })}
            className="w-full px-2 py-1 rounded border border-[var(--border-dim)] bg-[var(--bg-tertiary)] text-xs font-mono text-[var(--text-primary)] focus:outline-none focus:border-[var(--accent-cyan)]" />
        </div>
        <div>
          <label className="block text-[9px] font-mono text-[var(--text-muted)] mb-0.5">Longitude</label>
          <input type="number" step="0.0001" value={gps.longitude}
            onChange={e => setGps({ ...gps, longitude: Number(e.target.value) })}
            className="w-full px-2 py-1 rounded border border-[var(--border-dim)] bg-[var(--bg-tertiary)] text-xs font-mono text-[var(--text-primary)] focus:outline-none focus:border-[var(--accent-cyan)]" />
        </div>
      </div>
      <div className="flex gap-2">
        <button onClick={handleCalibrate}
          disabled={!clickedPixel || status === 'calibrating'}
          className="px-3 py-1 rounded border border-[var(--zone-green)] bg-[var(--zone-green)]/10 text-[var(--zone-green)] text-[10px] font-mono font-bold uppercase tracking-wider hover:bg-[var(--zone-green)]/20 disabled:opacity-40">
          {status === 'calibrating' ? 'Calibrating...' : 'Calibrate Height'}
        </button>
        <button onClick={() => onDone(result ?? undefined)}
          className="px-3 py-1 rounded border border-[var(--border-dim)] text-[var(--text-muted)] text-[10px] font-mono font-bold uppercase tracking-wider hover:text-[var(--text-secondary)]">
          {status === 'success' ? 'Done' : 'Cancel'}
        </button>
      </div>
      {status === 'success' && result !== null && (
        <div className="text-[10px] font-mono text-[var(--zone-green)] bg-[var(--zone-green)]/10 border border-[var(--zone-green)]/30 rounded p-2">
          Height calibrated: {result.toFixed(2)}m above ground
        </div>
      )}
      {status === 'error' && (
        <div className="text-[10px] font-mono text-[var(--zone-red)] bg-[var(--zone-red)]/10 border border-[var(--zone-red)]/30 rounded p-2">
          Calibration failed — ensure the clicked point is visible ground
        </div>
      )}
    </div>
  );
}

function CameraPoseSection() {
  const [feeds, setFeeds] = useState<Record<string, any> | null>(null);
  const [editing, setEditing] = useState<string | null>(null);
  const [calibrating, setCalibrating] = useState<string | null>(null);
  const [calibratingHeight, setCalibratingHeight] = useState<string | null>(null);
  const [poseValues, setPoseValues] = useState<FeedPoseConfig | null>(null);
  const [saveState, setSaveState] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle');

  useEffect(() => {
    fetch(`${BACKEND_URL}/config/feeds`)
      .then(r => r.json())
      .then(data => setFeeds(data))
      .catch(() => {});
  }, []);

  const startEdit = (feedId: string, feedDef: any) => {
    setEditing(feedId);
    setCalibrating(null);
    setCalibratingHeight(null);
    setPoseValues({
      feed_id: feedId,
      name: feedDef.name || feedId,
      gps: {
        latitude: feedDef.position?.latitude ?? 0,
        longitude: feedDef.position?.longitude ?? 0,
        altitude: feedDef.position?.altitude ?? 0,
      },
      orientation: {
        pitch: feedDef.orientation?.pitch ?? 0,
        yaw: feedDef.orientation?.yaw ?? 0,
        roll: feedDef.orientation?.roll ?? 0,
      },
      fov: feedDef.fov ?? 90,
    });
  };

  const handleSavePose = async () => {
    if (!editing || !poseValues || !feeds) return;
    setSaveState('saving');
    try {
      const updatedFeeds = { ...feeds };
      updatedFeeds[editing] = {
        ...updatedFeeds[editing],
        position: poseValues.gps,
        orientation: poseValues.orientation,
        fov: poseValues.fov,
      };

      const res = await fetch(`${BACKEND_URL}/config/feeds`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ feeds: updatedFeeds }),
      });

      if (res.ok) {
        setFeeds(updatedFeeds);

        // Also push GPS position update to runtime
        await fetch(`${BACKEND_URL}/feeds/${editing}/position`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(poseValues.gps),
        });

        setSaveState('saved');
        setEditing(null);
      } else {
        setSaveState('error');
      }
    } catch {
      setSaveState('error');
    }
    setTimeout(() => setSaveState('idle'), 2500);
  };

  if (!feeds) return null;

  return (
    <div className="rounded-lg border border-[var(--border-dim)] bg-[var(--bg-secondary)]/60 overflow-hidden">
      <div className="px-4 py-2.5 border-b border-[var(--border-dim)] bg-[var(--bg-tertiary)]/50">
        <h2 className="text-[10px] font-bold font-mono uppercase tracking-widest text-[var(--accent-cyan)]">
          Camera Pose (Per Feed)
        </h2>
      </div>
      <div className="px-4 py-2 space-y-3">
        {Object.entries(feeds).map(([feedId, feedDef]: [string, any]) => (
          <div key={feedId} className="border border-[var(--border-dim)] rounded p-3">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-mono font-semibold text-[var(--text-primary)]">
                {feedDef.name || feedId}
              </span>
              <div className="flex items-center">
                <button
                  onClick={() => startEdit(feedId, feedDef)}
                  className="text-[10px] font-mono text-[var(--accent-cyan)] hover:underline"
                >
                  {editing === feedId ? 'Editing...' : 'Edit Pose'}
                </button>
                <span className="text-[var(--border-dim)] mx-1">|</span>
                <button
                  onClick={() => { setCalibrating(feedId); setEditing(null); setCalibratingHeight(null); }}
                  className="text-[10px] font-mono text-[var(--zone-green)] hover:underline"
                >
                  Calibrate Orientation
                </button>
                <span className="text-[var(--border-dim)] mx-1">|</span>
                <button
                  onClick={() => { setCalibratingHeight(feedId); setEditing(null); setCalibrating(null); }}
                  className="text-[10px] font-mono text-[var(--zone-yellow,orange)] hover:underline"
                >
                  Calibrate Height
                </button>
              </div>
            </div>

            {calibratingHeight === feedId ? (
              <HeightCalibrationTool feedId={feedId} onDone={() => setCalibratingHeight(null)} />
            ) : calibrating === feedId ? (
              <CalibrationTool feedId={feedId} onDone={() => setCalibrating(null)} />
            ) : editing === feedId && poseValues ? (
              <div className="space-y-2">
                <div className="grid grid-cols-3 gap-2">
                  <div>
                    <label className="block text-[9px] font-mono text-[var(--text-muted)] mb-0.5">Latitude</label>
                    <input type="number" step="0.0001" value={poseValues.gps.latitude}
                      onChange={e => setPoseValues({...poseValues, gps: {...poseValues.gps, latitude: Number(e.target.value)}})}
                      className="w-full px-2 py-1 rounded border border-[var(--border-dim)] bg-[var(--bg-tertiary)] text-xs font-mono text-[var(--text-primary)] focus:outline-none focus:border-[var(--accent-cyan)]" />
                  </div>
                  <div>
                    <label className="block text-[9px] font-mono text-[var(--text-muted)] mb-0.5">Longitude</label>
                    <input type="number" step="0.0001" value={poseValues.gps.longitude}
                      onChange={e => setPoseValues({...poseValues, gps: {...poseValues.gps, longitude: Number(e.target.value)}})}
                      className="w-full px-2 py-1 rounded border border-[var(--border-dim)] bg-[var(--bg-tertiary)] text-xs font-mono text-[var(--text-primary)] focus:outline-none focus:border-[var(--accent-cyan)]" />
                  </div>
                  <div>
                    <label className="block text-[9px] font-mono text-[var(--text-muted)] mb-0.5">Altitude (m)</label>
                    <input type="number" step="0.1" value={poseValues.gps.altitude}
                      onChange={e => setPoseValues({...poseValues, gps: {...poseValues.gps, altitude: Number(e.target.value)}})}
                      className="w-full px-2 py-1 rounded border border-[var(--border-dim)] bg-[var(--bg-tertiary)] text-xs font-mono text-[var(--text-primary)] focus:outline-none focus:border-[var(--accent-cyan)]" />
                  </div>
                </div>
                <div className="grid grid-cols-3 gap-2">
                  <div>
                    <label className="block text-[9px] font-mono text-[var(--text-muted)] mb-0.5">Pitch (deg)</label>
                    <input type="number" step="1" value={poseValues.orientation.pitch}
                      onChange={e => setPoseValues({...poseValues, orientation: {...poseValues.orientation, pitch: Number(e.target.value)}})}
                      className="w-full px-2 py-1 rounded border border-[var(--border-dim)] bg-[var(--bg-tertiary)] text-xs font-mono text-[var(--text-primary)] focus:outline-none focus:border-[var(--accent-cyan)]" />
                  </div>
                  <div>
                    <label className="block text-[9px] font-mono text-[var(--text-muted)] mb-0.5">Yaw (heading)</label>
                    <input type="number" step="1" value={poseValues.orientation.yaw}
                      onChange={e => setPoseValues({...poseValues, orientation: {...poseValues.orientation, yaw: Number(e.target.value)}})}
                      className="w-full px-2 py-1 rounded border border-[var(--border-dim)] bg-[var(--bg-tertiary)] text-xs font-mono text-[var(--text-primary)] focus:outline-none focus:border-[var(--accent-cyan)]" />
                  </div>
                  <div>
                    <label className="block text-[9px] font-mono text-[var(--text-muted)] mb-0.5">Roll (deg)</label>
                    <input type="number" step="1" value={poseValues.orientation.roll}
                      onChange={e => setPoseValues({...poseValues, orientation: {...poseValues.orientation, roll: Number(e.target.value)}})}
                      className="w-full px-2 py-1 rounded border border-[var(--border-dim)] bg-[var(--bg-tertiary)] text-xs font-mono text-[var(--text-primary)] focus:outline-none focus:border-[var(--accent-cyan)]" />
                  </div>
                </div>
                <div className="w-1/3">
                  <label className="block text-[9px] font-mono text-[var(--text-muted)] mb-0.5">FOV (deg)</label>
                  <input type="number" step="1" min="10" max="180" value={poseValues.fov}
                    onChange={e => setPoseValues({...poseValues, fov: Number(e.target.value)})}
                    className="w-full px-2 py-1 rounded border border-[var(--border-dim)] bg-[var(--bg-tertiary)] text-xs font-mono text-[var(--text-primary)] focus:outline-none focus:border-[var(--accent-cyan)]" />
                </div>
                <div className="flex gap-2 pt-1">
                  <button onClick={handleSavePose}
                    disabled={saveState === 'saving'}
                    className="px-3 py-1 rounded border border-[var(--accent-cyan)] bg-[var(--accent-cyan)]/10 text-[var(--accent-cyan)] text-[10px] font-mono font-bold uppercase tracking-wider hover:bg-[var(--accent-cyan)]/20">
                    {saveState === 'saving' ? 'Saving...' : saveState === 'saved' ? 'Saved' : 'Save Pose'}
                  </button>
                  <button onClick={() => setEditing(null)}
                    className="px-3 py-1 rounded border border-[var(--border-dim)] text-[var(--text-muted)] text-[10px] font-mono font-bold uppercase tracking-wider hover:text-[var(--text-secondary)]">
                    Cancel
                  </button>
                </div>
              </div>
            ) : (
              <div className="text-[10px] font-mono text-[var(--text-muted)] space-y-0.5">
                <div>GPS: {feedDef.position?.latitude ?? '—'}, {feedDef.position?.longitude ?? '—'} | Alt: {feedDef.position?.altitude ?? '—'}m</div>
                <div>Orientation: pitch={feedDef.orientation?.pitch ?? '—'} yaw={feedDef.orientation?.yaw ?? '—'} roll={feedDef.orientation?.roll ?? '—'}</div>
                <div>FOV: {feedDef.fov ?? '—'}°</div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

export function AdminPanel({ onBack }: AdminPanelProps) {
  const [config, setConfig] = useState<Record<string, unknown> | null>(null);
  const [originalConfig, setOriginalConfig] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(true);
  const [saveState, setSaveState] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle');

  const fetchConfig = useCallback(async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/config`);
      if (res.ok) {
        const data = await res.json();
        setConfig(data);
        setOriginalConfig(JSON.parse(JSON.stringify(data)));
      }
    } catch {
      // Backend unavailable
    }
  }, []);

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      await fetchConfig();
      setLoading(false);
    };
    load();
  }, [fetchConfig]);

  const handleFieldChange = (sectionKey: string, fieldKey: string, value: unknown) => {
    if (!config) return;
    setConfig(setNestedValue(config, [sectionKey, fieldKey], value));
  };

  const hasChanges = config && originalConfig
    ? JSON.stringify(config) !== JSON.stringify(originalConfig)
    : false;

  const handleSave = async () => {
    if (!config) return;
    setSaveState('saving');
    try {
      const res = await fetch(`${BACKEND_URL}/config`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      });
      if (res.ok) {
        const data = await res.json();
        setConfig(data.config);
        setOriginalConfig(JSON.parse(JSON.stringify(data.config)));
        setSaveState('saved');
        // Notify sibling components (e.g. DroneControlPanel) of config change
        window.dispatchEvent(new CustomEvent('config-updated', { detail: data.config }));
      } else {
        setSaveState('error');
      }
    } catch {
      setSaveState('error');
    }
    setTimeout(() => setSaveState('idle'), 2500);
  };

  const handleReset = () => {
    if (originalConfig) {
      setConfig(JSON.parse(JSON.stringify(originalConfig)));
    }
  };

  const renderField = (section: SectionDef, field: FieldDef) => {
    if (!config) return null;
    const value = getNestedValue(config, [section.key, field.key]);

    // Check if this field is gated by another boolean field
    const isDisabledByGate = field.enabledBy
      ? !Boolean(getNestedValue(config, [section.key, field.enabledBy]))
      : false;

    if (field.type === 'select') {
      return (
        <div key={field.key} className="py-2">
          <label className="block text-xs font-mono text-[var(--text-primary)] mb-1">{field.label}</label>
          <select
            value={String(value ?? '')}
            onChange={(e) => handleFieldChange(section.key, field.key, e.target.value)}
            disabled={isDisabledByGate}
            className={`w-full px-2 py-1.5 rounded border border-[var(--border-dim)] bg-[var(--bg-tertiary)] text-xs font-mono text-[var(--text-primary)] focus:outline-none focus:border-[var(--accent-cyan)] transition-colors cursor-pointer ${isDisabledByGate ? 'opacity-40 cursor-not-allowed' : ''}`}
          >
            {field.options?.map((opt) => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>
          {field.hint && (
            <p className="text-[9px] font-mono text-[var(--text-muted)] mt-0.5">{field.hint}</p>
          )}
        </div>
      );
    }

    if (field.type === 'boolean') {
      const checked = Boolean(value);
      return (
        <div key={field.key} className="flex items-center justify-between py-2">
          <div className="flex-1 min-w-0">
            <span className="text-xs font-mono text-[var(--text-primary)]">{field.label}</span>
            {field.hint && (
              <p className="text-[9px] font-mono text-[var(--text-muted)] mt-0.5">{field.hint}</p>
            )}
          </div>
          <button
            onClick={() => handleFieldChange(section.key, field.key, !checked)}
            className={`
              flex items-center gap-2 px-2 py-1 rounded-lg border transition-all duration-200 ml-3
              ${checked
                ? 'border-[var(--zone-green)] bg-[var(--zone-green)]/10 text-[var(--zone-green)]'
                : 'border-[var(--border-dim)] bg-[var(--bg-tertiary)] text-[var(--text-muted)]'
              }
            `}
          >
            <div className={`
              w-7 h-3.5 rounded-full transition-colors duration-200 relative
              ${checked ? 'bg-[var(--zone-green)]' : 'bg-[var(--bg-primary)]'}
            `}>
              <div className={`
                absolute top-0.5 w-2.5 h-2.5 rounded-full bg-white transition-transform duration-200
                ${checked ? 'translate-x-3.5' : 'translate-x-0.5'}
              `} />
            </div>
            <span className="text-[10px] font-mono">{checked ? 'On' : 'Off'}</span>
          </button>
        </div>
      );
    }

    if (field.type === 'string') {
      return (
        <div key={field.key} className={`py-2 ${isDisabledByGate ? 'opacity-40' : ''}`}>
          <label className="block text-xs font-mono text-[var(--text-primary)] mb-1">{field.label}</label>
          <input
            type="text"
            value={String(value ?? '')}
            onChange={(e) => handleFieldChange(section.key, field.key, e.target.value)}
            disabled={isDisabledByGate}
            className={`w-full px-2 py-1.5 rounded border border-[var(--border-dim)] bg-[var(--bg-tertiary)] text-xs font-mono text-[var(--text-primary)] focus:outline-none focus:border-[var(--accent-cyan)] transition-colors ${isDisabledByGate ? 'cursor-not-allowed' : ''}`}
          />
          {field.hint && (
            <p className="text-[9px] font-mono text-[var(--text-muted)] mt-0.5">{field.hint}</p>
          )}
        </div>
      );
    }

    // number
    return (
      <div key={field.key} className={`py-2 ${isDisabledByGate ? 'opacity-40' : ''}`}>
        <div className="flex items-center justify-between mb-1">
          <label className="text-xs font-mono text-[var(--text-primary)]">{field.label}</label>
          <span className="text-[10px] font-mono text-[var(--accent-cyan)]">{String(value ?? '')}</span>
        </div>
        <input
          type="number"
          value={value != null ? Number(value) : ''}
          onChange={(e) => {
            const v = e.target.value === '' ? '' : Number(e.target.value);
            handleFieldChange(section.key, field.key, v);
          }}
          step={field.step}
          min={field.min}
          max={field.max}
          disabled={isDisabledByGate}
          className={`w-full px-2 py-1.5 rounded border border-[var(--border-dim)] bg-[var(--bg-tertiary)] text-xs font-mono text-[var(--text-primary)] focus:outline-none focus:border-[var(--accent-cyan)] transition-colors ${isDisabledByGate ? 'cursor-not-allowed' : ''}`}
        />
        {field.hint && (
          <p className="text-[9px] font-mono text-[var(--text-muted)] mt-0.5">{field.hint}</p>
        )}
      </div>
    );
  };

  if (loading) {
    return (
      <div className="h-full flex items-center justify-center bg-[var(--bg-primary)] tactical-grid">
        <div className="flex items-center gap-3 text-[var(--accent-cyan)]">
          <Loader2 className="animate-spin" size={20} />
          <span className="text-sm font-mono">Loading configuration...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col bg-[var(--bg-primary)] tactical-grid">
      {/* Header */}
      <header className="flex items-center justify-between px-4 py-3 border-b border-[var(--border-dim)] bg-[var(--bg-secondary)]/80 backdrop-blur-sm">
        <div className="flex items-center gap-3">
          <button
            onClick={onBack}
            className="flex items-center gap-1.5 px-2 py-1 rounded border border-[var(--border-dim)] bg-[var(--bg-tertiary)] text-[var(--text-muted)] hover:text-[var(--accent-cyan)] hover:border-[var(--accent-cyan)] transition-all duration-200"
          >
            <ArrowLeft size={12} />
            <span className="text-[10px] font-bold font-mono uppercase tracking-wider">Back</span>
          </button>
          <div className="h-4 w-px bg-[var(--border-dim)]" />
          <h1 className="text-sm font-bold tracking-wider uppercase text-glow-cyan">
            System Configuration
          </h1>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={handleReset}
            disabled={!hasChanges}
            className={`
              flex items-center gap-1.5 px-2 py-1 rounded border text-[10px] font-bold font-mono uppercase tracking-wider transition-all
              ${hasChanges
                ? 'border-[var(--zone-yellow)] bg-[var(--zone-yellow)]/10 text-[var(--zone-yellow)] hover:bg-[var(--zone-yellow)]/20'
                : 'border-[var(--border-dim)] bg-[var(--bg-tertiary)] text-[var(--text-muted)] cursor-not-allowed opacity-50'
              }
            `}
          >
            <RefreshCw size={10} />
            Reset
          </button>
          <button
            onClick={handleSave}
            disabled={!hasChanges || saveState === 'saving'}
            className={`
              flex items-center gap-1.5 px-2 py-1 rounded border text-[10px] font-bold font-mono uppercase tracking-wider transition-all
              ${saveState === 'saved'
                ? 'border-[var(--zone-green)] bg-[var(--zone-green)]/10 text-[var(--zone-green)]'
                : saveState === 'error'
                ? 'border-[var(--zone-red)] bg-[var(--zone-red)]/10 text-[var(--zone-red)]'
                : hasChanges && saveState !== 'saving'
                ? 'border-[var(--accent-cyan)] bg-[var(--accent-cyan)]/10 text-[var(--accent-cyan)] hover:bg-[var(--accent-cyan)]/20'
                : 'border-[var(--border-dim)] bg-[var(--bg-tertiary)] text-[var(--text-muted)] cursor-not-allowed opacity-50'
              }
            `}
          >
            {saveState === 'saving' ? <Loader2 size={10} className="animate-spin" /> :
             saveState === 'saved' ? <Check size={10} /> :
             saveState === 'error' ? <AlertTriangle size={10} /> :
             <Save size={10} />}
            {saveState === 'saving' ? 'Saving...' :
             saveState === 'saved' ? 'Saved' :
             saveState === 'error' ? 'Error' :
             'Save'}
          </button>
        </div>
      </header>

      {/* Content */}
      <main className="flex-1 min-h-0 overflow-y-auto p-4">
        {config && (
          <div className="max-w-3xl mx-auto space-y-4">
            {CONFIG_SECTIONS.map((section) => (
              <div
                key={section.key}
                className="rounded-lg border border-[var(--border-dim)] bg-[var(--bg-secondary)]/60 overflow-hidden"
              >
                <div className="px-4 py-2.5 border-b border-[var(--border-dim)] bg-[var(--bg-tertiary)]/50">
                  <h2 className="text-[10px] font-bold font-mono uppercase tracking-widest text-[var(--accent-cyan)]">
                    {section.label}
                  </h2>
                </div>
                <div className="px-4 py-1 divide-y divide-[var(--border-dim)]/50">
                  {section.fields.map((field) => renderField(section, field))}
                </div>
              </div>
            ))}
            <CameraPoseSection />
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="px-4 py-2 border-t border-[var(--border-dim)] bg-[var(--bg-secondary)]/80">
        <div className="flex items-center justify-between text-[10px] font-mono text-[var(--text-muted)]">
          <span>RUNTIME CONFIG — CHANGES DO NOT PERSIST TO DISK</span>
          <span>
            {hasChanges
              ? 'UNSAVED CHANGES'
              : 'UP TO DATE'
            }
          </span>
        </div>
      </footer>
    </div>
  );
}
