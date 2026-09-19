import { useEffect, useMemo, useRef, useState } from "react";
import { useIsMobile } from "../../lib/useIsMobile";
import { geoMercator, geoPath, type GeoProjection } from "d3-geo";
import { feature, mesh } from "topojson-client";
import type { FeatureCollection, MultiLineString } from "geojson";
import type { Topology } from "topojson-specification";
import { RotateCcw } from "lucide-react";
import type { TrajectoirePoint } from "../../services/api";
import { severityHue } from "../../lib/severity";

const VIEW_W = 960;
const VIEW_H = 520;
const PAD = 36;
const MAX_ZOOM = 24;

// Altitude colour ramp: light green at ground level to a fairly dark blue at
// cruise. Two separate flights use two separate projections (one fit tight
// to each one's own bbox), so this is deliberately independent of anomaly
// severity — severity still shows up elsewhere (badge, ring gauge) — the
// path itself is read the same way FlightRadar-style maps read altitude.
const ALT_HUE_GROUND = 140;
const ALT_HUE_CRUISE = 213;
const ALT_LIGHT_GROUND = 72;
const ALT_LIGHT_CRUISE = 26;
const ALT_SAT = 62;
const MARKER_NEUTRAL_HUE = 195; // cyan — no anomaly status available yet (flight in progress)

interface WorldData {
  coastlines: MultiLineString;
  internalBorders: MultiLineString;
  land: FeatureCollection;
}

// 50m resolution (~750 KB) rather than the 110m default (~100 KB) — the
// coarser file showed visibly chunky, low-vertex coastlines once zoomed in
// tight on a short local flight, looking like a rendering glitch rather
// than an actual coastline. Loaded via a dynamic import (not a static one)
// specifically so Vite code-splits it into its own chunk instead of
// inflating the main bundle everyone downloads up front — it only loads
// once a map actually needs to be drawn, and only once (module-level
// cache, shared by every VueTrajectoire instance).
//
// Antarctica and Greenland both used to be excluded from the land fill
// entirely: Mercator sends the poles to infinity, and projecting them
// through a *flight's own tight-fit projection* (scale chosen to fill the
// view with a small bbox, sometimes a fraction of a degree wide) blows
// their polygons up to grotesque size, which d3-geo's clipExtent then
// closes with a straight edge tracing the clip rectangle — a visible
// spike or seam, confirmed directly with a standalone node script (the
// jump was ~7x the viewport width for a short local flight). That fix
// traded a real bug for a worse-looking one: Greenland/Antarctica just
// never rendered at all, at any zoom, for any flight.
// The actual fix is architectural, not a bigger exclusion list: land/
// coastlines/borders are no longer projected through the flight's own
// (arbitrarily extreme) projection at all. They're computed once through
// WORLD_FIT — a fixed, always-reasonable whole-globe Mercator fit that
// never blows up (verified: even Antarctica's own polygon jump is ~9px at
// this scale, against a 960px viewport) — and then carried into the
// flight's coordinate space with a single affine transform (computed from
// the two projections' scale/translate, see `ratio`/`alignDx`/`alignDy`
// below) instead of being re-projected per flight. No exclusion needed.
let worldDataPromise: Promise<WorldData> | null = null;
function loadWorldData(): Promise<WorldData> {
  if (!worldDataPromise) {
    worldDataPromise = import("world-atlas/countries-50m.json").then((module) => {
      const world = module.default as unknown as Topology;
      // Split into two meshes so coastlines can be drawn thicker than
      // internal (country-to-country) borders: topojson's mesh filter is
      // called with the two geometries sharing a given arc — for a true
      // coastline that's the *same* geometry on both sides (nothing
      // borders the ocean), for a shared land border it's two countries.
      return {
        coastlines: mesh(world, world.objects.countries as never, (a, b) => a === b) as MultiLineString,
        internalBorders: mesh(world, world.objects.countries as never, (a, b) => a !== b) as MultiLineString,
        // Land fill — rendered under the border strokes so the ocean tint
        // only shows through where there's no land, i.e. the sea, instead
        // of tinting uniformly.
        land: feature(world, world.objects.countries as never) as unknown as FeatureCollection,
      };
    });
  }
  return worldDataPromise;
}

// Reference projection fit to the entire globe, on the same extent as any
// per-flight projection below — gives the scale/translate a flight's own
// projection would need to reach "whole world visible" exactly. Computed
// once: it never depends on which flight is loaded or on the (lazily
// loaded) topology data, only on the fixed viewBox/padding.
const WORLD_FIT = geoMercator().fitExtent(
  [
    [PAD, PAD],
    [VIEW_W - PAD, VIEW_H - PAD],
  ],
  { type: "Sphere" },
);
const WORLD_SCALE = WORLD_FIT.scale();
const WORLD_TRANSLATE = WORLD_FIT.translate();
// One Mercator "wrap" in WORLD_FIT's own (fixed, never-distorted) space —
// tiling the world map at multiples of this, then carrying the whole tiled
// group into the flight's coordinate space with the ratio/align transform
// below, is what lets the map background use a sane, constant scale
// regardless of how tight any given flight's own projection is.
const WORLD_TILE_PERIOD = WORLD_SCALE * 2 * Math.PI;
// Fixed (WORLD_TILE_PERIOD never changes), so this is a plain constant
// rather than a useMemo — same 5x3 grid the flight-space tileOffsets used
// to cover, just expressed in WORLD_FIT's own units now.
const WORLD_TILE_OFFSETS: { dx: number; dy: number }[] = [];
for (let i = -2; i <= 2; i++) {
  for (let j = -1; j <= 1; j++) {
    WORLD_TILE_OFFSETS.push({ dx: i * WORLD_TILE_PERIOD, dy: j * WORLD_TILE_PERIOD });
  }
}

interface Projected {
  x: number;
  y: number;
  altitude: number | null;
  vitesse: number | null;
  timestamp: number;
}

/** Fills in missing altitudes for display (carries the nearest known value
 * forward/backward) — never fed back into a calculation. */
function fillAltitudes(points: TrajectoirePoint[]): (number | null)[] {
  const filled: (number | null)[] = points.map((p) => p.altitude);
  for (let i = 1; i < filled.length; i++) if (filled[i] === null) filled[i] = filled[i - 1];
  for (let i = filled.length - 2; i >= 0; i--) if (filled[i] === null) filled[i] = filled[i + 1];
  return filled;
}

/** Mercator projection tightly fit to the flight's own bounding box — the
 * point is to actually read the trajectory clearly, a little border context
 * around it, not a wide zoomed-out world view. Further zoom is up to the
 * viewer (wheel). */
function buildProjection(trajectoire: TrajectoirePoint[]): GeoProjection {
  const lats = trajectoire.map((p) => p.lat);
  const lons = trajectoire.map((p) => p.lon);
  const latMin = Math.min(...lats);
  const latMax = Math.max(...lats);
  const lonMin = Math.min(...lons);
  const lonMax = Math.max(...lons);
  const latSpan = Math.max(latMax - latMin, 0.08);
  const lonSpan = Math.max(lonMax - lonMin, 0.08);
  const marginLat = latSpan * 0.12;
  const marginLon = lonSpan * 0.12;
  // Winding order matters here: d3-geo treats a small "counterclockwise in
  // lon/lat" ring as the *complement* of the area it looks like it encloses
  // (its right-hand-rule convention for a small ring, seen from outside the
  // sphere, is clockwise in a standard lon=x/lat=y plot). Get this backwards
  // and fitExtent silently fits the *rest of the planet* instead of the
  // flight's own bbox — which is exactly what showed up as "the whole Earth"
  // instead of a tight crop.
  const bbox = {
    type: "Feature" as const,
    properties: {},
    geometry: {
      type: "Polygon" as const,
      coordinates: [
        [
          [lonMin - marginLon, latMin - marginLat],
          [lonMin - marginLon, latMax + marginLat],
          [lonMax + marginLon, latMax + marginLat],
          [lonMax + marginLon, latMin - marginLat],
          [lonMin - marginLon, latMin - marginLat],
        ],
      ],
    },
  };
  return geoMercator().fitExtent(
    [
      [PAD, PAD],
      [VIEW_W - PAD, VIEW_H - PAD],
    ],
    bbox,
  );
}

function project(trajectoire: TrajectoirePoint[], projection: GeoProjection): Projected[] {
  const altitudes = fillAltitudes(trajectoire);
  return trajectoire.map((p, i) => {
    const [x, y] = projection([p.lon, p.lat]) ?? [0, 0];
    return { x, y, altitude: altitudes[i], vitesse: p.vitesse, timestamp: p.timestamp };
  });
}

/** Heading (0 = north, clockwise) between two points, used to orient the
 * aircraft marker. */
function bearing(a: TrajectoirePoint, b: TrajectoirePoint): number {
  const toRad = Math.PI / 180;
  const dLon = (b.lon - a.lon) * toRad;
  const y = Math.sin(dLon) * Math.cos(b.lat * toRad);
  const x = Math.cos(a.lat * toRad) * Math.sin(b.lat * toRad) - Math.sin(a.lat * toRad) * Math.cos(b.lat * toRad) * Math.cos(dLon);
  return ((Math.atan2(y, x) * 180) / Math.PI + 360) % 360;
}

/** Heading using a point a few samples back rather than just the last
 * segment — a single ADS-B sample pair can be noisy enough (near-duplicate
 * positions while taxiing/slow) to point the marker the wrong way. */
function currentHeading(trajectoire: TrajectoirePoint[]): number {
  const n = trajectoire.length;
  const backIdx = Math.max(0, n - 1 - Math.min(5, n - 1));
  return bearing(trajectoire[backIdx], trajectoire[n - 1]);
}

function formatTime(timestamp: number): string {
  return new Date(timestamp * 1000).toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function nearestIndex(points: Projected[], x: number, y: number): number {
  let best = 0;
  let bestDist = Infinity;
  for (let i = 0; i < points.length; i++) {
    const d = (points[i].x - x) ** 2 + (points[i].y - y) ** 2;
    if (d < bestDist) {
      bestDist = d;
      best = i;
    }
  }
  return best;
}

interface Props {
  trajectoire: TrajectoirePoint[];
  /** anomaly score (high = normal) — colours the current-position marker; null = neutral colour */
  severity?: number | null;
  enVol?: boolean;
  origineVille?: string | null;
  destinationVille?: string | null;
}

export function VueTrajectoire({ trajectoire, severity = null, enVol = false, origineVille = null, destinationVille = null }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);
  // Border/land geometry loads asynchronously (see loadWorldData) — the
  // flight path itself never waits on it, this just fills in behind it
  // once the chunk arrives.
  const [worldData, setWorldData] = useState<WorldData | null>(null);
  useEffect(() => {
    let cancelled = false;
    loadWorldData().then((data) => {
      if (!cancelled) setWorldData(data);
    });
    return () => {
      cancelled = true;
    };
  }, []);
  // zoom and pan live in one state object updated by a single, pure setView
  // call. They used to be two separate useState calls with the wheel handler
  // nesting a setPan(...) inside the setZoom(...) updater — React StrictMode
  // deliberately invokes updater functions twice in development to catch
  // exactly that kind of impurity, so that nested call fired twice per wheel
  // tick and applied the pan shift twice, sending the map flying much further
  // than intended. A single pure updater has nothing to double-apply.
  const [view, setView] = useState({ zoom: 1, pan: { x: 0, y: 0 } });
  const { zoom, pan } = view;
  const [isDragging, setIsDragging] = useState(false);
  const draggingRef = useRef(false);
  // Every finger/pointer currently down, in client coordinates. A mouse
  // only ever has one entry here, so desktop behaviour (single-pointer drag
  // pan) is unchanged — a second entry only appears with a second finger.
  const pointersRef = useRef(new Map<number, { x: number; y: number }>());
  const pinchRef = useRef<{ dist: number; mid: { x: number; y: number } } | null>(null);
  const isMobile = useIsMobile();
  const lastPosRef = useRef({ x: 0, y: 0 });

  // Fresh view whenever a different flight is loaded.
  useEffect(() => {
    setView({ zoom: 1, pan: { x: 0, y: 0 } });
  }, [trajectoire]);

  // buildProjection fits tightly to the flight's own (small) bbox, so its
  // scale can be enormous — fine for projecting the flight's own points
  // directly (below), but no longer used to project the world map itself
  // (see loadWorldData's comment) — a clipExtent hack used to paper over
  // that mismatch here, no longer needed now that the map background is
  // projected separately, at a fixed sane scale, and carried in via an
  // affine transform instead.
  const projection = useMemo(() => buildProjection(trajectoire), [trajectoire]);
  const points = useMemo(() => project(trajectoire, projection), [trajectoire, projection]);

  // Cumulative on-screen (projected) distance up to each point — walking
  // the reveal forward by *distance* rather than by point count is what
  // makes the plane move at a constant visual speed. Real ADS-B samples are
  // unevenly spaced (dense while manoeuvring, sparse in steady cruise), so
  // an equal-points-per-frame ramp would speed up and slow down with the
  // sampling density itself — exactly the "à-coups" (jerkiness) reported.
  const cumDist = useMemo(() => {
    const d = [0];
    for (let i = 1; i < points.length; i++) {
      d.push(d[i - 1] + Math.hypot(points[i].x - points[i - 1].x, points[i].y - points[i - 1].y));
    }
    return d;
  }, [points]);
  const totalDist = cumDist[cumDist.length - 1] ?? 0;

  // Reveal the path progressively from takeoff to the current position on
  // load, instead of drawing the whole trajectory at once — a linear ramp
  // (no easing) of *distance travelled*, driven by requestAnimationFrame,
  // over a fixed 3s regardless of the flight's actual length or point count.
  const [revealDist, setRevealDist] = useState(0);
  useEffect(() => {
    let raf = 0;
    const start = performance.now();
    const durationMs = 3000;
    setRevealDist(0);
    function tick(now: number) {
      const t = Math.min(1, (now - start) / durationMs);
      setRevealDist(t * totalDist);
      if (t < 1) raf = requestAnimationFrame(tick);
    }
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [trajectoire, totalDist]);

  // Find which real segment the current reveal distance falls in, and how
  // far across it (0..1) — the plane and the path's leading edge sit at
  // this exact interpolated point, not snapped to the nearest sample, so
  // the motion is smooth between samples instead of hopping point to point.
  let segIdx = 0;
  while (segIdx < cumDist.length - 2 && cumDist[segIdx + 1] < revealDist) segIdx++;
  const segP0 = points[segIdx];
  const segP1 = points[Math.min(segIdx + 1, points.length - 1)];
  const segLen = cumDist[segIdx + 1] - cumDist[segIdx];
  const segFrac = segLen > 0 ? Math.min(1, Math.max(0, (revealDist - cumDist[segIdx]) / segLen)) : 1;
  const planeX = segP0.x + segFrac * (segP1.x - segP0.x);
  const planeY = segP0.y + segFrac * (segP1.y - segP0.y);
  const planeAltitude =
    segP0.altitude !== null && segP1.altitude !== null ? segP0.altitude + segFrac * (segP1.altitude - segP0.altitude) : segP0.altitude;
  const arrived = revealDist >= totalDist;
  const visiblePoints: Projected[] = [
    ...points.slice(0, segIdx + 1),
    { x: planeX, y: planeY, altitude: planeAltitude, vitesse: segP1.vitesse, timestamp: segP1.timestamp },
  ];
  // Projected once through the fixed WORLD_FIT projection — never through
  // the flight's own (arbitrarily extreme) one — so these never need any
  // per-feature exclusion or distortion workaround (see loadWorldData's
  // comment). Depends only on worldData, not on which flight is loaded, so
  // switching flights no longer recomputes any of this either.
  const worldCoastPath = useMemo(() => (worldData ? geoPath(WORLD_FIT)(worldData.coastlines) ?? "" : ""), [worldData]);
  const worldInternalPath = useMemo(() => (worldData ? geoPath(WORLD_FIT)(worldData.internalBorders) ?? "" : ""), [worldData]);
  const worldLandPath = useMemo(() => (worldData ? geoPath(WORLD_FIT)(worldData.land) ?? "" : ""), [worldData]);
  // Both `projection` (the flight's own tight fit) and WORLD_FIT are plain
  // Mercator projections with no rotation, so they're related by a single
  // affine transform: scale by the ratio of their scales, then translate by
  // the difference between the flight's own translate and the (ratio-
  // scaled) world one. Wrapping the WORLD_FIT-space map in
  // `translate(alignDx alignDy) scale(ratio)` carries it exactly into the
  // flight's coordinate space, where it lines up with the flight path
  // itself (projected directly through `projection`, unaffected by any of
  // this) — without ever re-projecting a single point of the map itself
  // through the flight's own, potentially extreme, scale.
  const ratio = projection.scale() / WORLD_SCALE;
  const [projTx, projTy] = projection.translate();
  const alignDx = projTx - ratio * WORLD_TRANSLATE[0];
  const alignDy = projTy - ratio * WORLD_TRANSLATE[1];
  // One Mercator "wrap" (360° of longitude) in this projection's own output
  // pixels — repeating the border paths at multiples of this offset tiles
  // the map seamlessly instead of leaving empty space once you pan or zoom
  // out past the rendered extent. Reused as the vertical period too: a
  // Mercator map doesn't actually repeat top-to-bottom, but a square tile
  // is the simplest way to give the same "never run out of map" effect in
  // both directions, which is what was asked for here.
  // Flight-space equivalent of WORLD_TILE_PERIOD (equal to it times `ratio`,
  // by construction — see worldLandPath's comment) — still needed here for
  // clampPan below, which operates in flight/pan space, not WORLD_FIT space.
  const tilePeriod = projection.scale() * 2 * Math.PI;
  // How far this view can zoom out: the point where the flight's own tight
  // projection, scaled down, matches what a whole-world fit would look like
  // — past that, you'd just see the globe shrink into empty padding.
  const minZoom = WORLD_SCALE / projection.scale();

  // Keeps the pan from ever dragging the visible viewport past the tiled
  // border/land grid (which only covers dx in roughly [-2.5, 2.5] tile
  // periods and dy in [-1.5, 1.5]) — past that edge there's nothing drawn
  // at all, which reads as the map having broken rather than "you've
  // panned off the (deliberately finite, repeating) tiled area." Dragging
  // now simply stops at that edge instead of revealing the empty void.
  function clampPan(candidate: { x: number; y: number }, z: number): { x: number; y: number } {
    const marginX = 2.5 * tilePeriod * z;
    const marginY = 1.5 * tilePeriod * z;
    const minX = Math.min(VIEW_W - marginX, marginX);
    const maxX = Math.max(VIEW_W - marginX, marginX);
    const minY = Math.min(VIEW_H - marginY, marginY);
    const maxY = Math.max(VIEW_H - marginY, marginY);
    return {
      x: Math.min(Math.max(candidate.x, minX), maxX),
      y: Math.min(Math.max(candidate.y, minY), maxY),
    };
  }
  const hue = severity !== null ? severityHue(severity, 1) : MARKER_NEUTRAL_HUE;
  const known = points.map((p) => p.altitude).filter((a): a is number => a !== null);
  const altMin = known.length ? Math.min(...known) : 0;
  const altMax = known.length ? Math.max(...known) : 0;

  // Converts a client (mouse) position to the SVG's own viewBox coordinate
  // space via the browser's own screen-to-user-space matrix, rather than a
  // hand-rolled ratio off getBoundingClientRect() — that manual version is
  // only correct if the rendered box's aspect ratio exactly matches the
  // viewBox's, and any mismatch (subpixel rounding, a constrained parent,
  // preserveAspectRatio letterboxing) throws the cursor anchor off exactly
  // the way that was reported. getScreenCTM() is authoritative regardless.
  function toViewBox(clientX: number, clientY: number): { x: number; y: number } {
    const svg = svgRef.current;
    const ctm = svg?.getScreenCTM();
    if (!svg || !ctm) return { x: 0, y: 0 };
    const pt = svg.createSVGPoint();
    pt.x = clientX;
    pt.y = clientY;
    const local = pt.matrixTransform(ctm.inverse());
    return { x: local.x, y: local.y };
  }

  // A native (non-passive) listener, not React's onWheel: browsers/React may
  // treat wheel listeners as passive by default, in which case
  // preventDefault() is silently ignored and the page scrolls underneath
  // instead of just the map zooming.
  // Shared by the wheel (desktop) and pinch (touch) gestures: multiplies the
  // zoom by `factor`, keeping the view-box point (mx, my) fixed on screen.
  // Kept in a ref so the wheel listener below — registered once per
  // projection/minZoom, not per render — always calls the latest closure.
  function applyZoom(mx: number, my: number, factor: number) {
    const [flightTx, flightTy] = projection.translate();
    setView((v) => {
      const raw = v.zoom * factor;
      if (raw <= minZoom) {
        // Snap to a properly centred whole-world view rather than
        // whatever pan the cursor-anchored math would otherwise leave.
        return { zoom: minZoom, pan: { x: WORLD_TRANSLATE[0] - minZoom * flightTx, y: WORLD_TRANSLATE[1] - minZoom * flightTy } };
      }
      const nz = Math.min(raw, MAX_ZOOM);
      // Anchor the point under the cursor: whatever local point is at
      // (mx,my) before this tick must still be at (mx,my) after it.
      const nextPan = { x: mx - (mx - v.pan.x) * (nz / v.zoom), y: my - (my - v.pan.y) * (nz / v.zoom) };
      return { zoom: nz, pan: clampPan(nextPan, nz) };
    });
  }
  const applyZoomRef = useRef(applyZoom);
  applyZoomRef.current = applyZoom;

  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;
    function onWheelNative(event: WheelEvent) {
      event.preventDefault();
      const { x: mx, y: my } = toViewBox(event.clientX, event.clientY);
      applyZoomRef.current(mx, my, Math.exp(-event.deltaY * 0.0015));
    }
    svg.addEventListener("wheel", onWheelNative, { passive: false });
    return () => svg.removeEventListener("wheel", onWheelNative);
  }, [projection, minZoom]);

  if (points.length < 2) {
    // margin: 0 matters here — the default <p> margins collapse through the
    // wrappers above and fall outside the measured content height that
    // sizes the Trajectory card's reveal, so the last line gets clipped.
    return <p style={{ color: "var(--text-faint)", fontSize: 14.5, margin: 0 }}>Not enough trajectory points to draw a path.</p>;
  }

  // Altitude drives both hue and lightness: light green on the ground,
  // fairly dark blue at cruise — a conventional altitude reading, decoupled
  // from anomaly severity (which still shows on the current-position marker).
  function colorAt(alt: number | null): string {
    const t = alt === null || altMax <= altMin ? 0.5 : (alt - altMin) / (altMax - altMin);
    const h = ALT_HUE_GROUND + t * (ALT_HUE_CRUISE - ALT_HUE_GROUND);
    const l = ALT_LIGHT_GROUND + t * (ALT_LIGHT_CRUISE - ALT_LIGHT_GROUND);
    return `hsl(${h.toFixed(0)} ${ALT_SAT}% ${l.toFixed(0)}%)`;
  }

  function pinchState() {
    const [a, b] = [...pointersRef.current.values()];
    return { dist: Math.hypot(a.x - b.x, a.y - b.y), mid: { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 } };
  }

  function handlePointerDown(event: React.PointerEvent<SVGSVGElement>) {
    pointersRef.current.set(event.pointerId, { x: event.clientX, y: event.clientY });
    event.currentTarget.setPointerCapture(event.pointerId);
    if (pointersRef.current.size === 2) {
      // Second finger down: switch from dragging to pinching.
      draggingRef.current = false;
      setIsDragging(false);
      pinchRef.current = pinchState();
      return;
    }
    draggingRef.current = true;
    setIsDragging(true);
    lastPosRef.current = toViewBox(event.clientX, event.clientY);
  }

  function handlePointerMove(event: React.PointerEvent<SVGSVGElement>) {
    if (pointersRef.current.has(event.pointerId)) {
      pointersRef.current.set(event.pointerId, { x: event.clientX, y: event.clientY });
    }
    if (pointersRef.current.size >= 2 && pinchRef.current) {
      const next = pinchState();
      const prev = pinchRef.current;
      pinchRef.current = next;
      if (prev.dist > 0 && next.dist > 0) {
        const { x: mx, y: my } = toViewBox(next.mid.x, next.mid.y);
        applyZoom(mx, my, next.dist / prev.dist);
        // Two fingers sliding together pan the map too, like a native one.
        const before = toViewBox(prev.mid.x, prev.mid.y);
        const dx = mx - before.x;
        const dy = my - before.y;
        setView((v) => ({ ...v, pan: clampPan({ x: v.pan.x + dx, y: v.pan.y + dy }, v.zoom) }));
      }
      return;
    }
    const { x, y } = toViewBox(event.clientX, event.clientY);
    if (draggingRef.current) {
      const dx = x - lastPosRef.current.x;
      const dy = y - lastPosRef.current.y;
      lastPosRef.current = { x, y };
      setView((v) => ({ ...v, pan: clampPan({ x: v.pan.x + dx, y: v.pan.y + dy }, v.zoom) }));
      return;
    }
    // hit-test in the flight's own data space, i.e. undo the current pan/zoom
    setHoverIdx(nearestIndex(points, (x - pan.x) / zoom, (y - pan.y) / zoom));
  }

  function endDrag(event?: React.PointerEvent<SVGSVGElement>) {
    if (event) pointersRef.current.delete(event.pointerId);
    pinchRef.current = null;
    // One finger left after a pinch: carry on as a drag from where it is,
    // rather than leaving the map stuck until the next touch.
    if (pointersRef.current.size === 1) {
      const [remaining] = [...pointersRef.current.values()];
      lastPosRef.current = toViewBox(remaining.x, remaining.y);
      draggingRef.current = true;
      setIsDragging(true);
      return;
    }
    draggingRef.current = false;
    setIsDragging(false);
  }

  function handleKeyDown(event: React.KeyboardEvent<SVGSVGElement>) {
    if (event.key === "ArrowRight") setHoverIdx((i) => Math.min((i ?? -1) + 1, points.length - 1));
    else if (event.key === "ArrowLeft") setHoverIdx((i) => Math.max((i ?? points.length) - 1, 0));
  }

  function resetView() {
    setView({ zoom: 1, pan: { x: 0, y: 0 } });
  }

  const hover = hoverIdx !== null ? points[hoverIdx] : null;
  const last = visiblePoints[visiblePoints.length - 1];
  // While the reveal is animating, orient the plane along the exact segment
  // it's currently crossing (smooth, continuous rotation); once arrived,
  // switch to the noise-reduced last-5-samples heading used at rest — a
  // single raw segment can be noisy (near-duplicate points while taxiing).
  const segHeading = ((Math.atan2(segP1.x - segP0.x, -(segP1.y - segP0.y)) * 180) / Math.PI + 360) % 360;
  const heading = arrived ? currentHeading(trajectoire) : segHeading;
  const accent = `hsl(${hue.toFixed(0)} 85% 62%)`;
  const inv = 1 / zoom; // counter-scale so strokes/markers stay a constant size on screen at any zoom
  // The 960-unit-wide view box is drawn at roughly a third of that on a
  // phone, which shrinks the city names to a few pixels — scaled up there so
  // they read at about the size they have on desktop.
  const cityLabelScale = isMobile ? 2.08 : 1;

  return (
    <section>
      <div style={{ position: "relative" }}>
        <svg
          ref={svgRef}
          viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
          role="img"
          aria-label={`Flight path, ${points.length} points, altitude between ${Math.round(altMin)} and ${Math.round(altMax)} meters`}
          tabIndex={0}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={endDrag}
          onPointerCancel={endDrag}
          onPointerLeave={() => !draggingRef.current && setHoverIdx(null)}
          onKeyDown={handleKeyDown}
          style={{
            width: "100%",
            height: "auto",
            display: "block",
            background: "var(--bg)",
            borderRadius: 12,
            cursor: isDragging ? "grabbing" : "grab",
            outline: "none",
            touchAction: "none",
            // Dragging to pan can sweep the pointer across the city-name
            // labels and select their text like any other text on the page
            // — not what a drag-to-pan gesture should do anywhere on the map.
            userSelect: "none",
            WebkitUserSelect: "none",
          }}
        >
          <defs>
            <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="3.5" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          <g transform={`translate(${pan.x} ${pan.y}) scale(${zoom})`}>
            {/* minimalist map: border outlines, plus a barely-there white
                tint for the sea, with the land shape painted back over it in
                the page's own background colour — so only the ocean picks up
                the tint, not the whole map. Coastlines drawn thicker than
                internal country borders.
                The tint used to be one <rect> per tile (each oversized by a
                hair so neighbours would overlap instead of exactly abut) —
                but per-tile rects each get their own edge antialiasing from
                the browser's own rasterizer, and at the extreme zoom-out
                needed to see several tiles at once, two independently
                antialiased edges meeting (even with a real, correctly
                computed overlap in local coordinates) can still composite to
                a visibly darker seam, because the antialiasing happens after
                that overlap math, at the rendered pixel level. Confirmed
                directly: re-rendering the exact same SVG markup into an
                offscreen canvas showed no seam at all, only the browser's own
                on-screen compositing did — a rasterizer artifact, not a
                geometry bug. Since the tint is the exact same flat colour on
                every tile anyway, the actual fix is to not tile it at all:
                one single rect spanning the whole grid has no inter-tile
                edges left to seam. Sized to exactly match the tile grid
                extent, same as clampPan's own margins below (in flight-
                space units there; WORLD_TILE_PERIOD here, since this whole
                group is wrapped in the align transform to convert).
                The map itself (rect, land, borders, coastlines) is drawn in
                WORLD_FIT's own coordinate space and carried into the
                flight's space by this one wrapping transform — see
                worldLandPath's comment for why. `inv` alone would counter
                only the outer scale(zoom); dividing by `ratio` too counters
                this extra scale(ratio) layer, so strokes stay a constant
                width on screen at any zoom, same as everywhere else. */}
            <g transform={`translate(${alignDx} ${alignDy}) scale(${ratio})`}>
              <rect
                x={-2.5 * WORLD_TILE_PERIOD}
                y={-1.5 * WORLD_TILE_PERIOD}
                width={5 * WORLD_TILE_PERIOD}
                height={3 * WORLD_TILE_PERIOD}
                fill="rgba(255, 255, 255, 0.045)"
              />
              {WORLD_TILE_OFFSETS.map(({ dx, dy }) => (
                <g key={`${dx}-${dy}`} transform={`translate(${dx} ${dy})`}>
                  <path d={worldLandPath} fill="var(--bg)" stroke="none" />
                  <path d={worldInternalPath} fill="none" stroke="rgba(255,255,255,0.22)" strokeWidth={(0.6 * inv) / ratio} strokeLinejoin="round" />
                  <path d={worldCoastPath} fill="none" stroke="rgba(255,255,255,0.4)" strokeWidth={(1.3 * inv) / ratio} strokeLinejoin="round" />
                </g>
              ))}
            </g>

            {/* diffuse glow under the path — only the revealed portion, from
                takeoff to the animated current position */}
            <g filter="url(#glow)" opacity={0.55}>
              {visiblePoints.slice(1).map((p, i) => (
                <line key={`g${i}`} x1={visiblePoints[i].x} y1={visiblePoints[i].y} x2={p.x} y2={p.y} stroke={colorAt(p.altitude)} strokeWidth={4 * inv} strokeLinecap="round" />
              ))}
            </g>
            {/* crisp path */}
            {visiblePoints.slice(1).map((p, i) => (
              <line key={i} x1={visiblePoints[i].x} y1={visiblePoints[i].y} x2={p.x} y2={p.y} stroke={colorAt(p.altitude)} strokeWidth={2.4 * inv} strokeLinecap="round" />
            ))}

            {/* origin */}
            <circle cx={points[0].x} cy={points[0].y} r={4 * inv} fill="var(--bg)" stroke="var(--text-muted)" strokeWidth={1.5 * inv} />
            {origineVille && (
              <g transform={`translate(${points[0].x} ${points[0].y}) scale(${inv * cityLabelScale})`} opacity={0.75}>
                <text
                  x={9}
                  y={-8}
                  fontFamily="var(--font-mono)"
                  fontSize={10.5}
                  fontWeight={300}
                  letterSpacing={1}
                  fill="var(--text)"
                  stroke="var(--bg)"
                  strokeWidth={3}
                  paintOrder="stroke"
                  style={{ textTransform: "uppercase" }}
                >
                  {origineVille}
                </text>
              </g>
            )}
            {destinationVille && arrived && (
              <g transform={`translate(${last.x} ${last.y}) scale(${inv * cityLabelScale})`} opacity={0.75}>
                <text
                  x={9}
                  y={-8}
                  fontFamily="var(--font-mono)"
                  fontSize={10.5}
                  fontWeight={300}
                  letterSpacing={1}
                  fill="var(--text)"
                  stroke="var(--bg)"
                  strokeWidth={3}
                  paintOrder="stroke"
                  style={{ textTransform: "uppercase" }}
                >
                  {destinationVille}
                </text>
              </g>
            )}

            {/* current position: a slow, thin, discreet pulsing ring once
                the reveal has finished and the marker is actually resting
                at the current position — shown only for a flight still
                airborne (statut === "en_vol"), never for one that's landed,
                since it reads as "this is a live position" specifically.
                Pulsing while it's still travelling along the path would
                read as a second, unrelated animation, hence gated on
                `arrived` too. */}
            {enVol && arrived && (
              <circle cx={last.x} cy={last.y} r={12 * inv} fill="none" stroke={accent} strokeWidth={1 * inv} filter="url(#glow)">
                <animate attributeName="r" values={`${10 * inv};${26 * inv}`} dur="2.6s" repeatCount="indefinite" />
                <animate attributeName="opacity" values="0.9;0" dur="2.6s" repeatCount="indefinite" />
              </circle>
            )}
            <g transform={`translate(${last.x} ${last.y}) rotate(${heading}) scale(${inv})`} filter="url(#glow)">
              <path d="M0,-11 L7,9 L0,4.5 L-7,9 Z" fill={accent} />
            </g>

            {hover && <circle cx={hover.x} cy={hover.y} r={5 * inv} fill={accent} stroke="var(--bg)" strokeWidth={2 * inv} />}
          </g>
        </svg>

        {(zoom !== 1 || pan.x !== 0 || pan.y !== 0) && (
          <button
            type="button"
            onClick={resetView}
            title="Reset view"
            style={{
              position: "absolute",
              top: 10,
              right: 10,
              display: "inline-flex",
              alignItems: "center",
              gap: 6,
              padding: "6px 10px",
              fontSize: 12,
              fontWeight: 600,
              color: "var(--text-muted)",
              background: "rgba(13, 15, 17, 0.85)",
              border: "1px solid var(--border-strong)",
              borderRadius: 8,
              cursor: "pointer",
            }}
          >
            <RotateCcw size={13} />
            Reset
          </button>
        )}
      </div>

      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 12, marginTop: 12, fontSize: 13, color: "var(--text-muted)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ textTransform: "uppercase", letterSpacing: 0.6, fontSize: 12 }}>Altitude</span>
          <span style={{ display: "inline-block", width: 120, height: 6, borderRadius: 3, background: `linear-gradient(to right, ${colorAt(altMin)}, ${colorAt(altMax)})` }} />
          <span style={{ fontFamily: "var(--font-mono)" }}>
            {Math.round(altMin)} – {Math.round(altMax)} m
          </span>
        </div>
        <div aria-live="polite" style={{ fontFamily: "var(--font-mono)", color: hover ? "var(--text)" : "var(--text-faint)" }}>
          {hover
            ? `${formatTime(hover.timestamp)}  ·  ${hover.altitude !== null ? `${Math.round(hover.altitude)} m` : "alt —"}  ·  ${hover.vitesse !== null ? `${Math.round(hover.vitesse * 3.6)} km/h` : "spd —"}`
            : isMobile
              ? "pinch to zoom"
              : "hover the path  ·  scroll to zoom  ·  ←/→ to scrub"}
        </div>
      </div>

      <details style={{ marginTop: 10 }}>
        <summary style={{ cursor: "pointer", color: "rgba(255, 255, 255, 0.7)", fontSize: 13 }}>Raw data ({points.length} points)</summary>
        <div style={{ maxHeight: 220, overflowY: "auto", marginTop: 8 }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13, fontFamily: "var(--font-mono)" }}>
            <thead>
              <tr style={{ textAlign: "left", color: "var(--text-faint)" }}>
                <th style={{ padding: "4px 8px", fontWeight: 500 }}>Time</th>
                <th style={{ padding: "4px 8px", fontWeight: 500 }}>Altitude (m)</th>
                <th style={{ padding: "4px 8px", fontWeight: 500 }}>Speed (km/h)</th>
              </tr>
            </thead>
            <tbody>
              {trajectoire.map((p, i) => (
                <tr key={i} style={{ borderTop: "1px solid var(--border)" }}>
                  <td style={{ padding: "3px 8px" }}>{formatTime(p.timestamp)}</td>
                  <td style={{ padding: "3px 8px" }}>{p.altitude !== null ? Math.round(p.altitude) : "—"}</td>
                  <td style={{ padding: "3px 8px" }}>{p.vitesse !== null ? Math.round(p.vitesse * 3.6) : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </section>
  );
}
