// CO2 weather viewer — renders a frame of the simulated excess-CO2 field as
// semi-transparent wildfire smoke over the globe (pale thin haze -> tan -> amber
// -> thick brown-gray, orange over active fire). A faint fresnel rim defines the limb.
//
// Day/night: the globe is lit by a sun whose direction is a pure function of the
// frame's (day, hour_utc) (matching sim/wind.py's subsolar latitude and the
// sources' local-solar-time model), so the terminator sweeps across the fixed
// globe as the movie's clock runs. The CO2 smoke and the atmosphere rim are
// sun-shaded too. ?daynight=0 reverts to the old flat always-lit look;
// ?sunHour=/  ?sunDay= pin the sun for static A/B; ?smokeSun=0 stops smoke shading.
//
// Two haze renderers, switch with HAZE_MODE (?haze=sprites|shell):
//   'shell' (default): the field splatted onto a 1024x512 equirect grid and drawn
//     as an alpha-blended shell. The splat kernel is circular in ARC length (the
//     longitude axis is widened by 1/cos(lat) toward the pole), so the poles render
//     as smooth caps instead of the old radial "starburst", and the body is a
//     continuous field -- no concentric Moire rings.
//   'sprites': one soft point-sprite per node, drawn at the node positions. Pole-
//     clean, but large overlapping sprites alias into concentric "lens" rings and
//     small ones expose the dot lattice -- so it's the fallback, not the default.
//
// Coordinate frame (matches the sim's node positions): x = coslat*coslon,
// y = coslat*sinlon, z = sinlat  =>  north pole at +z, lon 0 at +x.
//
// The field is peaky (a few hot nodes, most of the globe near zero). Splatting
// it onto a high-resolution equirect grid with a tight gaussian keeps the plumes
// detailed instead of smearing them into blobs, and alpha (not additive) blending
// over the terrain reads as smoke drifting over the surface rather than a glow.
//
// Capture contract (used by capture/capture.py):
//   window.__ready    -> true once meta+field are loaded and frame 0 is drawn
//   window.__settle(i)-> async; draws frame i, waits for GPU, resolves true
// The capture script loads the plain URL (no ?live), so no auto-advance runs
// and every frame is deterministic (no Math.random in the draw path).

import * as THREE from '/three/build/three.module.js';

// --- tunable look (tune here, then re-render) ------------------------------
const SHELL_REF_F = 0.30;     // haze normalizer = col_ref * f  (lower = broader haze)
const SHELL_EXPOSURE = 1.0;   // density slope (higher = more concentrated / more visible)
const CAM_DIST = 3.35;
const FOOV = 46;              // vertical field of view, degrees
// map-view focus: southern Europe / central Mediterranean (lat, lon in degrees)
// -- a more equatorial vantage than the old Oslo (59.9N) view, still Europe-centered
const FOCUS_LAT = 42.0, FOCUS_LON = 13.0;

// HAZE_MODE -- how the CO2 field is drawn (default 'shell'; override ?haze=sprites):
//   'shell'   : the field splatted onto a 1024x512 equirect grid and drawn as an
//               alpha-blended shell. Continuous field, and the polar cap is
//               zonal-meaned (flattenPoles) so it's smooth -- no "star". Default.
//   'sprites' : soft point-sprites at the node positions. Pole-clean, but large
//               sprites alias into concentric "lens" rings and small ones expose
//               the dot lattice, so it's the fallback.
// --- URL overrides for tuning (e.g. ?haze=sprites&fresnel=0&ref=1200) -------
const _Q = (typeof location !== 'undefined') ? new URLSearchParams(location.search) : null;
const _qn = (k, d) => { const v = _Q && _Q.get(k); return (v === null || v === '') ? d : parseFloat(v); };
const HAZE_MODE = (_Q && _Q.get('haze')) || 'shell';   // shell = continuous field; sprites = ?haze=sprites
const SPRITE_RADIUS_DEG = _qn('spriteR', 5.0);   // angular radius of each smoke sprite (degrees)
const SPRITE_OPACITY = _qn('spriteOp', 0.55);    // per-sprite alpha multiplier (0..1)
const FRESNEL_GAIN = _qn('fresnel', 0.30);       // atmosphere rim gain (0 disables)
const FRESNEL_POW  = _qn('fresnelPow', 3.5);
const FRESNEL_RAD  = _qn('fresnelRad', 1.012);

// --- day/night sun (tune here, then re-render) -----------------------------
// The sun's direction is a pure function of the frame's (day, hour_utc), so the
// day/night terminator sweeps across the fixed globe as the movie's clock runs
// -- and every capture frame stays deterministic (no wall clock, no Math.random).
// The model matches the sim: subsolar latitude 23.44*sin(2pi(day-80)/365)
// (sim/wind.py:subsolar_lat) and local solar time h = hour_utc + lon/15
// (sim/sources.py), so the subsolar longitude is (12 - hour_utc)*15.
const DAYNIGHT = !(_Q && _Q.get('daynight') === '0');   // ?daynight=0 -> old flat look (A/B)
const SMOKE_SUN = _qn('smokeSun', 1.0);                 // 0 = don't sun-shade the CO2 smoke
const NIGHT_LIGHT = _qn('nightLight', 0.30);            // night-side "moonlight" fill; 0 = near-black
const SKY = !(_Q && _Q.get('sky') === '0');             // ?sky=0 -> plain dark background (no Milky Way)
const _SUN_DAY   = (_Q && _Q.get('sunDay'))   ? parseFloat(_Q.get('sunDay'))   : null;  // static override
const _SUN_HOUR  = (_Q && _Q.get('sunHour'))  ? parseFloat(_Q.get('sunHour'))  : null;  // static override

// Sun direction (unit vector) in the globe frame at (day, hour_utc).
function sunDirection(day, hourUtc) {
  if (_SUN_DAY  != null) day = _SUN_DAY;
  if (_SUN_HOUR != null) hourUtc = _SUN_HOUR;
  const decl = THREE.MathUtils.degToRad(23.44 * Math.sin(2 * Math.PI * (day - 80.0) / 365.0));
  const lonSub = THREE.MathUtils.degToRad((12.0 - hourUtc) * 15.0);   // where local time is 12:00
  return new THREE.Vector3(
    Math.cos(decl) * Math.cos(lonSub),
    Math.cos(decl) * Math.sin(lonSub),
    Math.sin(decl),
  ).normalize();
}

// Shared sun uniforms, referenced by every lit material so one update lights them all.
const uSunDir   = { value: new THREE.Vector3(1, 0, 0) };
const uDaynight = { value: DAYNIGHT ? 1.0 : 0.0 };
const uSmokeSun = { value: SMOKE_SUN };
const uNightLight = { value: NIGHT_LIGHT };
let frames = null;            // meta.frames[i] = {day, hour}, set in init()

const canvas = document.getElementById('c');
const renderer = new THREE.WebGLRenderer({
  canvas, antialias: true, preserveDrawingBuffer: true,
});
renderer.setPixelRatio(1);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x02040a, 1);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x02040a);

const camera = new THREE.PerspectiveCamera(FOOV, window.innerWidth / window.innerHeight, 0.1, 200);
let uScale = 1.0;             // world-size -> pixel scale, set after sizing

// --- state -----------------------------------------------------------------
let N = 0, M = 0, ref = 1.0;
let base = null;              // Float32Array (N*3) unit node positions
let aColor, aAlpha, aSize, positions;
let cloudGeo = null;
let cloudMat = null;        // cloud material (update its uScale uniform on resize)
let uIdx = null, vIdx = null; // per-node equirect grid col/row (float), for splat
let shellField = null;        // Float32Array GW*GH, per-frame haze accumulator
let shellData = null;         // Uint8Array GW*GH*4, shell texture pixels
let shellTex = null;

// --- shared color ramp (intensity 0..1 -> rgb) -----------------------------
// cool base -> warm -> hot; used by both the haze and the plumes so they agree.
function ramp(I) {
  I = Math.max(0, Math.min(1, I));
  const s = [
    [0.00, 0.14, 0.30, 0.55],
    [0.42, 0.38, 0.50, 0.62],
    [0.64, 0.80, 0.70, 0.50],
    [0.84, 1.00, 0.72, 0.34],
    [1.00, 1.00, 0.90, 0.64],
  ];
  for (let i = 0; i < s.length - 1; i++) {
    if (I <= s[i + 1][0]) {
      const t = (I - s[i][0]) / (s[i + 1][0] - s[i][0]);
      return [s[i][1] + (s[i+1][1] - s[i][1]) * t,
              s[i][2] + (s[i+1][2] - s[i][2]) * t,
              s[i][3] + (s[i+1][3] - s[i][3]) * t];
    }
  }
  return [s[s.length-1][1], s[s.length-1][2], s[s.length-1][3]];
}
function smoothstep(e0, e1, x) {
  const t = Math.max(0, Math.min(1, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
}

// --- wildfire-smoke palette (density 0..1 -> rgb / alpha) ------------------
// pale thin haze -> tan -> amber -> thick brown-gray -> orange over active fire.
// Alpha is capped (~0.9) so smoke reads as semi-opaque layers over the terrain
// instead of an additive bloom that clips to white.
function smokeColor(I) {
  I = Math.max(0, Math.min(1, I));
  const s = [
    [0.00, 0.86, 0.83, 0.74],
    [0.30, 0.74, 0.66, 0.50],
    [0.55, 0.60, 0.52, 0.38],
    [0.75, 0.50, 0.42, 0.34],
    [0.90, 0.72, 0.46, 0.24],
    [1.00, 0.85, 0.42, 0.18],
  ];
  for (let i = 0; i < s.length - 1; i++) {
    if (I <= s[i + 1][0]) {
      const t = (I - s[i][0]) / (s[i + 1][0] - s[i][0]);
      return [s[i][1] + (s[i+1][1] - s[i][1]) * t,
              s[i][2] + (s[i+1][2] - s[i][2]) * t,
              s[i][3] + (s[i+1][3] - s[i][3]) * t];
    }
  }
  const l = s[s.length - 1];
  return [l[1], l[2], l[3]];
}
function smokeAlpha(I) {
  return Math.min(0.97, smoothstep(0.02, 0.35, I));
}

// --- deterministic helpers -------------------------------------------------
// small integer hash -> [0,1); used for per-plume variation (no Math.random).
function hash01(n) {
  let x = (n + 1) * 0x6d2b79f5;
  x = Math.imul(x ^ (x >>> 13), 0x5bd1e995);
  x ^= x >>> 15;
  return (x >>> 0) / 4294967296;
}

// Load an image as a THREE.Texture, awaiting full decode so the globe is fully
// textured before the first render (deterministic capture frames). The JPEG is
// stored as-is (linear-sRGB) so texture2D returns the raw sRGB bytes, matching
// how the shell bakes its display colors and this renderer writes raw to the
// framebuffer (ShaderMaterial -> no output color-space conversion).
function loadTexture(url) {
  return new Promise((resolve, reject) => {
    new THREE.TextureLoader().load(url,
      (tex) => { tex.colorSpace = THREE.LinearSRGBColorSpace; resolve(tex); },
      undefined,
      (err) => reject(new Error('failed to load texture ' + url + ': ' + err)));
  });
}

// --- surface-glow splat (field -> equirect shell) --------------------------
// The splat kernel is a gaussian that is circular in ARC LENGTH, not in grid
// cells. In equirect, a grid cell in longitude spans cos(lat) times less arc
// than one in latitude, so a cell-circular kernel near the pole smears into a
// radial spoke (the old north-pole "starburst"). Widen the longitude axis by
// 1/cos(lat) and the kernel stays a round cap -> smooth poles. At the equator
// (cos=1) this is byte-identical to the old circular kernel, so the body is
// unchanged. Each node's kernel is normalized to the old kernel's total mass
// (CIRC_SUM) so brightness is consistent from equator to pole.
const GW = 1024, GH = 512;
const SIG = 3.5;                 // arc sigma, in grid-cell units at the equator
const KER_R = 10;                // half-width of the (equator) kernel, in cells
const COSLAT_MIN = 0.10;         // clamp cos(lat): smaller -> wider pole kernel -> smoother cap
const _ir = 1 / (2 * SIG * SIG);
const wr = new Float32Array(2 * KER_R + 1);   // constant latitude-axis weights
let _sr = 0;
for (let dy = -KER_R; dy <= KER_R; dy++) { const w = Math.exp(-_ir * dy * dy); wr[dy + KER_R] = w; _sr += w; }
const CIRC_SUM = _sr * _sr;      // total mass of the old cell-circular kernel
const _WC = new Float32Array(GW + 1);         // scratch for per-node longitude weights
let cosLat = null;               // (N,) cos(lat) per node, set in computeGrid()

function splatNode(n, val) {
  const cx0 = Math.floor(uIdx[n]), cy0 = Math.floor(vIdx[n]);
  const sigC = SIG / Math.max(cosLat[n], COSLAT_MIN);   // wider in lon near the pole
  const ic = 1 / (2 * sigC * sigC);
  const Rc = Math.min(GW >> 1, Math.ceil(2.8 * sigC));
  let sc = 0;
  for (let dx = -Rc; dx <= Rc; dx++) { const w = Math.exp(-ic * dx * dx); _WC[dx + Rc] = w; sc += w; }
  const w0 = val * CIRC_SUM / (sc * _sr);
  for (let dy = -KER_R; dy <= KER_R; dy++) {
    const row = ((cy0 + dy) % GH + GH) % GH;
    const base = row * GW, wrow = w0 * wr[dy + KER_R];
    for (let dx = -Rc; dx <= Rc; dx++) {
      const c = ((cx0 + dx) % GW + GW) % GW;
      shellField[base + c] += wrow * _WC[dx + Rc];
    }
  }
}

// Clean the poles: only ~10 nodes ring each pole (at 86-89 deg), so their
// azimuthal modulation shows as a radial "star". Measured on the raw field, the
// azimuthal std is ~10-26 in the top ~16 rows (84-90 deg) = the star, then
// JUMPS to ~40-57 just below 82 deg = the REAL plume structure (European/Asian
// CO2, genuinely azimuthally structured). So:
//   * zonal-mean (row -> its azimuthal mean) ONLY the star region (POLE_BAND
//     rows), killing the spurious spokes while preserving the real cap mean;
//   * FADE that zonal-meaning out over POLE_FADE extra rows, so there is NO hard
//     border and the real high-latitude plume structure below is left intact.
// Net: a small, smooth Arctic cap -- not a big disc with a sharp edge through
// Europe (the old POLE_BAND=96 flattened 56-90N and left a hard line at 56N).
const POLE_BAND = _qn('poleband', 16);   // rows fully zonal-meaned (the star, 84-90N)
const POLE_FADE = 6;                     // extra rows where the zonal-mean fades out
function flattenPoles() {
  const TOTAL = POLE_BAND + POLE_FADE;
  for (const [start, dir] of [[0, 1], [GH - 1, -1]]) {
    const means = new Float32Array(TOTAL);
    for (let b = 0; b < TOTAL; b++) {
      const base = (start + dir * b) * GW;
      let m = 0;
      for (let c = 0; c < GW; c++) m += shellField[base + c];
      means[b] = m / GW;
    }
    // light smoothing of the mean profile (kills a local-min dot at the pole)
    const BW = 2, sm = new Float32Array(TOTAL);
    for (let b = 0; b < TOTAL; b++) {
      let s = 0, n = 0;
      for (let k = -BW; k <= BW; k++) { const r = b + k; if (r >= 0 && r < TOTAL) { s += means[r]; n++; } }
      sm[b] = s / n;
    }
    for (let b = 0; b < TOTAL; b++) {
      const base = (start + dir * b) * GW;
      const w = (b < POLE_BAND) ? 1 : Math.max(0, 1 - (b - POLE_BAND) / POLE_FADE);
      for (let c = 0; c < GW; c++) shellField[base + c] = shellField[base + c] * (1 - w) + sm[b] * w;
    }
  }
}

function buildShell() {
  shellField = new Float32Array(GW * GH);
  shellData = new Uint8Array(GW * GH * 4);
  shellTex = new THREE.DataTexture(shellData, GW, GH, THREE.RGBAFormat);
  shellTex.colorSpace = THREE.LinearSRGBColorSpace;   // raw bytes; we bake display colors
  shellTex.magFilter = THREE.LinearFilter;
  shellTex.minFilter = THREE.LinearFilter;
  shellTex.wrapS = THREE.RepeatWrapping;
  shellTex.wrapT = THREE.ClampToEdgeWrapping;
  shellTex.generateMipmaps = false;
  shellTex.needsUpdate = true;

  // UV: use SphereGeometry's built-in equirect uv, rotated so three.js's +y pole
  // (north) lands on the sim's +z north. This equals the node-frame equirect
  // (u=(atan2(y,x)+pi)/2pi, v=(asin(z)+pi/2)/pi) AND gives each pole a per-column
  // uv, so the fan triangles stay well-conditioned. Computing uv from position
  // instead collapses the pole to one uv (atan(0,0)=0) and the antimeridian fan
  // triangles smear the whole texture row -> the two streaks over the north pole.
  const geo = new THREE.SphereGeometry(1.015, 128, 96);
  geo.rotateX(Math.PI / 2);   // +y (three's pole) -> +z (sim north); see computeGrid()
  const mat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, depthTest: true,
    blending: THREE.NormalBlending, side: THREE.FrontSide,
    uniforms: { tex: { value: shellTex }, uSunDir: uSunDir, uDaynight: uDaynight, uSmokeSun: uSmokeSun },
    vertexShader: `
      varying vec2 vUV; varying vec3 vN;
      void main(){
        vUV = uv;
        vN = normalize(mat3(modelMatrix) * normal);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }`,
    fragmentShader: `
      varying vec2 vUV; varying vec3 vN;
      uniform sampler2D tex; uniform vec3 uSunDir; uniform float uDaynight; uniform float uSmokeSun;
      void main(){
        vec4 c = texture2D(tex, vUV);
        float lambert = max(dot(normalize(vN), uSunDir), 0.0);
        // floor 0.45 keeps the CO2 signal readable over the night side (data > realism)
        float shade = mix(1.0, (0.45 + 0.55 * lambert), uDaynight * uSmokeSun);
        gl_FragColor = vec4(c.rgb * shade, c.a);   // NormalBlending: src.rgb*src.a + dst*(1-src.a)
      }`,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.frustumCulled = false;
  return mesh;
}

// --- fresnel atmosphere (inner rim + outer halo), additive FrontSide, sun-aware:
// the day-side limb glows, the night-side rim stays a faint cool blue.
function buildAtmosphere(radius, pow, gain, hex) {
  const geo = new THREE.SphereGeometry(radius, 96, 64);
  const mat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, depthTest: true,
    blending: THREE.AdditiveBlending, side: THREE.FrontSide,
    uniforms: {
      uColor: { value: new THREE.Color(hex) },
      uPow: { value: pow }, uGain: { value: gain },
      uSunDir: uSunDir, uDaynight: uDaynight,
    },
    vertexShader: `
      varying vec3 vN; varying vec3 vV; varying vec3 vNw;
      void main(){
        vN = normalize(normalMatrix * normal);
        vNw = normalize(mat3(modelMatrix) * normal);      // world normal (for the sun)
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        vV = normalize(-mv.xyz);
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: `
      varying vec3 vN; varying vec3 vV; varying vec3 vNw;
      uniform vec3 uColor; uniform float uPow; uniform float uGain;
      uniform vec3 uSunDir; uniform float uDaynight;
      void main(){
        float f = 1.0 - clamp(dot(normalize(vN), normalize(vV)), 0.0, 1.0);
        float rim = pow(f, uPow);
        float sunlit = 0.5 + 0.5 * dot(normalize(vNw), uSunDir);   // 1 day side, 0 night
        float gain = uGain * mix(1.0, (0.15 + 0.85 * sunlit), uDaynight);
        vec3 col = uColor * mix(1.0, (0.55 + 0.75 * sunlit), uDaynight);
        gl_FragColor = vec4(col, rim * gain);
      }`,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.frustumCulled = false;
  return mesh;
}

// --- globe: earth texture on the SAME equirect UV as the CO2 shell (so the
// smoke sits over the right continents), lit by the moving sun for a
// realistic day/night terminator. uDaynight=0 -> the old flat, always-lit look.
function buildGlobe(earthTex) {
  const geo = new THREE.SphereGeometry(1.0, 128, 96);
  // Same built-in-uv + rotate as the shell: keeps the equirect poles well-
  // conditioned (no smear) and registered to the sim's +z-north node frame.
  geo.rotateX(Math.PI / 2);
  const mat = new THREE.ShaderMaterial({
    uniforms: { tex: { value: earthTex }, uSunDir: uSunDir, uDaynight: uDaynight, uNightLight: uNightLight },
    vertexShader: `
      varying vec2 vUV; varying vec3 vN;
      void main(){
        vUV = uv;
        vN = normalize(mat3(modelMatrix) * normal);   // world-space normal
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }`,
    fragmentShader: `
      varying vec2 vUV; varying vec3 vN;
      uniform sampler2D tex; uniform vec3 uSunDir; uniform float uDaynight; uniform float uNightLight;
      void main(){
        vec3 albedo = texture2D(tex, vUV).rgb;
        vec3 N = normalize(vN);
        float cosA = dot(N, uSunDir);                    // 1 subsolar, 0 terminator, -1 antipode
        float day  = smoothstep(-0.08, 0.14, cosA);      // 0 night -> 1 day, soft band
        float lambert = max(cosA, 0.0);                  // bright at subsolar, fades to the limb
        vec3 sunCol  = vec3(1.0, 0.985, 0.955);
        vec3 dayCol  = albedo * sunCol * (0.34 + 0.92 * lambert);
        // Night: a low "moonlight" fill (albedo-scaled, so land/sea contrast and
        // continent outlines stay readable) plus a faint cool ambient so it isn't
        // pure black. Still clearly darker/cooler than the sunlit day side.
        vec3 nightCol= albedo * uNightLight + vec3(0.012, 0.016, 0.028);
        float twi = day * (1.0 - day);                   // 0 at both ends, 1 in the band
        vec3 twiCol = vec3(1.0, 0.45, 0.20);             // sunset/sunrise warm tint
        vec3 lit = mix(nightCol, dayCol, day) + twi * twiCol * 0.12;
        gl_FragColor = vec4(mix(albedo, lit, uDaynight), 1.0);
      }`,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.frustumCulled = false;
  return mesh;
}

// --- Milky Way / starfield skybox ------------------------------------------
// A large inverted sphere with a procedurally-generated equirectangular sky:
// thousands of stars + a broad, soft Milky Way band (nebula glow + higher star
// density) + a few bright haloed stars. Fully offline (canvas) and deterministic
// (seeded RNG), so capture frames stay reproducible. Rendered with a raw
// ShaderMaterial (like the globe) to match the project's sRGB pipeline.
// (The old buildStars used size-attenuated points at r=40-60 -> sub-pixel and
// invisible. A texture on a skybox sidesteps that entirely.)
function _gauss(rng) {                               // Box-Muller, ~N(0,1)
  let u = 0, v = 0; while (u === 0) u = rng(); while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function makeSkyTexture(w = 4096, h = 2048) {
  const cv = document.createElement('canvas'); cv.width = w; cv.height = h;
  const ctx = cv.getContext('2d');
  const rng = mulberry32(0x51c45);                   // fixed seed -> deterministic sky

  // deep-space base: very dark blue, faint vertical gradient
  const bg = ctx.createLinearGradient(0, 0, 0, h);
  bg.addColorStop(0.0, '#04060d');
  bg.addColorStop(0.5, '#080b16');
  bg.addColorStop(1.0, '#04060d');
  ctx.fillStyle = bg; ctx.fillRect(0, 0, w, h);

  // Milky Way band: center row (px) as a function of column, tilted + arching.
  const bandTilt = 0.55, bandPhase = 0.30;
  const bandCenterY = (x) => h * 0.5 + (h * 0.5 * bandTilt) * Math.sin(2 * Math.PI * (x / w + bandPhase));
  const bandSigma = h * 0.10;                         // band half-width (px)

  // soft nebula glow along the band (additive)
  ctx.globalCompositeOperation = 'lighter';
  for (let i = 0, n = 46; i < n; i++) {
    const x = rng() * w, cy = bandCenterY(x);
    const y = cy + (rng() * 2 - 1) * bandSigma * (0.3 + rng());
    const r = h * (0.035 + 0.09 * rng());            // smaller puffs (no single giant blob)
    const roll = rng();
    const col = roll < 0.22 ? '255,214,170' : (roll < 0.6 ? '180,205,255' : '205,220,248');
    const a = 0.045 + 0.075 * rng();
    const g = ctx.createRadialGradient(x, y, 0, x, y, r);
    g.addColorStop(0, `rgba(${col},${a})`); g.addColorStop(1, `rgba(${col},0)`);
    ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
  }

  // stars: denser + brighter inside the band
  for (let i = 0, n = 9000; i < n; i++) {
    const x = rng() * w, cy = bandCenterY(x);
    const y = (rng() < 0.62) ? Math.max(0, Math.min(h, cy + _gauss(rng) * bandSigma * 0.8)) : rng() * h;
    const prox = Math.max(0, 1 - Math.abs(y - cy) / (bandSigma * 2.4));    // 1 on band, 0 far
    const bright = 0.12 + 0.88 * Math.pow(rng(), 2.5) * (0.45 + 0.55 * prox);
    const size = rng() < 0.985 ? (0.4 + rng() * 0.9) : (1.2 + rng() * 1.9); // a few larger
    const cr = (205 + 50 * rng()) | 0, cg = (210 + 45 * rng()) | 0;
    ctx.fillStyle = `rgba(${cr},${cg},255,${Math.max(0.05, Math.min(1, bright))})`;
    ctx.beginPath(); ctx.arc(x, y, size, 0, Math.PI * 2); ctx.fill();
  }

  // a handful of bright stars with a soft halo (kept modest so none reads as an orb)
  ctx.globalCompositeOperation = 'lighter';
  for (let i = 0; i < 42; i++) {
    const x = rng() * w, y = rng() * h, r = h * (0.003 + 0.006 * rng());
    const col = rng() < 0.3 ? '255,222,175' : '212,226,255';
    const a = 0.32 + 0.28 * rng();
    const g = ctx.createRadialGradient(x, y, 0, x, y, r * 2.6);
    g.addColorStop(0, `rgba(${col},${a})`); g.addColorStop(0.35, `rgba(${col},${a * 0.35})`); g.addColorStop(1, `rgba(${col},0)`);
    ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, r * 2.6, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = `rgba(255,255,255,${Math.min(0.95, a + 0.2)})`;
    ctx.beginPath(); ctx.arc(x, y, Math.max(0.6, r * 0.5), 0, Math.PI * 2); ctx.fill();
  }
  ctx.globalCompositeOperation = 'source-over';

  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.LinearSRGBColorSpace;       // raw bytes, like the earth texture
  tex.wrapS = THREE.RepeatWrapping; tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.magFilter = THREE.LinearFilter; tex.minFilter = THREE.LinearFilter;
  tex.generateMipmaps = false; tex.needsUpdate = true;
  return tex;
}

function buildSkybox(tex) {
  const geo = new THREE.SphereGeometry(60, 48, 32);
  const mat = new THREE.ShaderMaterial({
    uniforms: { tex: { value: tex } },
    side: THREE.BackSide, depthWrite: false,
    vertexShader: `
      varying vec2 vUV;
      void main(){ vUV = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
    fragmentShader: `
      varying vec2 vUV; uniform sampler2D tex;
      void main(){ gl_FragColor = vec4(texture2D(tex, vUV).rgb, 1.0); }`,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.frustumCulled = false;
  return mesh;
}

function buildClouds() {
  cloudGeo = new THREE.BufferGeometry();
  positions = new THREE.BufferAttribute(new Float32Array(N * 3), 3);
  positions.array.set(base);               // sprites sit at the (relaxed) node positions
  positions.needsUpdate = true;
  aColor = new THREE.BufferAttribute(new Float32Array(N * 3), 3);
  aAlpha = new THREE.BufferAttribute(new Float32Array(N), 1);
  aSize = new THREE.BufferAttribute(new Float32Array(N), 1);
  // gl_PointSize = aSize*uScale/-mv.z, so the sprite's world diameter is 2*aSize
  // independent of distance; aSize = sin(radius) gives an angular radius of `deg`.
  aSize.array.fill(Math.sin(THREE.MathUtils.degToRad(SPRITE_RADIUS_DEG)));
  cloudGeo.setAttribute('position', positions);
  cloudGeo.setAttribute('aColor', aColor);
  cloudGeo.setAttribute('aAlpha', aAlpha);
  cloudGeo.setAttribute('aSize', aSize);
  const mat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, depthTest: true,
    blending: THREE.NormalBlending,       // alpha smoke over terrain (reads as smoke, not glow)
    uniforms: { uScale: { value: uScale }, uSunDir: uSunDir, uDaynight: uDaynight, uSmokeSun: uSmokeSun },
    vertexShader: `
      attribute vec3 aColor; attribute float aAlpha; attribute float aSize;
      varying vec3 vColor; varying float vAlpha; varying vec3 vN; uniform float uScale;
      void main(){
        vColor = aColor; vAlpha = aAlpha;
        vN = normalize(position);        // sphere centered at origin -> radial normal
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        gl_Position = projectionMatrix * mv;
        gl_PointSize = aSize * (uScale / -mv.z);
      }`,
    fragmentShader: `
      varying vec3 vColor; varying float vAlpha; varying vec3 vN;
      uniform vec3 uSunDir; uniform float uDaynight; uniform float uSmokeSun;
      void main(){
        vec2 uv = gl_PointCoord - 0.5;
        float r = length(uv) * 2.0;      // 0 at center, 1 at the edge
        float a = exp(-6.0 * r * r);     // soft gaussian puff, a(1)=~0.0025
        if (a < 0.004) discard;
        float lambert = max(dot(normalize(vN), uSunDir), 0.0);
        float shade = mix(1.0, (0.45 + 0.55 * lambert), uDaynight * uSmokeSun);
        gl_FragColor = vec4(vColor * shade, vAlpha * a);
      }`,
  });
  cloudMat = mat;
  const pts = new THREE.Points(cloudGeo, mat);
  pts.frustumCulled = false;
  return pts;
}

// deterministic RNG (kept for the starfield)
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// --- per-frame draw ---------------------------------------------------------
// `col` is the current frame's column-integrated CO2 (N,). Loaded on demand.
function setFrame(col) {
  const Nf = N;
  const hazRef = ref * SHELL_REF_F;   // same normalizer the shell used

  if (HAZE_MODE === 'shell') {
    // 1) HAZE: splat the node field onto the equirect grid (arc-circular,
    //    pole-aware kernel), soft-clip, bake RGBA.
    shellField.fill(0);
    for (let n = 0; n < Nf; n++) {
      const val = col[n];
      if (val <= 0) continue;
      splatNode(n, val);
    }
    flattenPoles();   // zonal-mean the polar cap -> removes the discrete-node "star"
    // 2) BAKE: density -> smoke color + capped alpha (semi-opaque, over the terrain).
    for (let c = 0; c < GW * GH; c++) {
      const I = 1 - Math.exp(-SHELL_EXPOSURE * (shellField[c] / hazRef));
      const sm = smokeColor(I);
      shellData[c * 4 + 0] = (sm[0] * 255) | 0;
      shellData[c * 4 + 1] = (sm[1] * 255) | 0;
      shellData[c * 4 + 2] = (sm[2] * 255) | 0;
      shellData[c * 4 + 3] = (smokeAlpha(I) * 255) | 0;
    }
    shellTex.needsUpdate = true;
  } else {
    // SPRITES: each node is one soft point-sprite at its (relaxed) position.
    // Color/alpha come from the node's own column density, soft-clipped exactly
    // as the shell was; overlapping sprites composite into a continuous field.
    // This never touches the equirect grid, so the poles stay clean.
    const aC = aColor.array, aA = aAlpha.array;
    for (let n = 0; n < Nf; n++) {
      const val = col[n];
      const I = 1 - Math.exp(-SHELL_EXPOSURE * (val / hazRef));
      if (val <= 0 || I < 1e-4) { aA[n] = 0; continue; }
      const sm = smokeColor(I);
      aC[n * 3] = sm[0]; aC[n * 3 + 1] = sm[1]; aC[n * 3 + 2] = sm[2];
      aA[n] = smokeAlpha(I) * SPRITE_OPACITY;
    }
    aColor.needsUpdate = true;
    aAlpha.needsUpdate = true;
  }

  renderer.render(scene, camera);
}

// --- equirect grid index per node (matches the shell shader's UV exactly) ---
// u = (lon + pi) / 2pi, v = (lat + pi/2) / pi  ->  col in [0,GW), row in [0,GH)
function computeGrid() {
  uIdx = new Float32Array(N);
  vIdx = new Float32Array(N);
  cosLat = new Float32Array(N);
  const PI = Math.PI;
  for (let n = 0; n < N; n++) {
    const x = base[n * 3], y = base[n * 3 + 1], z = base[n * 3 + 2];
    const lon = Math.atan2(y, x);                            // [-pi, pi]
    const lat = Math.asin(THREE.MathUtils.clamp(z, -1, 1));  // [-pi/2, pi/2]
    uIdx[n] = (lon + PI) / (2 * PI) * GW;
    vIdx[n] = (lat + PI / 2) / PI * GH;
    cosLat[n] = Math.sqrt(Math.max(0.0, x * x + y * y));     // cos(lat); 1 at equator
  }
}

// --- sizing / point scale ---------------------------------------------------
function onResize() {
  const w = window.innerWidth, h = window.innerHeight;
  renderer.setPixelRatio(1);
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  uScale = h / Math.tan(THREE.MathUtils.degToRad(FOOV) * 0.5);
  if (cloudMat) cloudMat.uniforms.uScale.value = uScale;
  if (window.__ready) renderer.render(scene, camera);
}

// wait two frames so the GPU has presented the last render (deterministic shot)
function nextFrame() {
  return new Promise((res) =>
    requestAnimationFrame(() => requestAnimationFrame(res)));
}

// --- live auto-advance (only when ?live is in the URL) ----------------------
let liveFrame = 0, liveLast = 0, liveStepMs = 250;
async function startLive(fps) {
  liveStepMs = 1000 / (fps || 4);
  liveFrame = 0; liveLast = performance.now();
  async function tick(now) {
    if (now - liveLast >= liveStepMs) {
      liveLast = now;
      liveFrame = (liveFrame + 1) % M;
      await loadFrame(liveFrame);
    }
    requestAnimationFrame(tick);
  }
  tick(performance.now());
}

// Point the sun at this frame's (day, hour). Pure function of the frame time, so
// capture stays deterministic and the terminator sweeps as the movie advances.
function updateSun(i) {
  if (!frames) return;
  const fr = frames[i] || { day: 150, hour: 12 };
  uSunDir.value.copy(sunDirection(fr.day, fr.hour));
}

// --- load one frame on demand (keeps the browser's memory to a single frame) --
async function loadFrame(i) {
  updateSun(i);
  const buf = await (await fetch('/api/frame/' + i)).arrayBuffer();
  const d = new Float32Array(buf);        // (2, N) = [col, hfrac]
  setFrame(d.subarray(0, N));             // col drives the haze
}

// --- init ------------------------------------------------------------------
async function init() {
  try {
    const meta = await (await fetch('/api/meta')).json();
    N = meta.N; M = meta.M;
    frames = meta.frames;                 // per-frame {day, hour} -> drives the sun
    ref = (_Q && _Q.get('ref')) ? parseFloat(_Q.get('ref')) : meta.col_ref;  // ?ref= to override
    base = new Float32Array(meta.nodes);
    if (base.length !== N * 3)
      throw new Error('nodes length ' + base.length + ' != N*3 ' + N * 3);

    uScale = window.innerHeight / Math.tan(THREE.MathUtils.degToRad(FOOV) * 0.5);

    // scene assembly (globe is opaque; smoke + a faint rim are semi-transparent on top)
    const earthTex = await loadTexture('/earth.jpg');
    scene.add(buildGlobe(earthTex));
    if (HAZE_MODE === 'shell') scene.add(buildShell());
    else scene.add(buildClouds());
    if (FRESNEL_GAIN > 0)
      scene.add(buildAtmosphere(FRESNEL_RAD, FRESNEL_POW, FRESNEL_GAIN, 0x8fb4e8)); // rim to define the edge
    if (SKY) scene.add(buildSkybox(makeSkyTexture()));
    // (The sun is a direction, not a three.js Light: every material is a
    // ShaderMaterial lit in-shader via the shared uSunDir uniform, set per frame
    // by updateSun(). Scene lights would have no effect on these materials.)

    // Map-style view: camera directly above the focus point (southern Europe /
    // central Mediterranean), north pole as "up" -> equator runs left/right
    // (west-east), poles up/down.
    const fLat = THREE.MathUtils.degToRad(FOCUS_LAT), fLon = THREE.MathUtils.degToRad(FOCUS_LON);
    const focus = new THREE.Vector3(Math.cos(fLat) * Math.cos(fLon),
                                   Math.cos(fLat) * Math.sin(fLon),
                                   Math.sin(fLat));
    camera.position.copy(focus).multiplyScalar(CAM_DIST);
    camera.up.set(0, 0, 1);   // +z = north pole
    camera.lookAt(0, 0, 0);

    computeGrid();
    window.addEventListener('resize', onResize);

    await loadFrame(0);      // draw frame 0 so __ready means a frame is on screen
    window.__ready = true;

    if (location.search.includes('live')) startLive(meta.render && meta.render.fps);
  } catch (e) {
    window.__initError = (e && e.message) ? e.message : String(e);
    console.error('[viewer] init failed:', e);
    throw e;
  }
}

// capture contract: fetch + draw frame i, wait for the GPU to present, resolve true
window.__settle = async (i) => { await loadFrame(i); await nextFrame(); return true; };
// debug: azimuthal (horizontal) std-dev of the top/bottom pole rows, to check the star
window.__poleVar = () => {
  const sd = (r) => { const row = shellField.slice(r*GW, r*GW+GW); const m = row.reduce((a,b)=>a+b,0)/GW;
    return Math.sqrt(row.reduce((a,b)=>a+(b-m)*(b-m),0)/GW); };
  let topMax=0, topRow=-1, botMax=0, botRow=-1;
  for (let r=GH-50; r<GH; r++){ const v=sd(r); if(v>topMax){topMax=v; topRow=r;} }
  for (let r=0; r<50; r++){ const v=sd(r); if(v>botMax){botMax=v; botRow=r;} }
  return { GH, POLE_BAND, topPoleMaxStd:+topMax.toFixed(3), topRow, botPoleMaxStd:+botMax.toFixed(3), botRow };
};
// debug: azimuthal std of each pole row, to see how far the "star" extends
window.__starExtent = () => {
  const out = [];
  for (let k = 0; k <= 40; k += 2) {
    const r = GH - 1 - k;
    const row = shellField.slice(r * GW, r * GW + GW);
    const m = row.reduce((a, b) => a + b, 0) / GW;
    const sd = Math.sqrt(row.reduce((a, b) => a + (b - m) * (b - m), 0) / GW);
    out.push({ k, lat: +(90 - k * (180 / GH)).toFixed(2), sd: +sd.toFixed(2) });
  }
  return out;
};
init();
