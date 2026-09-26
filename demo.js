// Lumen in the browser: drag the projected corners onto the object, just like the app.
// Three layers: the room (2D), the projection (WebGL, same homography maths as the app's
// Metal renderer), and the guides/handles (2D).
(() => {
  const stage = document.getElementById('stage');
  if (!stage) return;
  const roomCv = stage.querySelector('.room'), glCv = stage.querySelector('.light'), uiCv = stage.querySelector('.guides');
  const room = roomCv.getContext('2d'), ui = uiCv.getContext('2d');
  const gl = glCv.getContext('webgl2', { premultipliedAlpha: true, antialias: true });
  const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const hint = document.getElementById('demo-hint'), fitBadge = document.getElementById('demo-fit');

  // Virtual stage space: 1000 × 640. Everything is authored here and scaled to the canvas.
  const VW = 1000, VH = 640;
  const P = (x, y) => ({ x, y });
  const grow = (k, cx, cy) => (x, y) => P(Math.round(cx + (x - cx) * k), Math.round(cy + (y - cy) * k));
  const G = grow(1.32, 500, 335), GP = grow(1.18, 510, 334);
  const SCENES = {
    box: {
      name: 'box',
      target: (() => {
        const L = G(352, 230), T = G(500, 148), R = G(648, 230), C = G(500, 312);
        const BL = G(352, 440), B = G(500, 522), BR = G(648, 440);
        return [[L, T, R, C], [L, C, B, BL], [C, R, BR, B]];
      })(),
      colors: [['#E4FF3A', '#101106'], ['#1AD9FF', '#3A1E8C'], ['#FF6B28', '#FF337F']]
    },
    poster: {
      name: 'poster',
      target: [[GP(330, 120), GP(690, 160), GP(672, 520), GP(346, 548)]],
      colors: [['#E4FF3A', '#101106']]
    }
  };
  // A misaligned projector: the whole image is shifted and a bit small, with slight keystone.
  // Jitter is keyed to the point, so corners that share a point on the object stay together.
  const key = p => `${p.x},${p.y}`;
  function misplace(p) {
    const cx = 500, cy = 330, k = 0.8;
    const j = Math.sin(p.x * 12.9898 + p.y * 78.233) * 43758.5453;
    const jx = (j - Math.floor(j) - .5) * 26, jy = (Math.sin(j) * .5) * 22;
    return P(cx + (p.x - cx) * k - 34 + jx, cy + (p.y - cy) * k - 26 + jy);
  }

  let scene = SCENES.box;
  let quads = [];            // current projected corners, per surface
  let content = 'grid', effect = 'none';
  let userPickedContent = false;
  let drag = null, hover = null, fitted = false, fitTime = -10, autoAnim = null;
  let scale = 1, dpr = 1, ox = 0, oy = 0, hs = 1;
  // On narrow screens the stage is taller and the view crops in around the object.
  let view = { x: 0, y: 0, w: VW, h: VH };
  const start = performance.now();

  function resetQuads() {
    quads = scene.target.map(q => q.map(misplace));
    fitted = false; fitBadge.classList.remove('on');
    if (!userPickedContent) setContent('grid', false);
    updateHint();
  }

  // ---------- geometry ----------
  function solve(a, b) {
    const n = b.length;
    for (let k = 0; k < n; k++) {
      let piv = k; for (let i = k + 1; i < n; i++) if (Math.abs(a[i][k]) > Math.abs(a[piv][k])) piv = i;
      if (Math.abs(a[piv][k]) < 1e-12) return null;
      [a[k], a[piv]] = [a[piv], a[k]]; [b[k], b[piv]] = [b[piv], b[k]];
      for (let i = k + 1; i < n; i++) { const f = a[i][k] / a[k][k]; for (let j = k; j < n; j++) a[i][j] -= f * a[k][j]; b[i] -= f * b[k]; }
    }
    const x = new Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) { let s = b[i]; for (let j = i + 1; j < n; j++) s -= a[i][j] * x[j]; x[i] = s / a[i][i]; }
    return x;
  }
  // Unit square → quad (TL, TR, BR, BL), returned row-major.
  function homography(q) {
    const src = [[0, 0], [1, 0], [1, 1], [0, 1]], A = [], B = [];
    for (let i = 0; i < 4; i++) {
      const [x, y] = src[i], u = q[i].x, v = q[i].y;
      A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]); B.push(u);
      A.push([0, 0, 0, x, y, 1, -v * x, -v * y]); B.push(v);
    }
    const h = solve(A, B);
    return h ? [...h, 1] : null;
  }
  function invert3(m) {
    const [a, b, c, d, e, f, g, h, i] = m;
    const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
    const det = a * A + b * B + c * C; if (Math.abs(det) < 1e-12) return null;
    return [A / det, -(b * i - c * h) / det, (b * f - c * e) / det,
            B / det, (a * i - c * g) / det, -(a * f - c * d) / det,
            C / det, -(a * h - b * g) / det, (a * e - b * d) / det];
  }
  const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

  // ---------- WebGL ----------
  let prog, uni = {}, tex = null, texReady = false;
  if (gl) {
    const vs = `#version 300 es
      in vec2 p; out vec2 vPos; uniform vec2 uRes;
      void main() { gl_Position = vec4(p, 0, 1); vPos = vec2((p.x * .5 + .5) * uRes.x, (.5 - p.y * .5) * uRes.y); }`;
    const fs = `#version 300 es
      precision highp float;
      in vec2 vPos; out vec4 o;
      uniform mat3 uInv; uniform float uT, uAspect, uFit;
      uniform int uPat, uFx; uniform vec3 uA, uB; uniform sampler2D uTex; uniform float uTexOK;
      float aa(float d, float w) { float fw = fwidth(d); return 1. - smoothstep(w - fw, w + fw, d); }
      float hash(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
      float vnoise(vec2 p) { vec2 i = floor(p), f = fract(p); vec2 u = f * f * (3. - 2. * f);
        return mix(mix(hash(i), hash(i + vec2(1, 0)), u.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), u.x), u.y); }
      float fbm(vec2 p) { float v = 0., a = .5; for (int i = 0; i < 5; i++) { v += a * vnoise(p); p = p * 2.03 + 17.; a *= .5; } return v; }
      vec3 hue(vec3 c, float a) {
        mat3 toY = mat3(.299,.596,.211, .587,-.274,-.523, .114,-.322,.312);
        mat3 toR = mat3(1.,1.,1., .956,-.272,-1.106, .621,-.647,1.703);
        vec3 y = toY * c; float s = sin(a), co = cos(a);
        y.yz = vec2(y.y * co - y.z * s, y.y * s + y.z * co); return clamp(toR * y, 0., 1.);
      }
      vec3 pattern(vec2 uv) {
        vec2 p = vec2(uv.x * uAspect, uv.y); float t = uT;
        if (uPat == 0) {
          vec2 g = abs(fract(p * 8. + .5) - .5) / 8.;
          float l = max(aa(g.x, .0016), aa(g.y, .0016)) * .55;
          vec2 e = min(uv, 1. - uv) * vec2(uAspect, 1.);
          float b = aa(min(e.x, e.y), .01);
          float r = aa(abs(length(p - vec2(uAspect, 1.) * .5) - .3), .004);
          float d = max(aa(abs(uv.x - uv.y), .003), aa(abs(uv.x + uv.y - 1.), .003)) * .6;
          return mix(uB, uA, max(max(l, b), max(r, d)));
        }
        if (uPat == 1) {
          float v = sin(p.x * 6. + t) + sin(p.y * 5. - t * 1.3) + sin((p.x + p.y) * 4. + t * .7) + sin(length(p - .5) * 8. - t);
          return mix(uA, uB, .5 + .5 * sin(v * 1.2));
        }
        if (uPat == 2) { float s = fract(length(p - vec2(uAspect, 1.) * .5) * 5. - t * .3); return mix(uA, uB, smoothstep(.45, .55, s)); }
        if (uPat == 3) { float s = fract((p.x + p.y) * 4. - t * .25); return mix(uA, uB, smoothstep(.48, .52, s)); }
        if (uPat == 5) { // fire
          vec2 q = vec2(p.x * 3., uv.y * 3. + t * 1.4);
          float n = fbm(q + fbm(q * 1.5 - vec2(0, t)));
          float heat = clamp(n * 1.6 - uv.y * .3 + (1. - uv.y) * .9 - .55, 0., 1.);
          vec3 col = mix(vec3(.25, .02, .02), vec3(1., .45, .05), clamp(heat * 1.6, 0., 1.));
          col = mix(col, vec3(1., .95, .75), clamp(heat * 2. - 1.2, 0., 1.));
          return col * clamp(heat * 3., 0., 1.);
        }
        if (uPat == 6) { // lava
          float f = 0.;
          for (int i = 0; i < 6; i++) { float fi = float(i);
            vec2 c = vec2(.5 * uAspect + sin(t * .3 + fi * 1.7) * .35 * uAspect, .5 + cos(t * .23 + fi * 2.3) * .38);
            vec2 d = p - c; f += .018 / max(dot(d, d), 1e-4); }
          return mix(vec3(.15, .02, .2), vec3(1., .42, .16), clamp(smoothstep(.9, 1.2, f) + smoothstep(.3, 1., f) * .4, 0., 1.));
        }
        if (uPat == 7) { // stars
          vec3 col = vec3(.01, .02, .06);
          for (int l = 0; l < 3; l++) { float fl = float(l);
            vec2 q = p * (8. + fl * 7.) + vec2(t * (.05 + fl * .04), fl * 13.);
            vec2 cell = floor(q), f = fract(q) - .5;
            float h = hash(cell + fl * 31.);
            vec2 off = vec2(hash(cell + 7.), hash(cell + 3.)) - .5;
            float star = smoothstep(.08 - fl * .02, 0., length(f - off * .6)) * step(.72, h);
            col += vec3(1.) * star * (.6 + .4 * sin(t * (2. + h * 4.) + h * 40.)); }
          return col;
        }
        if (uPat == 4) {
          if (uTexOK < .5) return vec3(.06);
          float qa = uAspect, ta = .75; vec2 sc = qa < ta ? vec2(qa / ta, 1.) : vec2(1., ta / qa);
          return texture(uTex, uv * sc + (1. - sc) * .5).rgb;
        }
        return uA;
      }
      void main() {
        vec3 h = uInv * vec3(vPos, 1.);
        if (h.z <= 0.) discard;
        vec2 uv = h.xy / h.z;
        vec2 fw = clamp(fwidth(uv), vec2(1e-5), vec2(.02));
        vec2 ep = min(uv, 1. - uv) / fw;
        float m = clamp(min(ep.x, ep.y) + .5, 0., 1.);
        if (m <= 0.) discard;
        float t = uT * .9; vec2 s = uv;
        if (uFx == 3) { vec2 c = (uv - .5) * vec2(uAspect, 1.); float seg = 6.2831 / 8.; float a = atan(c.y, c.x) + t * .5;
          a = abs(mod(a + 100. * seg, seg) - seg * .5); s = vec2(cos(a), sin(a)) * length(c) / vec2(uAspect, 1.) + .5; s = 1. - abs(fract(s * .5) * 2. - 1.); }
        if (uFx == 4) s = clamp(uv + vec2(sin(uv.y * 18. + t * 3.), cos(uv.x * 14. + t * 2.4)) * .018, 0., 1.);
        if (uFx == 5) s = fract(uv + vec2(t * .12, 0.));
        vec3 col = pattern(s);
        if (uFx == 1) col *= mix(1., .5 + .5 * sin(t * 4.), .8);
        if (uFx == 2) col = hue(col, t * 1.5);
        col += uFit * vec3(.35);            // flash on a perfect fit
        o = vec4(col * m, m);
      }`;
    const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) console.error(gl.getShaderInfoLog(s)); return s; };
    prog = gl.createProgram();
    gl.attachShader(prog, sh(gl.VERTEX_SHADER, vs)); gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(prog); gl.useProgram(prog);
    const buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(prog, 'p'); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    ['uRes', 'uInv', 'uT', 'uAspect', 'uPat', 'uFx', 'uA', 'uB', 'uTex', 'uTexOK', 'uFit'].forEach(n => uni[n] = gl.getUniformLocation(prog, n));
    gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    tex = gl.createTexture();
    const img = new Image();
    img.onload = () => {
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      texReady = true;
    };
    img.src = 'img/art-sunset.jpg';
  } else {
    stage.classList.add('no-gl');
  }
  const hex = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16) / 255);
  const PAT = { grid: 0, plasma: 1, rings: 2, stripes: 3, photo: 4, fire: 5, lava: 6, stars: 7 };
  const FX = { none: 0, pulse: 1, hue: 2, kaleido: 3, ripple: 4, drift: 5 };

  // ---------- sizing ----------
  function resize() {
    const r = stage.getBoundingClientRect();
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    [roomCv, glCv, uiCv].forEach(c => { c.width = Math.round(r.width * dpr); c.height = Math.round(r.height * dpr); });
    view = r.width < 560 ? { x: 170, y: 50, w: 660, h: 560 } : { x: 0, y: 0, w: VW, h: VH };
    scale = Math.min(r.width / view.w, r.height / view.h);
    ox = (r.width - view.w * scale) / 2 - view.x * scale; oy = (r.height - view.h * scale) / 2 - view.y * scale;
    hs = Math.min(1, Math.max(.75, r.width / 700));
    drawRoom();
  }
  const toPx = p => P((ox + p.x * scale) * dpr, (oy + p.y * scale) * dpr);
  const fromEvent = e => { const r = stage.getBoundingClientRect(); return P((e.clientX - r.left - ox) / scale, (e.clientY - r.top - oy) / scale); };

  // ---------- the room ----------
  function drawRoom() {
    const w = roomCv.width, h = roomCv.height;
    room.setTransform(1, 0, 0, 1, 0, 0);
    const g = room.createRadialGradient(w * .5, h * .42, 0, w * .5, h * .42, Math.max(w, h) * .75);
    g.addColorStop(0, '#1C1C22'); g.addColorStop(1, '#09090B');
    room.fillStyle = g; room.fillRect(0, 0, w, h);
    const floorY = toPx(P(0, scene === SCENES.box ? 500 : 620)).y;
    const fg = room.createLinearGradient(0, floorY, 0, h); fg.addColorStop(0, '#111114'); fg.addColorStop(1, '#070709');
    room.fillStyle = fg; room.fillRect(0, floorY, w, h - floorY);
    room.strokeStyle = 'rgba(255,255,255,.05)'; room.lineWidth = dpr; room.beginPath(); room.moveTo(0, floorY); room.lineTo(w, floorY); room.stroke();
    const shades = ['#26262C', '#1B1B20', '#141418'];
    scene.target.forEach((q, i) => {
      const pts = q.map(toPx);
      if (scene === SCENES.box && i === 0) {   // soft contact shadow under the box
        const b = toPx(scene.target[1][2]), sw = 250 * scale * dpr;
        const sg = room.createRadialGradient(b.x, b.y, 0, b.x, b.y, sw);
        sg.addColorStop(0, 'rgba(0,0,0,.55)'); sg.addColorStop(1, 'rgba(0,0,0,0)');
        room.fillStyle = sg; room.beginPath(); room.ellipse(b.x, b.y, sw, sw * .2, 0, 0, 7); room.fill();
      }
      room.fillStyle = scene === SCENES.box ? shades[i] : '#1E1E24';
      room.beginPath(); pts.forEach((p, k) => k ? room.lineTo(p.x, p.y) : room.moveTo(p.x, p.y)); room.closePath(); room.fill();
      room.strokeStyle = 'rgba(255,255,255,.14)'; room.lineWidth = 1.5 * dpr; room.stroke();
    });
    if (scene === SCENES.poster) {   // a frame, so it reads as a canvas on the wall
      const pts = scene.target[0].map(toPx);
      room.strokeStyle = '#2C2C33'; room.lineWidth = 10 * scale * dpr; room.lineJoin = 'round';
      room.beginPath(); pts.forEach((p, k) => k ? room.lineTo(p.x, p.y) : room.moveTo(p.x, p.y)); room.closePath(); room.stroke();
    }
  }

  // ---------- per frame ----------
  function frame(now) {
    const t = (now - start) / 1000;
    if (autoAnim) stepAuto(now);
    drawLight(t);
    drawGuides(t);
    if (visible) requestAnimationFrame(frame);
  }

  function drawLight(t) {
    if (!gl) return;
    gl.viewport(0, 0, glCv.width, glCv.height);
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
    gl.uniform2f(uni.uRes, glCv.width, glCv.height);
    gl.uniform1f(uni.uT, reduceMotion ? 1.5 : t);
    gl.uniform1i(uni.uFx, FX[effect]);
    gl.uniform1i(uni.uPat, PAT[content]);
    gl.uniform1f(uni.uTexOK, texReady ? 1 : 0);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, tex); gl.uniform1i(uni.uTex, 0);
    const flash = Math.max(0, 1 - (t - fitTime) / .5) * .8;
    gl.uniform1f(uni.uFit, flash);
    quads.forEach((q, i) => {
      const px = q.map(toPx);
      const H = homography(px); if (!H) return;
      const inv = invert3(H); if (!inv) return;
      gl.uniformMatrix3fv(uni.uInv, false, [inv[0], inv[3], inv[6], inv[1], inv[4], inv[7], inv[2], inv[5], inv[8]]);
      const w = (dist(q[0], q[1]) + dist(q[3], q[2])) / 2, h = (dist(q[0], q[3]) + dist(q[1], q[2])) / 2;
      gl.uniform1f(uni.uAspect, h > 1 ? w / h : 1);
      const [a, b] = scene.colors[i];
      gl.uniform3fv(uni.uA, hex(a)); gl.uniform3fv(uni.uB, hex(b));
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    });
  }

  function drawGuides(t) {
    ui.setTransform(1, 0, 0, 1, 0, 0); ui.clearRect(0, 0, uiCv.width, uiCv.height);
    if (fitted && !drag) return;   // clean stage once it fits
    const beam = '#E4FF3A';
    quads.forEach((q, qi) => {
      const pts = q.map(toPx);
      ui.beginPath(); pts.forEach((p, k) => k ? ui.lineTo(p.x, p.y) : ui.moveTo(p.x, p.y)); ui.closePath();
      ui.strokeStyle = 'rgba(0,0,0,.55)'; ui.lineWidth = 4 * dpr; ui.stroke();
      ui.strokeStyle = beam; ui.lineWidth = 1.6 * dpr; ui.stroke();
      q.forEach((c, i) => {
        const p = pts[i], target = scene.target[qi][i];
        const placed = dist(c, target) < .5;
        const active = drag && drag.q === qi && drag.i === i;
        const isHover = hover && hover.q === qi && hover.i === i;
        const r = (active ? 13 : isHover ? 12 : 10) * dpr * hs;
        if (!placed && !active && !reduceMotion) {   // breathing ring invites a drag
          const k = (t * .9 + (qi * 4 + i) * .13) % 1;
          ui.strokeStyle = `rgba(228,255,58,${.5 * (1 - k)})`; ui.lineWidth = 2 * dpr;
          ui.beginPath(); ui.arc(p.x, p.y, r + k * 18 * dpr, 0, 7); ui.stroke();
        }
        ui.beginPath(); ui.arc(p.x, p.y, r, 0, 7);
        ui.fillStyle = active || placed ? beam : 'rgba(10,10,12,.85)'; ui.fill();
        ui.strokeStyle = beam; ui.lineWidth = 2.4 * dpr; ui.stroke();
        ui.beginPath(); ui.arc(p.x, p.y, 3 * dpr, 0, 7); ui.fillStyle = active || placed ? '#0A0A0C' : beam; ui.fill();
      });
    });
    if (drag) {   // readout pill, like the app
      const c = quads[drag.q][drag.i], p = toPx(c);
      const txt = `${['TL', 'TR', 'BR', 'BL'][drag.i]}  ${Math.round(c.x * 1.92)}  ${Math.round(c.y * 1.6875)}`;
      ui.font = `600 ${12 * dpr}px ui-monospace, SF Mono, Menlo, monospace`;
      const w = ui.measureText(txt).width + 22 * dpr, h = 26 * dpr;
      const x = Math.min(Math.max(p.x - w / 2, 6 * dpr), uiCv.width - w - 6 * dpr), y = Math.max(p.y - 52 * dpr, 6 * dpr);
      ui.fillStyle = beam; ui.beginPath(); ui.roundRect(x, y, w, h, h / 2); ui.fill();
      ui.fillStyle = '#0A0A0C'; ui.textBaseline = 'middle'; ui.fillText(txt, x + 11 * dpr, y + h / 2 + dpr);
    }
  }

  // ---------- interaction ----------
  function nearest(pt, radius) {
    let best = null;
    quads.forEach((q, qi) => q.forEach((c, i) => {
      const d = dist(c, pt);
      if (d < radius && (!best || d < best.d)) best = { q: qi, i, d };
    }));
    return best;
  }
  const hitRadius = e => (e.pointerType === 'touch' ? 44 : 28) / scale;
  const snapRadius = () => 22 / Math.max(scale, .4);

  uiCv.addEventListener('pointerdown', e => {
    const pt = fromEvent(e), hit = nearest(pt, hitRadius(e));
    if (!hit) return;
    autoAnim = null;
    const k0 = key(scene.target[hit.q][hit.i]);
    const linked = [];
    scene.target.forEach((q, qi) => q.forEach((p, i) => { if (key(p) === k0) linked.push([qi, i]); }));
    drag = { q: hit.q, i: hit.i, linked, dx: quads[hit.q][hit.i].x - pt.x, dy: quads[hit.q][hit.i].y - pt.y };
    uiCv.setPointerCapture(e.pointerId);
    stage.classList.add('dragging');
    e.preventDefault();
  });
  uiCv.addEventListener('pointermove', e => {
    const pt = fromEvent(e);
    if (!drag) {
      hover = nearest(pt, hitRadius(e));
      uiCv.style.cursor = hover ? 'grab' : 'default';
      return;
    }
    let p = P(pt.x + drag.dx, pt.y + drag.dy);
    const target = scene.target[drag.q][drag.i];
    if (dist(p, target) < snapRadius()) p = P(target.x, target.y);   // snap onto the object's corner
    drag.linked.forEach(([qi, i]) => { quads[qi][i] = P(p.x, p.y); });
    uiCv.style.cursor = 'grabbing';
  });
  const end = () => {
    if (!drag) return;
    const c = quads[drag.q][drag.i], target = scene.target[drag.q][drag.i];
    if (dist(c, target) < snapRadius()) { drag.linked.forEach(([qi, i]) => { quads[qi][i] = P(target.x, target.y); }); if (navigator.vibrate) navigator.vibrate(8); }
    drag = null; stage.classList.remove('dragging');
    checkFit();
  };
  uiCv.addEventListener('pointerup', end);
  uiCv.addEventListener('pointercancel', end);

  // Counts distinct points on the object (a box has 7), not per-surface corners.
  function points() {
    const m = new Map();
    quads.forEach((q, qi) => q.forEach((c, i) => { const k = key(scene.target[qi][i]); m.set(k, (m.get(k) ?? true) && dist(c, scene.target[qi][i]) < .5); }));
    return m;
  }
  function placedCount() { let n = 0; points().forEach(v => { if (v) n++; }); return n; }
  function updateHint() {
    const total = points().size, n = placedCount();
    hint.innerHTML = n === 0
      ? `Drag the glowing corners onto the ${scene === SCENES.box ? 'box' : 'frame'}.`
      : `<strong>${n} / ${total}</strong> corners placed${n < total ? '. Keep going.' : ''}`;
  }
  function checkFit() {
    updateHint();
    if (!fitted && placedCount() === points().size) {
      fitted = true; fitTime = (performance.now() - start) / 1000;
      fitBadge.classList.add('on');
      if (!userPickedContent) setContent('plasma', false);
      stage.classList.add('fit'); setTimeout(() => stage.classList.remove('fit'), 700);
    }
  }

  // "Auto-align": animate every corner home.
  function autoAlign() {
    autoAnim = { from: quads.map(q => q.map(c => P(c.x, c.y))), t0: performance.now() };
  }
  function stepAuto(now) {
    const k = Math.min(1, (now - autoAnim.t0) / 900), e = k < .5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2;
    quads = autoAnim.from.map((q, qi) => q.map((c, i) => { const t = scene.target[qi][i]; return P(c.x + (t.x - c.x) * e, c.y + (t.y - c.y) * e); }));
    if (k >= 1) { autoAnim = null; quads = scene.target.map(q => q.map(p => P(p.x, p.y))); checkFit(); }
  }

  // ---------- controls ----------
  function setActive(group, value) {
    document.querySelectorAll(`[data-${group}]`).forEach(b => b.classList.toggle('on', b.dataset[group] === value));
  }
  function setContent(v, byUser = true) { content = v; if (byUser) userPickedContent = true; setActive('content', v); }
  document.querySelectorAll('[data-content]').forEach(b => b.addEventListener('click', () => setContent(b.dataset.content)));
  document.querySelectorAll('[data-effect]').forEach(b => b.addEventListener('click', () => { effect = b.dataset.effect; setActive('effect', effect); }));
  document.querySelectorAll('[data-scene]').forEach(b => b.addEventListener('click', () => {
    scene = SCENES[b.dataset.scene]; setActive('scene', b.dataset.scene); resetQuads(); drawRoom();
  }));
  document.getElementById('demo-reset').addEventListener('click', resetQuads);
  document.getElementById('demo-auto').addEventListener('click', autoAlign);

  // Only animate while on screen.
  let visible = false;
  new IntersectionObserver(([e]) => {
    const was = visible; visible = e.isIntersecting;
    if (visible && !was) requestAnimationFrame(frame);
  }).observe(stage);
  new ResizeObserver(resize).observe(stage);

  resetQuads(); resize();
})();
