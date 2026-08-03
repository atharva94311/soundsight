import * as THREE from 'three';

export const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
export const lerp = (a, b, t) => a + (b - a) * t;
export const smoothstep = (t) => { t = clamp(t, 0, 1); return t * t * (3 - 2 * t); };
export const easeOut = (t) => 1 - Math.pow(1 - clamp(t, 0, 1), 3);

/** Deterministic hash noise — stable jitter without Math.random flicker. */
export function noise(x) {
  const s = Math.sin(x * 127.1) * 43758.5453;
  return s - Math.floor(s);
}

// ---------------------------------------------------------------------------
//  Canvas-backed textures (OLED, phone screen, laptop screen)
// ---------------------------------------------------------------------------
export function makeScreen(w, h) {
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 4;
  texture.minFilter = THREE.LinearFilter;
  return {
    canvas, ctx, texture, w, h,
    flush() { texture.needsUpdate = true; },
  };
}

export function roundRect(ctx, x, y, w, h, r) {
  const rr = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr);
  ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
}

// ---------------------------------------------------------------------------
//  Materials
// ---------------------------------------------------------------------------
export function matte(color, opts = {}) {
  return new THREE.MeshStandardMaterial({
    color, roughness: opts.roughness ?? 0.85, metalness: opts.metalness ?? 0.05, ...opts,
  });
}

export function glowMat(color, intensity = 1) {
  return new THREE.MeshStandardMaterial({
    color: 0x000000, emissive: color, emissiveIntensity: intensity,
    roughness: 0.4, metalness: 0,
  });
}

/** Additive halo sprite — the visible "bloom" around every LED. */
const haloTex = (() => {
  const c = document.createElement('canvas');
  c.width = c.height = 128;
  const g = c.getContext('2d');
  const grad = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  grad.addColorStop(0.0, 'rgba(255,255,255,1)');
  grad.addColorStop(0.25, 'rgba(255,255,255,0.55)');
  grad.addColorStop(0.6, 'rgba(255,255,255,0.12)');
  grad.addColorStop(1.0, 'rgba(255,255,255,0)');
  g.fillStyle = grad;
  g.fillRect(0, 0, 128, 128);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
})();

export function makeHalo(color, size = 0.5) {
  const s = new THREE.Sprite(new THREE.SpriteMaterial({
    map: haloTex, color, blending: THREE.AdditiveBlending,
    transparent: true, depthWrite: false, opacity: 0,
  }));
  s.scale.setScalar(size);
  return s;
}

// ---------------------------------------------------------------------------
//  Screen-space HTML tags anchored to world positions
// ---------------------------------------------------------------------------
export class TagLayer {
  constructor(container, camera) {
    this.container = container;
    this.camera = camera;
    this.tags = [];
    this._v = new THREE.Vector3();
  }

  add(html, className = '') {
    const el = document.createElement('div');
    el.className = `tag ${className}`;
    el.innerHTML = html;
    this.container.appendChild(el);
    const tag = {
      el,
      anchor: new THREE.Vector3(),
      offset: [0, 0],
      visible: true,
      scaleWithDepth: true,
      setHTML: (h) => { el.innerHTML = h; },
      setVisible: (v) => { tag.visible = v; if (!v) el.style.display = 'none'; },
      dispose: () => { el.remove(); this.tags = this.tags.filter((t) => t !== tag); },
    };
    this.tags.push(tag);
    return tag;
  }

  /** `safe` is the rectangle not covered by HUD panels: {l, r, t, b}. */
  update(width, height, safe = null) {
    const cam = this.camera;
    for (const t of this.tags) {
      if (!t.visible) { t.el.style.display = 'none'; continue; }
      this._v.copy(t.anchor).project(cam);
      const behind = this._v.z > 1;
      if (behind) { t.el.style.display = 'none'; continue; }
      const x = (this._v.x * 0.5 + 0.5) * width + t.offset[0];
      const y = (-this._v.y * 0.5 + 0.5) * height + t.offset[1];
      // Hide once any part of the label would slide under a HUD panel, not just
      // its anchor point — otherwise wide labels get visibly clipped.
      if (safe) {
        const hw = t.el.offsetWidth / 2;
        if (x - hw < safe.l || x + hw > safe.r || y < safe.t || y > safe.b) {
          t.el.style.display = 'none';
          continue;
        }
      }
      t.el.style.display = 'block';
      t.el.style.transform = `translate(-50%,-50%) translate(${x.toFixed(1)}px,${y.toFixed(1)}px)`;
    }
  }
}

// ---------------------------------------------------------------------------
//  Small geometry helpers
// ---------------------------------------------------------------------------
export function box(w, h, d, mat, x = 0, y = 0, z = 0) {
  const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
  m.position.set(x, y, z);
  m.castShadow = true;
  m.receiveShadow = true;
  return m;
}

export function cyl(rt, rb, h, mat, seg = 20) {
  const m = new THREE.Mesh(new THREE.CylinderGeometry(rt, rb, h, seg), mat);
  m.castShadow = true;
  return m;
}

export const V = (x, y, z) => new THREE.Vector3(x, y, z);
