import * as THREE from 'three';
import { clamp, easeOut } from './util.js';

/**
 * Compact orbit controller with damping and scripted camera flights.
 * Left drag  = orbit   |   Right / Shift drag = pan   |   Wheel = dolly
 */
export class Orbit {
  constructor(camera, dom) {
    this.camera = camera;
    this.dom = dom;
    this.target = new THREE.Vector3(0, 0.6, 0);

    this.theta = -0.85;      // azimuth
    this.phi = 0.95;         // polar from +Y
    this.radius = 17;

    this._t = { theta: this.theta, phi: this.phi, radius: this.radius };
    this._targetGoal = this.target.clone();

    this.minRadius = 1.2;
    this.maxRadius = 40;
    this.minPhi = 0.08;
    this.maxPhi = Math.PI / 2 - 0.02;

    this.flight = null;
    this.enabled = true;

    this._drag = null;
    this._bind();
    this.update(0);
  }

  _bind() {
    const dom = this.dom;
    dom.addEventListener('contextmenu', (e) => e.preventDefault());

    dom.addEventListener('pointerdown', (e) => {
      if (!this.enabled) return;
      dom.setPointerCapture(e.pointerId);
      this._drag = {
        id: e.pointerId, x: e.clientX, y: e.clientY,
        pan: e.button === 2 || e.shiftKey,
        moved: 0,
      };
      this.flight = null;
    });

    dom.addEventListener('pointermove', (e) => {
      const d = this._drag;
      if (!d || d.id !== e.pointerId) return;
      const dx = e.clientX - d.x;
      const dy = e.clientY - d.y;
      d.x = e.clientX; d.y = e.clientY;
      d.moved += Math.abs(dx) + Math.abs(dy);

      if (d.pan) {
        const k = this._t.radius * 0.0016;
        const right = new THREE.Vector3().setFromMatrixColumn(this.camera.matrix, 0);
        const up = new THREE.Vector3(0, 1, 0).cross(right).normalize().negate();
        const move = right.multiplyScalar(-dx * k).add(up.multiplyScalar(-dy * k));
        move.y = 0;
        this._targetGoal.add(move);
        this._clampTarget();
      } else {
        this._t.theta -= dx * 0.005;
        this._t.phi = clamp(this._t.phi - dy * 0.005, this.minPhi, this.maxPhi);
      }
    });

    const end = (e) => { if (this._drag && this._drag.id === e.pointerId) this._drag = null; };
    dom.addEventListener('pointerup', end);
    dom.addEventListener('pointercancel', end);

    dom.addEventListener('wheel', (e) => {
      if (!this.enabled) return;
      e.preventDefault();
      this.flight = null;
      const f = Math.exp(e.deltaY * 0.0012);
      this._t.radius = clamp(this._t.radius * f, this.minRadius, this.maxRadius);
    }, { passive: false });
  }

  _clampTarget() {
    this._targetGoal.x = clamp(this._targetGoal.x, -14, 14);
    this._targetGoal.z = clamp(this._targetGoal.z, -12, 12);
    this._targetGoal.y = clamp(this._targetGoal.y, -1, 6);
  }

  /** Fly to a framing: look at `target` from spherical (theta, phi, radius). */
  flyTo({ target, theta, phi, radius, duration = 1.25 }) {
    this.flight = {
      t: 0, duration,
      from: {
        target: this.target.clone(),
        theta: this._t.theta, phi: this._t.phi, radius: this._t.radius,
      },
      to: {
        target: target ? target.clone() : this._targetGoal.clone(),
        theta: theta ?? this._t.theta,
        phi: phi ?? this._t.phi,
        radius: radius ?? this._t.radius,
      },
    };
    // Take the short way around the circle.
    const f = this.flight;
    let d = f.to.theta - f.from.theta;
    while (d > Math.PI) d -= Math.PI * 2;
    while (d < -Math.PI) d += Math.PI * 2;
    f.to.theta = f.from.theta + d;
  }

  update(dt) {
    if (this.flight) {
      const f = this.flight;
      f.t += dt;
      const k = easeOut(f.t / f.duration);
      this._t.theta = THREE.MathUtils.lerp(f.from.theta, f.to.theta, k);
      this._t.phi = THREE.MathUtils.lerp(f.from.phi, f.to.phi, k);
      this._t.radius = THREE.MathUtils.lerp(f.from.radius, f.to.radius, k);
      this._targetGoal.lerpVectors(f.from.target, f.to.target, k);
      if (f.t >= f.duration) this.flight = null;
    }

    const s = 1 - Math.pow(0.001, Math.max(dt, 1 / 240));
    this.theta = THREE.MathUtils.lerp(this.theta, this._t.theta, s);
    this.phi = THREE.MathUtils.lerp(this.phi, this._t.phi, s);
    this.radius = THREE.MathUtils.lerp(this.radius, this._t.radius, s);
    this.target.lerp(this._targetGoal, s);

    const sp = Math.sin(this.phi);
    this.camera.position.set(
      this.target.x + this.radius * sp * Math.sin(this.theta),
      this.target.y + this.radius * Math.cos(this.phi),
      this.target.z + this.radius * sp * Math.cos(this.theta),
    );
    this.camera.lookAt(this.target);
  }

  get dragDistance() { return this._drag ? this._drag.moved : 0; }
}
