import * as THREE from 'three';
import { APT, ROOMS, WALLS, DEVICES } from './config.js';
import { matte, glowMat, makeHalo, makeScreen, box, cyl, V } from './util.js';

const ROOM_BY_ID = Object.fromEntries(ROOMS.map((r) => [r.id, r]));

export function buildWorld(scene) {
  const refs = {
    devices: new Map(),
    rooms: {},
    floods: {},
    roomLights: [],
    screens: {},
    person: null,
    visitor: null,
    doorbellBtn: null,
    ceilingStrip: null,
    group: new THREE.Group(),
  };
  scene.add(refs.group);
  const root = refs.group;

  // -----------------------------------------------------------------------
  //  Ground + room floors
  // -----------------------------------------------------------------------
  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(60, 60),
    new THREE.MeshStandardMaterial({ color: 0x080a0e, roughness: 1 }),
  );
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = -0.06;
  ground.receiveShadow = true;
  root.add(ground);

  const slab = box(APT.maxX - APT.minX + 0.5, 0.12, APT.maxZ - APT.minZ + 0.5,
    matte(0x11151d, { roughness: 0.95 }), 0, -0.06, 0);
  root.add(slab);

  for (const r of ROOMS) {
    const w = r.x1 - r.x0, d = r.z1 - r.z0;
    const f = new THREE.Mesh(
      new THREE.PlaneGeometry(w - 0.04, d - 0.04),
      new THREE.MeshStandardMaterial({ color: r.floor, roughness: 0.92, metalness: 0.02 }),
    );
    f.rotation.x = -Math.PI / 2;
    f.position.set((r.x0 + r.x1) / 2, 0.001, (r.z0 + r.z1) / 2);
    f.receiveShadow = true;
    root.add(f);
    refs.rooms[r.id] = { def: r, floor: f, center: V((r.x0 + r.x1) / 2, 0, (r.z0 + r.z1) / 2) };

    // Per-room flood plane used for the "whole room flashes" alert.
    const flood = new THREE.Mesh(
      new THREE.PlaneGeometry(w, d),
      new THREE.MeshBasicMaterial({
        color: 0xff2222, transparent: true, opacity: 0,
        blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide,
      }),
    );
    flood.rotation.x = Math.PI / 2;
    flood.position.set((r.x0 + r.x1) / 2, APT.wallH + 0.28, (r.z0 + r.z1) / 2);
    root.add(flood);

    const fl = new THREE.PointLight(0xff2200, 0, Math.max(w, d) * 1.3, 2);
    fl.position.set((r.x0 + r.x1) / 2, APT.wallH - 0.2, (r.z0 + r.z1) / 2);
    root.add(fl);

    const amb = new THREE.PointLight(0xffe6c4, 20, 20, 2);
    amb.position.set((r.x0 + r.x1) / 2, APT.wallH + 0.9, (r.z0 + r.z1) / 2);
    root.add(amb);
    refs.roomLights.push(amb);

    refs.floods[r.id] = { plane: flood, light: fl };
  }

  // -----------------------------------------------------------------------
  //  Walls — translucent so the interior stays readable from any angle
  // -----------------------------------------------------------------------
  const wallMat = new THREE.MeshStandardMaterial({
    color: 0x8ea3bd, transparent: true, opacity: 0.16,
    roughness: 0.9, depthWrite: false, side: THREE.DoubleSide,
  });
  const capMaterial = new THREE.MeshStandardMaterial({
    color: 0x93a9c6, emissive: 0x0d1622, roughness: 0.6,
  });

  for (const [x0, z0, x1, z1, ext] of WALLS) {
    const dx = x1 - x0, dz = z1 - z0;
    const len = Math.hypot(dx, dz);
    if (len < 0.001) continue;
    const h = ext ? APT.wallH : APT.wallH - 0.15;
    const w = Math.abs(dx) > Math.abs(dz) ? len : APT.wallT;
    const d = Math.abs(dx) > Math.abs(dz) ? APT.wallT : len;
    const m = box(w, h, d, wallMat, (x0 + x1) / 2, h / 2, (z0 + z1) / 2);
    m.castShadow = false;
    m.receiveShadow = false;
    root.add(m);
    // bright cap line along the top edge
    const cap = box(w, 0.035, d, capMaterial, (x0 + x1) / 2, h + 0.015, (z0 + z1) / 2);
    root.add(cap);
  }

  // Windows — emissive panes on the exterior shell
  const paneMat = new THREE.MeshStandardMaterial({
    color: 0x101a26, emissive: 0x2a4f7a, emissiveIntensity: 0.7,
    transparent: true, opacity: 0.5, roughness: 0.2,
  });
  const addPane = (x, z, w, d) => root.add(box(w, 0.85, d, paneMat, x, 1.0, z));
  addPane(-2.4, -4.98, 1.5, 0.06);
  addPane(4.6, -4.98, 1.6, 0.06);
  addPane(1.2, 4.98, 1.8, 0.06);
  addPane(6.98, 1.0, 0.06, 1.6);

  // -----------------------------------------------------------------------
  //  Main door + doorbell + visitor
  // -----------------------------------------------------------------------
  const doorMat = matte(0x4a3b2f, { roughness: 0.7 });
  const door = box(0.08, 2.02, 1.46, doorMat, -6.98, 1.01, 0);
  root.add(door);
  root.add(box(0.06, 0.06, 0.12, matte(0xc9a227, { metalness: 0.8, roughness: 0.3 }), -6.9, 1.0, 0.55));

  // doorbell push-button, outside
  const bellPlate = box(0.05, 0.14, 0.1, matte(0x1b2230), -7.25, 1.15, 0.85);
  root.add(bellPlate);
  const bellLed = new THREE.Mesh(new THREE.SphereGeometry(0.028, 12, 12), glowMat(0x3b82f6, 1.2));
  bellLed.position.set(-7.29, 1.15, 0.85);
  root.add(bellLed);
  const bellHalo = makeHalo(0x3b82f6, 0.55);
  bellHalo.position.copy(bellLed.position);
  root.add(bellHalo);
  refs.doorbellBtn = { led: bellLed, halo: bellHalo };

  // visitor standing outside
  refs.visitor = makePerson(0x2f6fbf);
  refs.visitor.position.set(-8.15, 0, 0.55);
  refs.visitor.rotation.y = Math.PI / 2;
  root.add(refs.visitor);

  // -----------------------------------------------------------------------
  //  BEDROOM
  // -----------------------------------------------------------------------
  const woodMat = matte(0x3a2f28, { roughness: 0.75 });
  const fabricMat = matte(0x2b3550, { roughness: 1 });

  root.add(box(2.0, 0.35, 2.2, woodMat, 5.6, 0.175, -3.8));
  root.add(box(1.9, 0.24, 2.1, matte(0x525f78, { roughness: 1 }), 5.6, 0.46, -3.8));
  root.add(box(2.0, 0.72, 0.1, woodMat, 5.6, 0.45, -4.92));
  root.add(box(0.85, 0.15, 0.36, matte(0xaeb6c2, { roughness: 1 }), 5.6, 0.63, -4.55));
  const blanket = box(1.88, 0.2, 1.5, fabricMat, 5.6, 0.62, -3.35);
  root.add(blanket);

  // sleeping user
  const sleeper = new THREE.Group();
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.135, 20, 16), matte(0xc9a68a, { roughness: 0.9 }));
  head.position.set(5.6, 0.72, -4.24);
  head.castShadow = true;
  sleeper.add(head);
  const bodyBump = new THREE.Mesh(
    new THREE.CapsuleGeometry(0.28, 0.9, 6, 14),
    fabricMat,
  );
  bodyBump.rotation.x = Math.PI / 2;
  bodyBump.position.set(5.6, 0.66, -3.5);
  bodyBump.castShadow = true;
  sleeper.add(bodyBump);
  root.add(sleeper);
  refs.person = sleeper;

  // nightstand
  root.add(box(1.1, 0.55, 0.8, woodMat, 3.8, 0.275, -4.05));

  // wardrobe
  root.add(box(0.62, 2.0, 1.7, matte(0x2c333f), 1.42, 1.0, -1.7));

  // ceiling RGB strip (room-wide visual channel)
  const CEIL = APT.wallH + 0.3;            // shared "ceiling" plane for fixtures
  // Unlit on purpose — a lit plate sits close enough to the room lamp to blow out.
  const plateMat = new THREE.MeshBasicMaterial({
    color: 0x46586e, transparent: true, opacity: 0.16,
    depthWrite: false, side: THREE.DoubleSide,
  });
  const ceilPlate = (x, z, s) => {
    const p = new THREE.Mesh(new THREE.PlaneGeometry(s, s), plateMat);
    p.rotation.x = Math.PI / 2;
    p.position.set(x, CEIL + 0.04, z);
    root.add(p);
  };
  ceilPlate(5.0, -3.0, 2.2);

  const strip = new THREE.Mesh(
    new THREE.TorusGeometry(0.85, 0.045, 10, 40),
    glowMat(0x223044, 0.4),
  );
  strip.rotation.x = Math.PI / 2;
  strip.position.set(5.0, CEIL, -3.0);
  root.add(strip);
  const stripHalo = makeHalo(0xff3322, 2.6);
  stripHalo.position.copy(strip.position);
  root.add(stripHalo);
  refs.ceilingStrip = { mesh: strip, halo: stripHalo };

  // -----------------------------------------------------------------------
  //  KITCHEN
  // -----------------------------------------------------------------------
  const counterMat = matte(0x27303c, { roughness: 0.6 });
  root.add(box(4.7, 0.9, 0.6, counterMat, -1.5, 0.45, -4.65));
  root.add(box(4.72, 0.06, 0.62, matte(0x39434f, { roughness: 0.35, metalness: 0.3 }), -1.5, 0.93, -4.65));
  // stove
  root.add(box(0.62, 0.02, 0.5, matte(0x14181e, { roughness: 0.4 }), -2.6, 0.965, -4.62));
  // pressure cooker
  const cookerG = new THREE.Group();
  const pot = cyl(0.17, 0.17, 0.26, matte(0x9aa6b4, { metalness: 0.85, roughness: 0.25 }));
  pot.position.y = 0.13;
  cookerG.add(pot);
  const lid = cyl(0.18, 0.18, 0.04, matte(0xb6c2cf, { metalness: 0.9, roughness: 0.2 }));
  lid.position.y = 0.28;
  cookerG.add(lid);
  const whistle = cyl(0.03, 0.045, 0.07, matte(0x6b7684, { metalness: 0.8 }));
  whistle.position.y = 0.33;
  cookerG.add(whistle);
  cookerG.position.set(-2.6, 0.975, -4.6);
  root.add(cookerG);
  refs.cooker = { group: cookerG, whistle };

  // fridge
  root.add(box(0.72, 1.8, 0.72, matte(0x3b444f, { roughness: 0.5, metalness: 0.25 }), 0.5, 0.9, -3.1));

  // sink
  root.add(box(0.5, 0.04, 0.38, matte(0x5b6774, { metalness: 0.7, roughness: 0.3 }), -0.4, 0.94, -4.65));

  // side table with a glass on it (glass-break source)
  root.add(box(0.7, 0.75, 0.7, woodMat, -3.1, 0.375, -2.1));
  const glassObj = cyl(0.055, 0.045, 0.14, new THREE.MeshStandardMaterial({
    color: 0xbfe9f5, transparent: true, opacity: 0.4, roughness: 0.1, metalness: 0.1,
  }));
  glassObj.position.set(-3.1, 0.82, -2.1);
  root.add(glassObj);
  refs.glassObj = glassObj;

  // smoke detector / fire alarm on the ceiling
  const smokeG = new THREE.Group();
  const smokeBody = cyl(0.13, 0.15, 0.06, matte(0x76808d, { roughness: 0.85 }));
  smokeG.add(smokeBody);
  const smokeLed = new THREE.Mesh(new THREE.SphereGeometry(0.022, 10, 10), glowMat(0xff2222, 0.5));
  smokeLed.position.set(0.07, -0.03, 0.05);
  smokeG.add(smokeLed);
  smokeG.position.set(-2.9, CEIL, -3.5);
  ceilPlate(-2.9, -3.5, 0.85);
  root.add(smokeG);
  const smokeHalo = makeHalo(0xff3311, 1.1);
  smokeHalo.position.copy(smokeG.position);
  root.add(smokeHalo);
  refs.smoke = { group: smokeG, led: smokeLed, halo: smokeHalo };

  // -----------------------------------------------------------------------
  //  LIVING ROOM
  // -----------------------------------------------------------------------
  // sofa
  const sofa = new THREE.Group();
  sofa.add(box(2.4, 0.38, 0.95, matte(0x35405c, { roughness: 1 }), 0, 0.19, 0));
  sofa.add(box(2.4, 0.5, 0.22, matte(0x3c4869, { roughness: 1 }), 0, 0.44, 0.4));
  sofa.add(box(0.22, 0.42, 0.95, matte(0x3c4869, { roughness: 1 }), -1.1, 0.4, 0));
  sofa.add(box(0.22, 0.42, 0.95, matte(0x3c4869, { roughness: 1 }), 1.1, 0.4, 0));
  sofa.position.set(0.6, 0, 3.7);
  root.add(sofa);

  root.add(box(1.3, 0.06, 0.65, matte(0x4a3b2f), 0.6, 0.42, 2.2));
  root.add(box(0.06, 0.4, 0.06, woodMat, 0.05, 0.2, 2.2));
  root.add(box(0.06, 0.4, 0.06, woodMat, 1.15, 0.2, 2.2));

  // rug
  const rug = new THREE.Mesh(new THREE.PlaneGeometry(3.4, 2.2),
    new THREE.MeshStandardMaterial({ color: 0x2a3145, roughness: 1 }));
  rug.rotation.x = -Math.PI / 2;
  rug.position.set(0.6, 0.006, 2.7);
  rug.receiveShadow = true;
  root.add(rug);

  // TV on the east wall — the distractor sound source
  root.add(box(0.08, 0.82, 1.45, matte(0x0d1117, { roughness: 0.4 }), 6.9, 1.25, 2.2));
  const tvScreen = new THREE.Mesh(
    new THREE.PlaneGeometry(1.34, 0.72),
    new THREE.MeshStandardMaterial({ color: 0x0a1520, emissive: 0x1b3a5c, emissiveIntensity: 0.6 }),
  );
  tvScreen.rotation.y = -Math.PI / 2;
  tvScreen.position.set(6.85, 1.25, 2.2);
  root.add(tvScreen);
  refs.tvScreen = tvScreen;
  root.add(box(0.4, 0.45, 1.6, matte(0x2c333f), 6.7, 0.225, 2.2));

  // crib (baby sound source)
  const crib = new THREE.Group();
  crib.add(box(0.95, 0.1, 0.62, woodMat, 0, 0.55, 0));
  for (const [dx, dz] of [[-0.44, -0.28], [0.44, -0.28], [-0.44, 0.28], [0.44, 0.28]]) {
    crib.add(box(0.06, 0.6, 0.06, woodMat, dx, 0.3, dz));
  }
  crib.add(box(0.9, 0.14, 0.56, matte(0xaeb6c2, { roughness: 1 }), 0, 0.67, 0));
  crib.position.set(2.1, 0, 3.4);
  root.add(crib);
  refs.crib = crib;

  // desk (hub)
  root.add(box(0.78, 0.06, 1.6, matte(0x4a3b2f), -3.5, 0.72, 0.6));
  root.add(box(0.07, 0.72, 0.07, woodMat, -3.5, 0.36, -0.1));
  root.add(box(0.07, 0.72, 0.07, woodMat, -3.5, 0.36, 1.3));

  // shoe rack in the corridor
  root.add(box(0.4, 0.5, 1.2, matte(0x2c333f), -6.7, 0.25, 3.2));

  // -----------------------------------------------------------------------
  //  DEVICES
  // -----------------------------------------------------------------------
  for (const def of DEVICES) {
    const built = buildDevice(def, refs);
    built.group.position.set(...def.pos);
    built.group.rotation.y = def.yaw;
    root.add(built.group);

    // coverage dome
    if (def.range > 0) {
      const ring = new THREE.Mesh(
        new THREE.RingGeometry(def.range - 0.05, def.range, 64),
        new THREE.MeshBasicMaterial({
          color: 0x38bdf8, transparent: true, opacity: 0.18,
          side: THREE.DoubleSide, depthWrite: false,
        }),
      );
      ring.rotation.x = -Math.PI / 2;
      ring.position.set(def.pos[0], 0.02, def.pos[2]);
      ring.visible = false;
      root.add(ring);
      const disc = new THREE.Mesh(
        new THREE.CircleGeometry(def.range, 64),
        new THREE.MeshBasicMaterial({
          color: 0x38bdf8, transparent: true, opacity: 0.045,
          side: THREE.DoubleSide, depthWrite: false,
        }),
      );
      disc.rotation.x = -Math.PI / 2;
      disc.position.set(def.pos[0], 0.015, def.pos[2]);
      disc.visible = false;
      root.add(disc);
      built.coverage = [ring, disc];
    }

    refs.devices.set(def.id, built);
  }

  return refs;
}

// ===========================================================================
//  Device construction — every child mesh carries explode metadata
// ===========================================================================
function part(mesh, name, price, ex) {
  mesh.userData.part = { name, price };
  mesh.userData.explode = ex;
  mesh.userData.home = mesh.position.clone();
  return mesh;
}

function buildDevice(def, refs) {
  const g = new THREE.Group();
  g.userData.deviceId = def.id;
  const out = { def, group: g, parts: [], led: null, halo: null, screen: null };

  const pcbMat = matte(0x0f3d2e, { roughness: 0.6 });
  const shellMat = matte(0xe8edf3, { roughness: 0.55 });
  const darkMat = matte(0x1a212b, { roughness: 0.6 });

  if (def.kind === 'node') {
    // shell
    const shell = part(box(0.20, 0.13, 0.055, shellMat, 0, 0, 0), 'Enclosure + magnet mount', 80, V(0, 0, -0.22));
    g.add(shell);
    // ESP32-S3 board
    const pcb = part(box(0.15, 0.085, 0.014, pcbMat, 0, 0, 0.02), 'ESP32-S3 DevKitC', 650, V(0, 0.16, 0.05));
    g.add(pcb);
    const can = part(box(0.045, 0.03, 0.008, matte(0xb9c2cc, { metalness: 0.8, roughness: 0.3 }), -0.04, 0.02, 0.03),
      'Wi-Fi / BLE radio', 0, V(0, 0.16, 0.05));
    g.add(can);
    // INMP441 mic
    const mic = part(box(0.032, 0.032, 0.01, matte(0x141a22), 0.055, -0.02, 0.031), 'INMP441 I²S mic', 180, V(0.2, 0.06, 0.16));
    g.add(mic);
    const grille = new THREE.Mesh(new THREE.CircleGeometry(0.02, 20),
      new THREE.MeshStandardMaterial({ color: 0x0a0e14, roughness: 0.9 }));
    grille.position.set(0.055, -0.02, 0.0295);
    g.add(grille);
    out.micAnchor = V(0.055, -0.02, 0.05);
    // WS2812 status LED
    const led = new THREE.Mesh(new THREE.BoxGeometry(0.026, 0.026, 0.008), glowMat(0x22c55e, 1.0));
    led.position.set(-0.06, -0.025, 0.031);
    part(led, 'WS2812 status LED', 25, V(-0.2, 0.06, 0.16));
    g.add(led);
    out.led = led;
    // battery
    const bat = part(cyl(0.023, 0.023, 0.115, matte(0x2b6cb0, { roughness: 0.5 })), '18650 + TP4056 + MT3608 boost', 235, V(0, -0.2, -0.02));
    bat.rotation.z = Math.PI / 2;
    bat.position.set(0, -0.03, -0.012);
    bat.userData.home = bat.position.clone();
    g.add(bat);
    // magnet ring on the back
    const mag = part(cyl(0.022, 0.022, 0.01, matte(0x6b7280, { metalness: 0.9, roughness: 0.4 })), 'Neodymium mount', 0, V(0, -0.1, -0.3));
    mag.rotation.x = Math.PI / 2;
    mag.position.set(0, 0, -0.03);
    mag.userData.home = mag.position.clone();
    g.add(mag);

    const halo = makeHalo(0x22c55e, 0.42);
    halo.position.copy(led.position);
    g.add(halo);
    out.halo = halo;

    // floating spectrum panel (billboard)
    const scr = makeScreen(640, 200);
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
      map: scr.texture, transparent: true, depthTest: false, opacity: 0,
    }));
    sprite.scale.set(1.6, 0.5, 1);          // overwritten per-frame to hold screen size
    sprite.position.set(0, 0.62, 0);
    sprite.renderOrder = 20;
    g.add(sprite);
    out.spectrum = { scr, sprite };
  }

  if (def.kind === 'beacon') {
    const base = part(cyl(0.115, 0.135, 0.075, shellMat, 32), 'Enclosure', 80, V(0, -0.25, 0));
    base.position.y = 0.037;
    base.userData.home = base.position.clone();
    g.add(base);

    const bodyG = cyl(0.105, 0.105, 0.16, matte(0x1e252f, { roughness: 0.5 }), 32);
    bodyG.position.y = 0.155;
    part(bodyG, 'ESP32 DevKit + MOSFET driver', 375, V(0, 0.02, -0.3));
    g.add(bodyG);

    // WS2812 ring
    const ring = new THREE.Mesh(new THREE.TorusGeometry(0.098, 0.016, 10, 40), glowMat(0x2c3a4d, 0.5));
    ring.rotation.x = Math.PI / 2;
    ring.position.y = 0.245;
    part(ring, 'WS2812 ring ×16', 220, V(0, 0.3, 0));
    g.add(ring);
    out.led = ring;
    const halo = makeHalo(0x3b82f6, 0.5);
    halo.position.y = 0.245;
    g.add(halo);
    out.halo = halo;

    // OLED
    const scr = makeScreen(512, 256);
    const oled = new THREE.Mesh(
      new THREE.PlaneGeometry(0.15, 0.075),
      new THREE.MeshBasicMaterial({ map: scr.texture, toneMapped: false }),
    );
    oled.position.set(0, 0.17, 0.107);
    part(oled, 'SSD1306 0.96" OLED', 190, V(0, 0.02, 0.34));
    g.add(oled);
    const bezel = box(0.17, 0.095, 0.006, darkMat, 0, 0.17, 0.104);
    g.add(bezel);
    out.screen = scr;
    refs.screens.beacon = scr;

    // vibration motor
    const motor = part(cyl(0.026, 0.026, 0.016, matte(0x9aa3ad, { metalness: 0.7 }), 20), 'ERM vibration motor', 60, V(0.28, -0.05, 0.05));
    motor.rotation.z = Math.PI / 2;
    motor.position.set(0.08, 0.08, 0);
    motor.userData.home = motor.position.clone();
    g.add(motor);

    const bat = part(cyl(0.023, 0.023, 0.11, matte(0x2b6cb0, { roughness: 0.5 })), '18650 + TP4056 + MT3608 boost', 235, V(-0.3, -0.05, 0));
    bat.rotation.z = Math.PI / 2;
    bat.position.set(0, 0.09, -0.05);
    bat.userData.home = bat.position.clone();
    g.add(bat);
  }

  if (def.kind === 'shaker') {
    const disc = part(cyl(0.075, 0.075, 0.022, matte(0x2b3550, { roughness: 1 }), 28), 'Fabric pouch', 40, V(0, 0.18, 0));
    g.add(disc);
    const motor = part(cyl(0.028, 0.028, 0.016, matte(0x9aa3ad, { metalness: 0.7 }), 20), 'ERM vibration motor', 120, V(0, -0.16, 0));
    motor.position.y = -0.004;
    motor.userData.home = motor.position.clone();
    g.add(motor);
    const drv = part(box(0.05, 0.006, 0.03, matte(0x0f3d2e), 0, 0.014, 0), 'MOSFET + flyback diode', 30, V(0.2, 0.05, 0));
    g.add(drv);
    out.led = motor;
    const halo = makeHalo(0xef4444, 0.45);
    halo.position.y = 0.06;
    g.add(halo);
    out.halo = halo;
  }

  if (def.kind === 'band') {
    const strap = part(new THREE.Mesh(new THREE.TorusGeometry(0.055, 0.014, 10, 32), matte(0x1e252f, { roughness: 0.9 })),
      'Strap + case', 90, V(0, -0.18, 0));
    strap.rotation.x = Math.PI / 2;
    strap.userData.home = strap.position.clone();
    g.add(strap);
    const head = part(box(0.062, 0.018, 0.042, matte(0x11161d), 0, 0.02, 0), 'ESP32-C3 Super Mini', 280, V(0, 0.16, 0));
    g.add(head);
    const scr = makeScreen(256, 128);
    const face = new THREE.Mesh(new THREE.PlaneGeometry(0.05, 0.03),
      new THREE.MeshBasicMaterial({ map: scr.texture, toneMapped: false }));
    face.rotation.x = -Math.PI / 2;
    face.position.set(0, 0.0295, 0);
    part(face, '0.96" OLED', 190, V(0, 0.28, 0));
    g.add(face);
    out.screen = scr;
    refs.screens.band = scr;
    const motor = part(cyl(0.016, 0.016, 0.01, matte(0x9aa3ad, { metalness: 0.7 }), 16), 'ERM motor', 60, V(0.2, 0.06, 0));
    motor.position.set(0.035, 0.012, 0);
    motor.userData.home = motor.position.clone();
    g.add(motor);
    const bat = part(box(0.03, 0.008, 0.024, matte(0x2b6cb0), -0.028, 0.012, 0), '400 mAh LiPo', 150, V(-0.2, 0.06, 0));
    g.add(bat);
    const halo = makeHalo(0xa855f7, 0.3);
    halo.position.y = 0.05;
    g.add(halo);
    out.halo = halo;
  }

  if (def.kind === 'phone') {
    const bodyM = part(box(0.075, 0.009, 0.152, matte(0x14181f, { roughness: 0.45 })), 'Existing phone', 0, V(0, 0, 0));
    g.add(bodyM);
    const scr = makeScreen(360, 740);
    const face = new THREE.Mesh(new THREE.PlaneGeometry(0.068, 0.142),
      new THREE.MeshBasicMaterial({ map: scr.texture, toneMapped: false }));
    face.rotation.x = -Math.PI / 2;
    face.position.y = 0.0048;
    g.add(face);
    out.screen = scr;
    out.screenMesh = face;
    refs.screens.phone = scr;
    const halo = makeHalo(0x3b82f6, 0.35);
    halo.position.y = 0.03;
    g.add(halo);
    out.halo = halo;
  }

  if (def.kind === 'hub') {
    const base = part(box(0.34, 0.014, 0.24, matte(0x2f3742, { roughness: 0.4, metalness: 0.3 })), 'Laptop (existing)', 0, V(0, 0, 0));
    g.add(base);
    const lid = new THREE.Group();
    const lidPanel = box(0.34, 0.225, 0.011, matte(0x2f3742, { roughness: 0.4, metalness: 0.3 }), 0, 0.112, 0);
    lid.add(lidPanel);
    const scr = makeScreen(640, 400);
    const face = new THREE.Mesh(new THREE.PlaneGeometry(0.315, 0.2),
      new THREE.MeshBasicMaterial({ map: scr.texture, toneMapped: false }));
    face.position.set(0, 0.112, 0.007);
    lid.add(face);
    lid.position.set(0, 0.007, -0.115);
    lid.rotation.x = -0.32;
    g.add(lid);
    out.screen = scr;
    refs.screens.hub = scr;
    const halo = makeHalo(0x22c55e, 0.5);
    halo.position.set(0, 0.12, -0.09);
    g.add(halo);
    out.halo = halo;
  }

  g.traverse((o) => {
    if (o.isMesh) {
      o.castShadow = true;
      if (o.userData.part) out.parts.push(o);
      if (!o.userData.home) o.userData.home = o.position.clone();
    }
  });

  return out;
}

// ---------------------------------------------------------------------------
function makePerson(color) {
  const g = new THREE.Group();
  const m = matte(color, { roughness: 0.9 });
  const legs = new THREE.Mesh(new THREE.CapsuleGeometry(0.13, 0.42, 5, 12), matte(0x28303c));
  legs.position.y = 0.4;
  g.add(legs);
  const torso = new THREE.Mesh(new THREE.CapsuleGeometry(0.17, 0.42, 5, 14), m);
  torso.position.y = 0.98;
  g.add(torso);
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.125, 18, 14), matte(0xc9a68a, { roughness: 0.9 }));
  head.position.y = 1.36;
  g.add(head);
  g.traverse((o) => { if (o.isMesh) o.castShadow = true; });
  return g;
}
