import { Application, Graphics, Text } from "pixi.js";
import "./style.css";

type Unit = {
  id: string;
  team: number;
  x: number;
  y: number;
  hp: number;
  max_hp: number;
  heading: number;
  target_id: string | null;
  alive: boolean;
};

type State = {
  time: number;
  step: number;
  world: { w: number; h: number };
  units: Unit[];
  stats: {
    blue_alive: number;
    red_alive: number;
    blue_damage: number;
    red_damage: number;
    blue_fire_efficiency: number;
    red_fire_efficiency: number;
  };
};

const app = new Application();
await app.init({
  resizeTo: window,
  background: "#030507",
  antialias: true,
});

document.getElementById("app")!.appendChild(app.canvas);

const worldLayer = new Graphics();
const unitLayer = new Graphics();
const trailLayer = new Graphics();

app.stage.addChild(worldLayer);
app.stage.addChild(trailLayer);
app.stage.addChild(unitLayer);

const hud = document.createElement("div");
hud.className = "hud";
hud.innerHTML = `
  <div class="title">WAR SIMULATOR · 100 UNITS</div>
  <div class="stats" id="stats">connecting...</div>
`;
document.body.appendChild(hud);

const bottom = document.createElement("div");
bottom.className = "bottom";
bottom.innerHTML = `
  <span id="clock">00:00</span>
  <span id="speed">PLAY SPEED: 10×</span>
`;
document.body.appendChild(bottom);

const clockEl = document.getElementById("clock")!;
const statsEl = document.getElementById("stats")!;

const positions = new Map<string, {x: number, y: number}[]>();

function worldToScreen(x: number, y: number, world: State["world"]) {
  const top = 70;
  const bottom = 50;
  const scale = Math.min(
    window.innerWidth / world.w,
    (window.innerHeight - top - bottom) / world.h
  );
  const ox = (window.innerWidth - world.w * scale) / 2;
  const oy = top + (window.innerHeight - top - bottom - world.h * scale) / 2;
  return { x: ox + x * scale, y: oy + y * scale, scale };
}

function draw(state: State) {
  worldLayer.clear();
  trailLayer.clear();
  unitLayer.clear();

  // Battle-field rectangle.
  const p0 = worldToScreen(0, 0, state.world);
  const p1 = worldToScreen(state.world.w, state.world.h, state.world);
  worldLayer.rect(p0.x, p0.y, p1.x - p0.x, p1.y - p0.y);
  worldLayer.stroke({ width: 1, color: 0x334455 });

  // Center line.
  const mid = worldToScreen(state.world.w / 2, 0, state.world);
  const mid2 = worldToScreen(state.world.w / 2, state.world.h, state.world);
  worldLayer.moveTo(mid.x, mid.y);
  worldLayer.lineTo(mid2.x, mid2.y);
  worldLayer.stroke({ width: 1, color: 0x22303a, alpha: 0.5 });

  for (const u of state.units) {
    if (!u.alive) continue;

    const p = worldToScreen(u.x, u.y, state.world);
    const r = Math.max(3, 3.5 * p.scale);

    // Short trajectory history.
    const hist = positions.get(u.id) ?? [];
    hist.push({x: p.x, y: p.y});
    while (hist.length > 25) hist.shift();
    positions.set(u.id, hist);

    if (hist.length > 1) {
      for (let i = 1; i < hist.length; i++) {
        trailLayer.moveTo(hist[i - 1].x, hist[i - 1].y);
        trailLayer.lineTo(hist[i].x, hist[i].y);
      }
      trailLayer.stroke({
        width: 1,
        color: u.team === 0 ? 0x2255aa : 0xaa3322,
        alpha: 0.25,
      });
    }

    // Unit body. beginPath() discards the stale current point that pixi
    // keeps after each fill()/stroke(), otherwise the next shape gets a
    // straight segment connected to it (e.g. a line from (0,0)).
    unitLayer.beginPath();
    unitLayer.circle(p.x, p.y, r);
    unitLayer.fill({
      color: u.team === 0 ? 0x2680ff : 0xff3d32,
      alpha: 0.95,
    });

    // HP ring.
    const hp = Math.max(0, u.hp / u.max_hp);
    unitLayer.beginPath();
    unitLayer.arc(
      p.x, p.y, r + 2,
      -Math.PI / 2,
      -Math.PI / 2 + hp * Math.PI * 2
    );
    unitLayer.stroke({
      width: 1,
      color: 0xffffff,
      alpha: 0.7,
    });
  }

  const perSide = state.units.length / 2;
  statsEl.textContent =
    `BLUE ${state.stats.blue_alive}/${perSide}   ` +
    `RED ${state.stats.red_alive}/${perSide}   ` +
    `FIRE ${state.stats.blue_fire_efficiency.toFixed(2)} / ` +
    `${state.stats.red_fire_efficiency.toFixed(2)}`;

  const minutes = Math.floor(state.time / 60);
  const seconds = Math.floor(state.time % 60);
  clockEl.textContent =
    `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

const ws = new WebSocket(`ws://${location.hostname}:8000/ws`);
ws.onmessage = (event) => {
  const state = JSON.parse(event.data) as State;
  draw(state);
};
