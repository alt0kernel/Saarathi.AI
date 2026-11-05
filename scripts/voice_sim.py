#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Saarathi AI - interactive supermarket simulator (v2).

Drives the real repo modules wherever they can be imported:
  StoreNavigationGraph (Dijkstra), MultiSensorObstacleFusion (Eq. 26),
  XXAISafetyAndDPOGate (Eq. 35), GoalOrientedGraphPlanner (Eq. 18),
  caqe_refine_embeddings / retrieve_sku_from_catalogue (Eqs. 28-30).
The torch-dependent ones fall back to numpy mirrors when torch is missing.

Data comes from configs/sim_store_data.json (42 SKUs, 8-node store graph,
shopper profiles, shopping lists, obstacle types, sensor specs).

Movement (v3): FREE-ROAM is the default. You choose the direction at every junction
(arrow keys, number keys or a mouse click), can stop anywhere, U-turn mid-aisle, browse
shelves and shop the list in any order. The Dijkstra route is only a dashed hint.
Use --guided for the old "follow the route" behaviour; --auto still drives itself.

Voice input (optional, offline): pip install vosk sounddevice, download a small Vosk model
(e.g. vosk-model-small-en-us or the en-in one) into ./models/ and just talk:
"turn left", "straight", "go back", "stop", "take me to dairy", "option two", "scan",
"browse", "where am I", "what's next". Falls back to SpeechRecognition (online) if installed.
Say "help" for the full list. [V] mutes/unmutes the mic.

Run:   python scripts/interactive_simulator.py
Demo:  python scripts/interactive_simulator.py --auto --chaos
CI:    python scripts/interactive_simulator.py --headless --seed 3 --lighting dim
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import zlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from saarathi.navigation.dijkstra_planner import StoreNavigationGraph
from saarathi.navigation.obstacle_fusion import MultiSensorObstacleFusion

# --- torch-dependent repo modules (optional) -------------------------------
HAVE_TORCH = False
_TORCH_ERR = None
try:
    import torch
    from saarathi.goal_gog.gog_parser import GoalOrientedGraphPlanner
    from saarathi.planner_mocle.xxai_safety_gate import XXAISafetyAndDPOGate
    from saarathi.retrieval_sku.caqe_expansion import (
        caqe_refine_embeddings,
        retrieve_sku_from_catalogue,
    )
    HAVE_TORCH = True
except Exception as exc:  # ImportError, or a broken torch install
    _TORCH_ERR = exc

DATA_PATH = os.path.join(ROOT, "configs", "sim_store_data.json")
WALK_SPEED = 1.8      # m/s
RETREAT_SPEED = 2.2   # m/s
MAX_SENSE_RANGE = 6.0

# ---------------------------------------------------------------------------
# Backend: real repo modules when torch is present, numpy mirrors otherwise
# ---------------------------------------------------------------------------
class _FallbackGate:
    """numpy mirror of XXAISafetyAndDPOGate (same phrasing and weights)."""
    COLOR_WORDS = {"red", "blue", "green", "yellow", "orange", "purple", "pink"}

    def __init__(self, weights=(0.25, 0.35, 0.20, 0.20), tau_obs=1.2):
        self.weights, self.tau_obs = weights, tau_obs

    def compute_priority(self, p, s, q, k):
        a, b, c, d = self.weights
        return a * p + b * s + c * q + d * k

    def format_dpo_speech(self, action_directive, sku_meta=None, is_novice=True):
        words = [w for w in action_directive.split() if w.lower() not in self.COLOR_WORDS]
        cleaned = " ".join(words)
        if sku_meta:
            allergens = ", ".join(sku_meta["allergens"]) if sku_meta["allergens"] else "none"
            msg = (f"{cleaned}. Found {sku_meta['name']}, {sku_meta['tactile_desc']}. "
                   f"Price: {sku_meta['price_inr']} rupees. Allergens: {allergens}.")
        else:
            msg = cleaned
        if is_novice:
            msg += " Your alignment is spot-on, keep going!"
        return msg


def _np_caqe_refine(z, boxes, radius=150.0):
    centers = np.stack([(boxes[:, 0] + boxes[:, 2]) * 0.5,
                        (boxes[:, 1] + boxes[:, 3]) * 0.5], axis=-1)
    dist = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
    mask = (dist > 0.0) & (dist <= radius)
    sigma = radius * 0.5
    w = np.exp(-(dist ** 2) / (2.0 * sigma ** 2)) * mask
    agg = z + w @ z
    return agg / np.maximum(np.linalg.norm(agg, axis=-1, keepdims=True), 1e-12)


class Backend:
    def __init__(self, tau_obs, tau_sim, radius):
        self.tau_sim, self.radius, self.have_torch = tau_sim, radius, HAVE_TORCH
        if HAVE_TORCH:
            self.gate = XXAISafetyAndDPOGate(tau_obs=tau_obs)
            self.gog = GoalOrientedGraphPlanner()
        else:
            self.gate, self.gog = _FallbackGate(tau_obs=tau_obs), None

    @staticmethod
    def _t(a):
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))

    def priority(self, p, s, q, k):
        if self.have_torch:
            return self.gate.compute_priority(self._t(p), self._t(s), self._t(q), self._t(k)).numpy()
        return self.gate.compute_priority(p, s, q, k)

    def refine(self, z, boxes):
        if self.have_torch:
            return caqe_refine_embeddings(self._t(z), self._t(boxes), radius=self.radius).numpy()
        return _np_caqe_refine(z, boxes, self.radius)

    def retrieve(self, z_tilde, cat_z, prio):
        if self.have_torch:
            return retrieve_sku_from_catalogue(self._t(z_tilde), self._t(cat_z), self._t(prio),
                                               tau_sim=self.tau_sim)
        sims = z_tilde @ cat_z.T
        best_j, best_s = sims.argmax(-1), sims.max(-1)
        i = int(np.argmax(prio))
        conf = float(best_s[i])
        return (int(best_j[i]), conf, True) if conf >= self.tau_sim else (None, conf, False)

    def plan_chain(self, query, start, sku):
        default = [f"Navigate from {start} to {sku['aisle']}", f"Scan shelf in {sku['aisle']}",
                   f"Disambiguate {sku['name']} via CAQE", f"Announce price and allergens for {sku['name']}"]
        if self.gog is None:
            return default
        try:
            return list(self.gog.parse_request_to_tuple(query, start, sku)[3])
        except Exception:
            return default


# ---------------------------------------------------------------------------
# Speech: one worker thread, urgent interrupt, per-text cooldown
# ---------------------------------------------------------------------------
class Speaker:
    def __init__(self, enabled=True):
        self.q: "queue.Queue[str]" = queue.Queue()
        self.last: Dict[str, float] = {}
        self.proc: Optional[subprocess.Popen] = None
        self.speaking, self.busy_until, self.muted = False, 0.0, False
        self.backend = "off"
        if enabled:
            if sys.platform == "darwin" and shutil.which("say"):
                self.backend = "say"
            else:
                try:
                    import pyttsx3  # noqa: F401
                    self.backend = "pyttsx3"
                except Exception:
                    self.backend = "console"
        if self.backend != "off":
            threading.Thread(target=self._worker, daemon=True).start()

    def is_busy(self) -> bool:
        """True while TTS is talking (used so the mic does not hear and obey our own voice)."""
        return (self.speaking or time.time() < self.busy_until or not self.q.empty()
                or bool(self.proc and self.proc.poll() is None))

    def say(self, text: str, urgent: bool = False, force: bool = False):
        if self.backend == "off" or self.muted:
            return
        now = time.time()
        if not force and now - self.last.get(text, -1e9) < 5.0:
            return
        self.last[text] = now
        if urgent:
            while not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    break
            if self.proc and self.proc.poll() is None:
                self.proc.terminate()
        self.q.put(text)

    def _worker(self):
        engine = None
        if self.backend == "pyttsx3":
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty("rate", 150)
            except Exception as exc:
                print(f"[TTS] audio engine unavailable ({exc}); printing speech instead")
                self.backend = "console"
        while True:
            text = self.q.get()
            self.speaking = True
            try:
                if self.backend == "say":
                    self.proc = subprocess.Popen(["say", "-r", "170", text])
                    self.proc.wait()
                elif self.backend == "pyttsx3" and engine is not None:
                    engine.say(text)
                    engine.runAndWait()
                else:
                    print(f"[TTS] {text}")
            except Exception:
                pass
            finally:
                self.speaking, self.busy_until = False, time.time() + 0.6


# ---------------------------------------------------------------------------
# Voice input: offline Vosk (preferred) or online SpeechRecognition (fallback)
# ---------------------------------------------------------------------------
class VoiceListener:
    STOP_RE = re.compile(r"\b(stop|halt|wait|freeze)\b")

    def __init__(self, phrases, speaker=None, backend="auto", model_path=None, device=None, enabled=True):
        self.q: "queue.Queue[str]" = queue.Queue()
        self.speaker, self.enabled = speaker, enabled
        self.backend, self.errors = "off", []
        self.last_heard, self.last_t = "", 0.0
        for b in (["vosk", "sr"] if backend == "auto" else [backend]):
            try:
                self._init_vosk(phrases, model_path, device) if b == "vosk" else self._init_sr(device)
                self.backend = b
                break
            except Exception as exc:
                self.errors.append(f"{b}: {exc}")

    # -- backends --
    @staticmethod
    def find_model(explicit=None):
        import glob
        cands = [explicit, os.environ.get("VOSK_MODEL")]
        for pat in (os.path.join(ROOT, "models", "vosk-model*"), os.path.join(ROOT, "vosk-model*"),
                    os.path.expanduser("~/.cache/vosk/vosk-model*")):
            cands += sorted(glob.glob(pat))
        return next((c for c in cands if c and os.path.isdir(c)), None)

    def _init_vosk(self, phrases, model_path, device):
        import sounddevice as sd
        import vosk
        vosk.SetLogLevel(-1)
        path = self.find_model(model_path)
        if not path:
            raise RuntimeError("no Vosk model folder found (put one in ./models/ or pass --vosk-model)")
        self.rec = vosk.KaldiRecognizer(vosk.Model(path), 16000, json.dumps(list(phrases) + ["[unk]"]))
        self.audio: "queue.Queue[bytes]" = queue.Queue()
        self.stream = sd.RawInputStream(samplerate=16000, blocksize=3200, device=device, dtype="int16",
                                        channels=1, callback=lambda d, f, t, st: self.audio.put(bytes(d)))
        self.stream.start()
        threading.Thread(target=self._vosk_loop, daemon=True).start()

    def _vosk_loop(self):
        early_stop = False
        while True:
            data = self.audio.get()
            if not self.enabled:
                continue
            if self.rec.AcceptWaveform(data):
                text = json.loads(self.rec.Result()).get("text", "").strip()
                skip_dup = early_stop and bool(self.STOP_RE.search(text))
                early_stop = False
                if text and text != "[unk]" and not skip_dup:
                    self._emit(text)
            else:
                part = json.loads(self.rec.PartialResult()).get("partial", "")
                if part and not early_stop and self.STOP_RE.search(part):   # react to "stop" immediately
                    early_stop = True
                    self.q.put("stop")

    def _init_sr(self, device):
        import speech_recognition as sr
        self.sr_rec = sr.Recognizer()
        mic = sr.Microphone(device_index=device)
        with mic as src:
            self.sr_rec.adjust_for_ambient_noise(src, 0.5)

        def cb(rec, audio):
            if not self.enabled:
                return
            try:
                self._emit(rec.recognize_google(audio, language="en-IN"))
            except Exception:
                pass
        self._sr_stop = self.sr_rec.listen_in_background(mic, cb, phrase_time_limit=4)

    # -- shared --
    def _emit(self, text):
        # echo guard: while we are speaking, the mic may hear us; only obey "stop"
        if self.speaker and self.speaker.is_busy() and not self.STOP_RE.search(text.lower()):
            return
        self.q.put(text)

    def poll(self):
        try:
            t = self.q.get_nowait()
        except queue.Empty:
            return None
        self.last_heard, self.last_t = t, time.time()
        return t


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def seeded_vec(seed: int, dim: int) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(dim)
    return v / np.linalg.norm(v)


class Catalogue:
    """Reference embeddings: siblings in a family share a base direction (so
    salted vs unsalted butter really are confusable), plus a per-SKU variant."""

    def __init__(self, skus, dim, fam_w, var_w):
        self.skus = skus
        self.index = {s["sku_id"]: i for i, s in enumerate(skus)}
        Z = []
        for s in skus:
            fam = seeded_vec(zlib.crc32(s["family"].encode()), dim)
            var = seeded_vec(int(s["embeddingSeed"]), dim)
            z = fam_w * fam + var_w * var
            Z.append(z / np.linalg.norm(z))
        self.Z = np.stack(Z).astype(np.float32)
        self.family: Dict[str, List[int]] = {}
        for i, s in enumerate(skus):
            self.family.setdefault(s["family"], []).append(i)
        self.base_stock = {s["sku_id"]: int(s.get("stock", 10)) for s in skus}


@dataclass
class Obstacle:
    oid: int
    kind: str
    label: str
    a: str
    b: str
    t: float            # 0..1 from a to b
    w_real: float
    speed: float        # m/s along the edge (signed)
    ttl: Optional[float]
    warned: bool = False
    handled: bool = False


@dataclass
class Goal:
    kind: str           # "item" | "checkout"
    node: str
    sku: Optional[dict] = None
    status: str = "pending"   # pending | found | wrong | missed | blocked | skipped
    attempts: int = 0


def ascii_safe(s: str) -> str:
    for a, b in (("₹", "Rs."), ("—", "-"), ("–", "-"), ("’", "'")):
        s = s.replace(a, b)
    return s.encode("ascii", "ignore").decode()


# ---------------------------------------------------------------------------
# Simulation engine (no drawing, fully headless-testable)
# ---------------------------------------------------------------------------
class SimEngine:
    def __init__(self, data, backend, seed=7, list_name=None, profile_name="Default",
                 lighting="normal", chaos=False, auto=False, free=True):
        self.data, self.be = data, backend
        self.seed, self.chaos, self.auto = seed, chaos, auto
        self.free = free          # free-roam: the user picks every direction
        self.lighting = lighting
        self.profile_name = profile_name
        self.list_names = list(data["shopping_lists"])
        self.profile_names = list(data["profiles"])
        self.list_name = list_name or self.list_names[0]
        r = data["retrieval"]
        self.cat = Catalogue(data["skus"], r["embedding_dim"], r["family_weight"], r["variant_weight"])
        sc = data["sensors"]
        self.fusion = MultiSensorObstacleFusion(tau_obs=sc["tau_obs"], f_cam=sc["mono"]["f_cam"])
        self.reset()

    # ---- lifecycle ---------------------------------------------------------
    def reset(self):
        self.rng = np.random.default_rng(self.seed)
        self.t = 0.0
        self.nav = StoreNavigationGraph()
        for u, v, w in self.data["layout"]["extra_edges"]:
            self.nav.graph.add_edge(u, v, weight=w)
        self.base_w = {frozenset((u, v)): d["weight"] for u, v, d in self.nav.graph.edges(data=True)}
        self.coords = {n: tuple(v["xy"]) for n, v in self.data["layout"]["nodes"].items()}
        self.stock = dict(self.cat.base_stock)
        self.avoid = set(self.data["profiles"][self.profile_name]["avoid_allergens"])
        self.obstacles: List[Obstacle] = []
        self.blocked: Dict[frozenset, set] = {}
        self._oid = 0
        self.node, self.prev_node, self.heading_to, self.prog = "Entrance", None, None, 0.0
        self.facing: Optional[str] = None     # neighbour the user is facing while standing at a node
        self.travel: Optional[str] = None     # "take me to X": auto-walk destination (free-roam)
        self.route = ["Entrance"]
        self.walking = self.retreating = self.no_route = False
        self.state = "NAVIGATING"
        self.auto_timer = 0.0
        self.chaos_timer = float(self.rng.uniform(6, 12))
        self.replan_pending = False
        self.encourage = True
        self.cart: List[dict] = []
        self.scan_log: List[dict] = []
        self.events: List[dict] = []
        self.speech_out: List[Tuple[str, bool]] = []
        self.last_speech = ""
        self.last_scan: Optional[dict] = None
        self.snap_flags: set = set()
        self.m = dict(dist_m=0.0, hazards=0, reroutes=0, scans=0, verified=0, wrong_item=0,
                      low_conf=0, misread_picks=0, allergen_blocks=0, missed=0)
        self.sens = dict(rgbd=6.0, tof=4.0, tri=300.0, fused=4.0, hazard=False, ob=None, d_true=None)
        self.log("start", f"list='{self.list_name}' profile='{self.profile_name}' "
                          f"lighting={self.lighting} seed={self.seed}")
        self.goals = self._build_goals()
        self.goal_idx = 0
        self.chain: List[str] = []
        self._begin_goal()
        if len(self.route) > 1:
            self.facing = self.route[1]

    # ---- small utilities ---------------------------------------------------
    def log(self, kind, msg):
        self.events.append({"t": round(self.t, 2), "type": kind, "msg": msg})

    def say(self, directive, urgent=False, sku_meta=None, novice=False, force=False):
        if sku_meta is None and directive[-1:] not in ".!?":
            directive += "."   # the gate appends its encouragement straight after the directive
        text =self.be.gate.format_dpo_speech(directive, sku_meta=sku_meta, is_novice=novice)
        self.last_speech = text
        self.speech_out.append((text, "force" if force else urgent))

    @property
    def goal(self) -> Goal:
        return self.goals[min(self.goal_idx, len(self.goals) - 1)]

    def elen(self, a, b):
        return self.base_w[frozenset((a, b))]

    def _route(self, a, b):
        r, c = self.nav.shortest_route(a, b)
        return r, c

    def user_xy(self):
        x0, y0 = self.coords[self.node]
        if self.heading_to is None:
            return x0, y0
        x1, y1 = self.coords[self.heading_to]
        return x0 + (x1 - x0) * self.prog, y0 + (y1 - y0) * self.prog

    @property
    def manual(self) -> bool:
        return self.free and not self.auto

    def _unit(self, a, b):
        (x0, y0), (x1, y1) = self.coords[a], self.coords[b]
        n = math.hypot(x1 - x0, y1 - y0) or 1.0
        return ((x1 - x0) / n, (y1 - y0) / n)

    def _name(self, node):
        return self.data["layout"]["nodes"][node].get("label") or node

    def _neighbors(self, node):
        return sorted(self.nav.graph.neighbors(node),
                      key=lambda n: math.atan2(self.coords[n][1] - self.coords[node][1],
                                               self.coords[n][0] - self.coords[node][0]))

    def _straight_next(self, prev, node):
        """Neighbour of `node` that continues most nearly straight from `prev`."""
        nbrs = [n for n in self.nav.graph.neighbors(node) if n != prev]
        if not nbrs:
            return prev
        if prev is None:
            return nbrs[0]
        v = self._unit(prev, node)
        return max(nbrs, key=lambda n: v[0] * self._unit(node, n)[0] + v[1] * self._unit(node, n)[1])

    def _rel_word(self, prev, a, b):
        if prev is None:
            if not self.facing:
                return "toward"
            v1, v2 = self._unit(a, self.facing), self._unit(a, b)
        else:
            v1, v2 = self._unit(prev, a), self._unit(a, b)
        ang = math.degrees(math.atan2(v1[0] * v2[1] - v1[1] * v2[0], v1[0] * v2[0] + v1[1] * v2[1]))
        if abs(ang) < 25:
            return "ahead"
        if abs(ang) > 150:
            return "behind"
        return "right" if ang > 0 else "left"

    def _hint(self):
        """Free-roam: the planned route is only a suggestion, recomputed from where you stand."""
        try:
            self.route, _ = self._route(self.node, self.travel or self.goal.node)
        except Exception:
            self.route = [self.node]

    def options(self):
        """Directions available right now (numbered for the 1-9 keys and the on-map chips)."""
        out = []
        if self.heading_to is not None:       # mid-aisle: carry on, or turn back
            a, b = self.node, self.heading_to
            L = self.elen(a, b)
            out.append(dict(key=1, node=b, dist=(1 - self.prog) * L, word="continue",
                            blocked=frozenset((a, b)) in self.blocked))
            out.append(dict(key=2, node=a, dist=self.prog * L, word="back",
                            blocked=False))
        else:
            for i, n in enumerate(self._neighbors(self.node), 1):
                out.append(dict(key=i, node=n, dist=self.elen(self.node, n),
                                word=self._rel_word(self.prev_node, self.node, n),
                                blocked=frozenset((self.node, n)) in self.blocked))
        return out[:9]

    def heading_vec(self):
        nxt = self.heading_to or (self.facing if self.manual else None) or \
            (self.route[1] if len(self.route) > 1 else None)
        if nxt is None:
            return None
        (x0, y0), (x1, y1) = self.coords[self.node], self.coords[nxt]
        n = math.hypot(x1 - x0, y1 - y0) or 1.0
        v = ((x1 - x0) / n, (y1 - y0) / n)
        return (-v[0], -v[1]) if self.retreating else v

    # ---- goals -------------------------------------------------------------
    def _build_goals(self) -> List[Goal]:
        items = [self.data["skus"][self.cat.index[i]] for i in self.data["shopping_lists"][self.list_name]]
        remaining = list(dict.fromkeys(s["aisle"] for s in items))
        order, cur = [], "Entrance"
        while remaining:
            nxt = min(remaining, key=lambda a: self._route(cur, a)[1])
            order.append(nxt)
            remaining.remove(nxt)
            cur = nxt
        goals = [Goal("item", a, s) for a in order for s in items if s["aisle"] == a]
        goals.append(Goal("checkout", "Checkout"))
        return goals

    def _begin_goal(self):
        g = self.goal
        self.route, _ = self._route(self.node, g.node)
        if g.kind == "item":
            self.chain = self.be.plan_chain(f"find {g.sku['name']}", self.node, g.sku)
            d = sum(self.elen(a, b) for a, b in zip(self.route, self.route[1:]))
            self.log("goal", f"{g.sku['name']} -> {g.node} ({d:.1f} m)")
            if self.node != g.node:
                self.say(f"Next item {g.sku['name']} in {g.node}, {d:.1f} meters", novice=self.encourage)
                self.encourage = False
        else:
            self.chain = [f"Navigate from {self.node} to Checkout"]
            self.log("goal", "Checkout")
            self.say("All items handled. Heading to checkout")
        self._refresh_state()

    def _finish_goal(self, status):
        self.goal.status = status
        if status == "found":
            self.encourage = True
        self.goal_idx += 1
        self._begin_goal()
        if self.goal.kind == "checkout" and self.node == "Checkout":
            self._complete()

    def _complete(self):
        self.state = "COMPLETE"
        got = sum(1 for g in self.goals if g.kind == "item" and g.status == "found")
        n = sum(1 for g in self.goals if g.kind == "item")
        total = sum(c["actual"]["price_inr"] for c in self.cart)
        self.say(f"Checkout reached. {got} of {n} items collected. Total {total:.0f} rupees", urgent=True)
        self.log("complete", f"{got}/{n} items, Rs.{total:.0f}")
        self.snap_flags.add("complete")

    # ---- obstacles ---------------------------------------------------------
    def _ahead_segments(self):
        """[(a, b, f0, L, offset_m)] for the path in front of the user."""
        if self.retreating:
            return []
        if self.manual:       # look along the way the user is actually facing
            segs = []
            if self.heading_to is not None:
                a, b, f0 = self.node, self.heading_to, self.prog
            elif self.facing:
                a, b, f0 = self.node, self.facing, 0.0
            else:
                return []
            L = self.elen(a, b)
            segs.append((a, b, f0, L, 0.0))
            off = (1 - f0) * L
            nxt = self._straight_next(a, b)
            if nxt and nxt != a and off <= MAX_SENSE_RANGE:
                segs.append((b, nxt, 0.0, self.elen(b, nxt), off))
            return segs
        if self.heading_to is not None:
            chain, f0 = [self.node] + self.route[1:], self.prog
        else:
            chain, f0 = list(self.route), 0.0
        segs, off = [], 0.0
        for i in range(len(chain) - 1):
            a, b = chain[i], chain[i + 1]
            L, start = self.elen(a, b), (f0 if i == 0 else 0.0)
            segs.append((a, b, start, L, off))
            off += (1 - start) * L
            if off > MAX_SENSE_RANGE:
                break
        return segs

    def _nearest_ahead(self):
        best = None
        segs = self._ahead_segments()
        for ob in self.obstacles:
            for a, b, f0, L, off in segs:
                if {a, b} == {ob.a, ob.b}:
                    f = ob.t if (ob.a, ob.b) == (a, b) else 1 - ob.t
                    if f >= f0 - 1e-6:
                        d = max(0.1, off + (f - f0) * L - ob.w_real * 0.5)
                        if best is None or d < best[1]:
                            best = (ob, d)
                    break
        return best if best else (None, None)

    def spawn_obstacle(self, ttl=None, moving=False) -> Optional[Obstacle]:
        segs = self._ahead_segments()
        if not segs:
            return None
        target = float(self.rng.uniform(2.2, 4.0))
        pick = segs[-1]
        for s in segs:
            if target <= s[4] + (1 - s[2]) * s[3]:
                pick = s
                break
        a, b, f0, L, off = pick
        f = min(0.85, max(f0 + 0.15, f0 + (target - off) / L))
        kinds = self.data["obstacle_types"]
        k = kinds[int(self.rng.integers(len(kinds)))]
        speed = k["speed_mps"] * float(self.rng.choice([-1, 1])) if moving else 0.0
        self._oid += 1
        ob = Obstacle(self._oid, k["kind"], k["label"], a, b, f, k["w_real_m"], speed, ttl)
        self.obstacles.append(ob)
        self.log("obstacle", f"{ob.label} spawned on {a}-{b}")
        return ob

    def toggle_obstacle(self):
        if self.obstacles:
            for ob in list(self.obstacles):
                self._remove_obstacle(ob)
        elif self.spawn_obstacle() is None:
            self.log("info", "No path ahead to place an obstacle on")

    def _block(self, ob):
        self.blocked.setdefault(frozenset((ob.a, ob.b)), set()).add(ob.oid)
        self.nav.mark_edge_blocked(ob.a, ob.b)

    def _remove_obstacle(self, ob):
        if ob in self.obstacles:
            self.obstacles.remove(ob)
        key = frozenset((ob.a, ob.b))
        ids = self.blocked.get(key)
        if ids and ob.oid in ids:
            ids.discard(ob.oid)
            if not ids:
                del self.blocked[key]
                self.nav.graph[ob.a][ob.b]["weight"] = self.base_w[key]
                self.replan_pending = True
                self.log("clear", f"{ob.label} gone, {ob.a}-{ob.b} reopened")

    def _tick_obstacles(self, dt):
        for ob in list(self.obstacles):
            if ob.speed:
                ob.t += ob.speed * dt / self.elen(ob.a, ob.b)
                if ob.t < 0.15:
                    ob.t, ob.speed = 0.15, abs(ob.speed)
                elif ob.t > 0.85:
                    ob.t, ob.speed = 0.85, -abs(ob.speed)
            if ob.ttl is not None:
                ob.ttl -= dt
                if ob.ttl <= 0:
                    self._remove_obstacle(ob)

    # ---- sensing and hazard logic -----------------------------------------
    def _sense(self):
        sc = self.data["sensors"]
        ob, d = self._nearest_ahead()
        if ob is None or d > MAX_SENSE_RANGE:
            rgbd, tof, w_real, w_pix = sc["rgbd"]["max_range_m"], sc["tof"]["max_range_m"], 0.5, 1.0
            ob = None
        else:
            rgbd = min(sc["rgbd"]["max_range_m"], max(0.1, d + self.rng.normal(0, sc["rgbd"]["noise_std_m"])))
            tof = min(sc["tof"]["max_range_m"], max(0.1, d + self.rng.normal(0, sc["tof"]["noise_std_m"])))
            w_real = ob.w_real
            w_pix = max(1.0, w_real * sc["mono"]["f_cam"] / d + self.rng.normal(0, sc["mono"]["bbox_noise_px"]))
        fused, haz = self.fusion.estimate_min_obstacle_distance(rgbd, tof, w_real_m=w_real, bbox_width_px=w_pix)
        tri = self.fusion.triangle_similarity_distance(w_real, w_pix)
        self.sens = dict(rgbd=rgbd, tof=tof, tri=tri, fused=fused, hazard=bool(haz), ob=ob, d_true=d)

    def _hazard_logic(self):
        s = self.sens
        ob = s["ob"]
        if ob is None or self.retreating:
            return
        if self.manual:
            if s["hazard"]:
                self.walking = False
                if not ob.handled:
                    ob.handled = True
                    self.m["hazards"] += 1
                    self._block(ob)
                    self.travel = None
                    self._hint()
                    self.log("hazard", f"{ob.label} at {s['fused']:.2f} m (tau={self.fusion.tau_obs})")
                    self.snap_flags.add("hazard")
                    self.say(f"Stop immediately. {ob.label} {s['fused']:.1f} meters ahead. "
                             f"Turn around, or wait for it to clear", urgent=True)
            elif s["fused"] < 3.0 and not ob.warned:
                ob.warned = True
                self.say(f"Caution. {ob.label} about {s['fused']:.0f} meters ahead")
            return
        if s["hazard"]:
            if not ob.handled:
                ob.handled = True
                self.m["hazards"] += 1
                self._block(ob)
                self.log("hazard", f"{ob.label} at {s['fused']:.2f} m (tau={self.fusion.tau_obs})")
                self.snap_flags.add("hazard")
                self.say(f"Stop immediately. {ob.label} {s['fused']:.1f} meters ahead", urgent=True)
                old = list(self.route)
                self._replan()
                if self.no_route:
                    self.say("No safe route. Wait here until the path clears", urgent=False)
                elif self.route != old:
                    via = self.route[2] if len(self.route) > 2 and not self.retreating else self.route[1]
                    self.say(f"Rerouting via {via}")
        elif s["fused"] < 3.0 and not ob.warned:
            ob.warned = True
            self.say(f"Caution. {ob.label} about {s['fused']:.0f} meters ahead")

    def _replan(self):
        """Pick the cheaper of: continue to the next node, or step back to the last one."""
        dest = self.goal.node
        old = list(self.route)
        cands = []
        if self.heading_to is None:
            r, c = self._route(self.node, dest)
            cands.append(("stay", r, c))
        else:
            a, b = self.node, self.heading_to
            L = self.elen(a, b)
            if self.nav.graph[a][b]["weight"] < 1e5:
                r2, c2 = self._route(b, dest)
                cands.append(("continue", [a] + r2, (1 - self.prog) * L + c2))
            r1, c1 = self._route(a, dest)
            cands.append(("back", r1, self.prog * L + c1))
        mode, route, cost = min(cands, key=lambda x: x[2])
        self.no_route = cost >= 1e5
        self.route = route
        self.retreating = (mode == "back" and self.heading_to is not None
                           and not (len(route) > 1 and route[1] == self.heading_to) and not self.no_route)
        if mode == "back" and not self.retreating and self.heading_to is not None and not self.no_route:
            pass  # continuing along the same edge after all
        if self.no_route:
            self.retreating = False
        if route != old and not self.no_route:
            self.m["reroutes"] += 1
            self.log("reroute", " > ".join(route))

    # ---- movement ----------------------------------------------------------
    def _turn_word(self, prev, a, b):
        if prev is None:
            return "Walk straight"
        (px, py), (ax, ay), (bx, by) = self.coords[prev], self.coords[a], self.coords[b]
        v1, v2 = (ax - px, ay - py), (bx - ax, by - ay)
        ang = math.degrees(math.atan2(v1[0] * v2[1] - v1[1] * v2[0], v1[0] * v2[0] + v1[1] * v2[1]))
        if abs(ang) < 25:
            return "Continue straight"
        if abs(ang) > 150:
            return "Turn around"
        return "Turn right" if ang > 0 else "Turn left"

    def _announce_leg(self, word=None):
        a, b = self.node, self.heading_to
        lead = {"ahead": "Continue straight", "left": "Turn left", "right": "Turn right",
                "behind": "Turn around", "toward": "Walk straight"}.get(word) or self._turn_word(self.prev_node, a, b)
        self.say(f"{lead} toward {b}, {self.elen(a, b):.1f} meters ahead", novice=False)

    def _arrive(self, node):
        self.prev_node, self.node = self.node, node
        self.heading_to, self.prog, self.walking = None, 0.0, False
        if self.route and self.route[0] != node:
            self.route = self.route[1:] if len(self.route) > 1 and self.route[1] == node else self.route
        if len(self.route) > 1 and self.route[0] != node:
            self.route = self._route(node, self.goal.node)[0]
        g = self.goal
        if node == g.node:
            self.route = [node]
            if g.kind == "checkout":
                self._complete()
            else:
                self.say(f"Arrived at {node}. Shelf in front of you. Press S, or say scan", urgent=False)
                self.log("arrive", f"{node} (shelf for {g.sku['name']})")
        else:
            self.route, _ = self._route(node, g.node)
            self.log("node", node)

    def _arrive_manual(self, node):
        self.prev_node, self.node = self.node, node
        self.heading_to, self.prog, self.walking = None, 0.0, False
        self.facing = self._straight_next(self.prev_node, node)
        # list items can be shopped in any order: promote one that lives in this aisle
        g = self.goal
        if not (g.kind == "item" and g.node == node):
            for j in range(self.goal_idx + 1, len(self.goals)):
                c = self.goals[j]
                if c.kind == "item" and c.status == "pending" and c.node == node:
                    self.goals.insert(self.goal_idx, self.goals.pop(j))
                    self.log("goal", f"free order: {c.sku['name']} is in {node}, taking it next")
                    self._begin_goal()
                    break
        arrived_dest = self.travel == node
        if arrived_dest:
            self.travel = None
        self._hint()
        g = self.goal
        if g.kind == "checkout" and node == g.node:
            self._complete()
        elif g.kind == "item" and node == g.node:
            self.say(f"Arrived at {self._name(node)}. {g.sku['name']} is on this shelf. Press S, or say scan",
                     urgent=False)
            self.log("arrive", f"{node} (shelf for {g.sku['name']})")
        elif self.travel:                 # passing through on the way somewhere: stay quiet
            self.log("node", node)
        else:
            self.log("node", node)
            self.describe(arrival=True)

    def _move_manual(self, dt):
        self.retreating = False
        if self.heading_to is None or not self.walking:
            return
        if self.sens["hazard"]:
            self.walking = False
            return
        L = self.elen(self.node, self.heading_to)
        step = WALK_SPEED * dt
        self.prog += step / L
        self.m["dist_m"] += step
        if self.prog >= 1.0:
            self._arrive_manual(self.heading_to)

    def _move(self, dt):
        if self.manual and self.state != "COMPLETE":
            return self._move_manual(dt)
        if self.state in ("COMPLETE", "AT_SHELF"):
            self.walking = False
            return
        if self.retreating:
            L = self.elen(self.node, self.heading_to)
            self.prog -= RETREAT_SPEED * dt / L
            self.m["dist_m"] += RETREAT_SPEED * dt
            if self.prog <= 0:
                self.prog, self.heading_to, self.retreating = 0.0, None, False
            return
        if self.sens["hazard"] or self.no_route or not self.walking:
            if self.sens["hazard"] or self.no_route:
                self.walking = False
            return
        if self.heading_to is None:
            if len(self.route) < 2:
                self.walking = False
                return
            self.heading_to, self.prog = self.route[1], 0.0
            self._announce_leg()
        L = self.elen(self.node, self.heading_to)
        step = WALK_SPEED * dt
        self.prog += step / L
        self.m["dist_m"] += step
        if self.prog >= 1.0:
            self._arrive(self.heading_to)

    def _refresh_state(self):
        if self.state == "COMPLETE":
            return
        g = self.goal
        if self.sens["hazard"] or self.retreating or self.no_route:
            self.state = "HAZARD_OVERRIDE"
        elif g.kind == "item" and self.heading_to is None and self.node == g.node:
            self.state = "AT_SHELF"
        else:
            self.state = "NAVIGATING"

    # ---- player actions ----------------------------------------------------
    def walk(self):
        if self.manual:
            if self.state == "COMPLETE":
                return
            if self.heading_to is not None:
                self.walking = True
            elif self.facing:
                self.go_to(self.facing)
            return
        if self.state == "NAVIGATING":
            self.walking = True

    # ---- free-roam controls ------------------------------------------------
    def take_control(self):
        if self.auto or not self.free:
            self.auto, self.free = False, True
            self.log("mode", "manual free-roam control")

    def go_to(self, nbr, _auto=False):
        """Head toward an adjacent node (from a junction), continue, or turn back (mid-aisle)."""
        if self.state == "COMPLETE":
            return
        if not _auto:
            self.travel = None
            self.take_control()
        if self.heading_to is not None:
            if nbr == self.heading_to:
                self.walking = True
            elif nbr == self.node:
                self.reverse()
            else:
                self.say("Finish this stretch first, or turn around")
            return
        if nbr not in self.nav.graph[self.node]:
            self.say(f"{self._name(nbr)} is not directly connected. Pick one of the numbered paths")
            return
        if frozenset((self.node, nbr)) in self.blocked:
            self.facing = nbr
            self.say(f"That way is blocked. Choose another path")
            return
        word = self._rel_word(self.prev_node, self.node, nbr)   # before facing changes
        self.facing = nbr
        self.heading_to, self.prog, self.walking = nbr, 0.0, True
        self._announce_leg(word)

    def reverse(self):
        """U-turn: works mid-aisle (swap ends of the edge) or at a junction (face where you came from)."""
        if self.state == "COMPLETE":
            return
        self.take_control()
        if self.heading_to is not None:
            self.node, self.heading_to = self.heading_to, self.node
            self.prog = 1.0 - self.prog
            self.facing = self.heading_to
            self.walking = True
            self.say(f"Turning around, heading back toward {self.heading_to}")
        elif self.prev_node and self.prev_node in self.nav.graph[self.node]:
            self.go_to(self.prev_node)
        else:
            self.say("Nothing behind you")

    def stop(self):
        self.take_control()
        self.travel = None
        self.walking = False
        self.say("Stopped", urgent=True)

    def goto_place(self, node):
        """Mouse click / voice 'take me to X': adjacent = step there; otherwise auto-walk the route."""
        if self.state == "COMPLETE":
            return
        self.take_control()
        name = self._name(node)
        if self.heading_to is None:
            if node == self.node:
                self.describe()
            elif node in self.nav.graph[self.node]:
                self.go_to(node)
            else:
                self.travel = node
                self._hint()
                if len(self.route) < 2 or frozenset((self.route[0], self.route[1])) in self.blocked:
                    self.travel = None
                    self.say(f"No clear way to {name} right now")
                    return
                dist = sum(self.elen(a, b) for a, b in zip(self.route, self.route[1:]))
                self.log("travel", f"to {node} via {' > '.join(self.route)}")
                self.say(f"Taking you to {name}, {dist:.1f} meters. Say stop to halt")
        else:                                   # mid-aisle: go whichever way is cheaper
            a, b = self.node, self.heading_to
            L = self.elen(a, b)
            if node == a:
                self.reverse()
            elif node == b:
                self.walking = True
            else:
                try:
                    fwd = (1 - self.prog) * L + self._route(b, node)[1]
                    back = self.prog * L + self._route(a, node)[1]
                except Exception:
                    fwd, back = 0, 1
                if back < fwd:
                    self.reverse()
                self.travel = node
                self.walking = True
                self.say(f"Taking you to {name}. Say stop to halt")

    def toggle_walk(self):
        if self.manual and self.heading_to is not None:
            self.walking = not self.walking
            self.say("Standing still" if not self.walking else "Walking on")
        elif self.manual:
            self.walk()

    def steer(self, dx, dy):
        """Arrow keys: map-relative direction. Picks the neighbour that best matches, or
        continues / reverses when you are mid-aisle."""
        if self.state == "COMPLETE":
            return
        self.take_control()
        d = math.hypot(dx, dy) or 1.0
        dx, dy = dx / d, dy / d
        if self.heading_to is not None:
            v = self._unit(self.node, self.heading_to)
            c = v[0] * dx + v[1] * dy
            if c > 0.3:
                self.walking = True
            elif c < -0.3:
                self.reverse()
            else:
                self.say("You are in an aisle. Go on, or turn back")
            return
        best, bc = None, 0.0
        for n in self._neighbors(self.node):
            u = self._unit(self.node, n)
            c = u[0] * dx + u[1] * dy
            if c > bc:
                best, bc = n, c
        if best is None or bc < 0.35:
            self.say("No path that way")
        else:
            self.go_to(best)

    def choose_number(self, k):
        opts = self.options()
        if 1 <= k <= len(opts):
            self.go_to(opts[k - 1]["node"])
        else:
            self.say(f"There is no option {k}")

    def choose_node(self, node):
        """Mouse click on the map."""
        self.goto_place(node)

    def describe(self, arrival=False):
        """Say where you are and which ways you can go ('where am I?')."""
        if self.heading_to is not None:
            self.say(f"Between {self.node} and {self.heading_to}, {(1 - self.prog) * self.elen(self.node, self.heading_to):.1f} "
                     f"meters to go. Continue, or turn back")
            return
        opts = self.options()
        parts = [f"{o['word']} to {self._name(o['node'])}, {o['dist']:.1f} meters"
                 + (", blocked" if o["blocked"] else "") for o in opts]
        head = f"At {self._name(self.node)}." if not arrival else f"Reached {self._name(self.node)}."
        pend = [g for g in self.goals if g.kind == "item" and g.status == "pending" and g.node == self.node]
        extra = f" {len(pend)} list item{'s' if len(pend) != 1 else ''} here." if pend else ""
        self.say(f"{head}{extra} Paths: " + "; ".join(parts))

    # ---- voice ---------------------------------------------------------------
    NUMWORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
                "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
                "won": 1, "to": 2, "too": 2, "for": 4}

    def place_keywords(self):
        """node -> spoken keywords ('aisle two', 'dairy', 'checkout' ...), built from the layout labels."""
        words = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        out = {}
        for n, nd in self.data["layout"]["nodes"].items():
            kws = set()
            if n.startswith("Aisle_"):
                k = int(n.split("_")[1])
                kws |= {f"aisle {k}", f"aisle {words[k]}"}
            kws |= {w for w in re.findall(r"[a-z]+", (nd.get("label") or "").lower()) if w not in ("and", "ready")}
            if n == "Entrance":
                kws |= {"entrance", "front door"}
            if n == "Checkout":
                kws |= {"checkout", "check out", "billing", "counter", "cashier"}
            out[n] = kws
        return out

    def _match_place(self, t):
        for n, kws in self.place_keywords().items():          # 'aisle two' style first (most specific)
            if any(re.search(rf"\b{re.escape(k)}\b", t) for k in kws if k.startswith("aisle")):
                return n
        for n, kws in self.place_keywords().items():
            if any(re.search(rf"\b{re.escape(k)}\b", t) for k in kws if not k.startswith("aisle")):
                return n
        return None

    @staticmethod
    def parse_command(t):
        """Pure text -> (command, arg). t is lower-case with punctuation removed. None if unknown."""
        has = lambda pat: re.search(pat, t) is not None
        if has(r"\bauto ?pilot\b"):
            return ("autopilot", not has(r"\b(off|stop|disable|cancel|manual)\b"))
        if has(r"\b(free ?roam|free mode|manual mode|i will drive|let me drive)\b"):
            return ("mode", "free")
        if has(r"\b(guided|guide me|follow the route)\b"):
            return ("mode", "guided")
        if has(r"\b(stop|halt|freeze|hold on|wait)\b"):
            return ("stop", None)
        if has(r"\b(help|commands|what can i say)\b"):
            return ("help", None)
        if has(r"\b(repeat|say that again|say again|pardon)\b"):
            return ("repeat", None)
        if has(r"\b(where am i|where are we|my options|which way|which ways|options|describe|directions)\b"):
            return ("where", None)
        if has(r"\b(browse|what s here|what is here|whats here|look around|what do you have|shelf contents)\b"):
            return ("browse", None)
        if has(r"\b(scan|check the shelf|identify)\b"):
            return ("scan", None)
        if has(r"\bskip\b"):
            return ("skip", None)
        if has(r"\b(cart|basket|total|how much)\b"):
            return ("cart", None)
        if has(r"\b(next item|what s next|whats next|what do i need|my list|shopping list|remaining|what is left)\b"):
            return ("list", None)
        if has(r"\b(turn around|u turn|go back|back|behind|reverse|backwards)\b"):
            return ("dir", "behind")
        if has(r"\bleft\b"):
            return ("dir", "left")
        if has(r"\bright\b"):
            return ("dir", "right")
        if has(r"\b(straight|ahead|forward|forwards|keep going|carry on|go on|continue)\b"):
            return ("dir", "ahead")
        m = re.search(r"\b(?:option|number|path|choice)\s+(\w+)\b", t) or re.fullmatch(r"\s*(\w+)\s*", t)
        if m and m.group(1) in SimEngine.NUMWORDS:
            return ("num", SimEngine.NUMWORDS[m.group(1)])
        for word, v in (("north", (0, -1)), ("up", (0, -1)), ("south", (0, 1)), ("down", (0, 1)),
                        ("west", (-1, 0)), ("east", (1, 0))):
            if has(rf"\b{word}\b"):
                return ("abs", v)
        if has(r"\b(go|walk|move|start|resume|proceed|continue)\b"):
            return ("go", None)
        return None

    def _voice_dir(self, word):
        if self.heading_to is not None:
            if word == "ahead":
                self.take_control()
                self.walking = True
            elif word == "behind":
                self.reverse()
            else:
                self.say("You are in an aisle. Say ahead, or turn around")
            return
        if word == "behind":
            self.reverse()
            return
        m = [o for o in self.options() if o["word"] == word]
        if len(m) == 1:
            self.go_to(m[0]["node"])
        elif not m:
            self.say(f"No path {'straight ahead' if word == 'ahead' else 'to the ' + word}. Say where am I for options")
        else:
            self.say(f"{len(m)} paths to the {word}. Say a number: "
                     + ", ".join(f"{o['key']} for {self._name(o['node'])}" for o in m))

    def handle_voice(self, text):
        """Run one spoken command. Returns (understood, label)."""
        t = re.sub(r"[^a-z0-9 ]", " ", text.lower().replace("'", " ")).strip()
        t = re.sub(r"\s+", " ", t)
        if not t:
            return False, "empty"
        cmd = self.parse_command(t)
        if cmd is None or cmd[0] in ("dir", "num", "go", "abs"):
            node = self._match_place(t)
            if node:
                cmd = ("place", node)
        if cmd is None:
            self.say("Sorry, I did not catch that. Say help for commands")
            return False, "unknown"
        c, a = cmd
        if c == "stop":
            self.stop()
        elif c == "place":
            self.goto_place(a)
        elif c == "dir":
            self._voice_dir(a)
        elif c == "num":
            self.choose_number(a)
        elif c == "abs":
            self.steer(*a)
        elif c == "go":
            self.take_control()
            self.walk()
            if self.heading_to is not None and self.walking:
                self.say("Walking on")
        elif c == "where":
            self.describe()
        elif c == "browse":
            self.browse()
        elif c == "scan":
            self.scan()
        elif c == "skip":
            self.skip_item()
        elif c == "autopilot":
            self.auto = bool(a)
            self.log("mode", f"autopilot {'on' if self.auto else 'off'} (voice)")
            self.say("Autopilot on" if self.auto else "Autopilot off. You are driving")
        elif c == "mode":
            self.auto, self.free = False, a == "free"
            self._hint()
            self.say("Free roam. You choose every direction" if self.free else "Guided mode. I follow the route")
        elif c == "repeat":
            if self.last_speech:
                self.speech_out.append((self.last_speech, "force"))
        elif c == "cart":
            n, total = len(self.cart), sum(x["actual"]["price_inr"] for x in self.cart)
            self.say(f"{n} item{'s' if n != 1 else ''} in your cart, {total:.0f} rupees total")
        elif c == "list":
            pend = [g for g in self.goals if g.kind == "item" and g.status == "pending"]
            if not pend:
                self.say("All list items handled. Say take me to checkout")
            else:
                first = pend[0]
                rest = ", ".join(g.sku["name"] for g in pend[1:4])
                self.say(f"Next: {first.sku['name']} in {self._name(first.node)}." + (f" Then {rest}." if rest else ""))
        elif c == "help":
            self.say("You can say: turn left, turn right, straight, go back, stop, go, "
                     "take me to dairy, aisle two, option three, scan, browse, where am I, "
                     "what is next, my cart, skip, or repeat", force=True)
        return True, f"{c}{'' if a is None else ' ' + str(a)}"

    def voice_phrases(self):
        """Closed vocabulary for the offline recogniser (much more accurate than open dictation)."""
        P = {"stop", "wait", "halt", "go", "walk", "move", "start", "resume", "continue", "keep going", "carry on",
             "help", "commands", "what can i say", "repeat", "say that again", "say again", "pardon",
             "where am i", "where are we", "my options", "which way", "options", "describe", "directions",
             "browse", "look around", "what do you have", "what is here", "whats here", "scan", "scan shelf",
             "check the shelf", "skip", "skip item", "my cart", "cart", "total", "how much", "next item",
             "whats next", "what is next", "what do i need", "my list", "shopping list", "what is left",
             "autopilot on", "autopilot off", "stop autopilot", "free roam", "free mode", "manual mode",
             "guided mode", "guide me", "i will drive", "let me drive"}
        for d in ("left", "right", "straight", "ahead", "forward", "back", "behind", "north", "south", "east", "west", "up", "down"):
            for pre in ("", "turn ", "go ", "take a ", "walk "):
                P.add(pre + d)
        P |= {"turn around", "u turn", "go back", "reverse"}
        for w in ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine"):
            for pre in ("", "option ", "number ", "path ", "choice "):
                P.add(pre + w)
        for kws in self.place_keywords().values():
            for k in kws:
                for pre in ("", "go to ", "take me to ", "head to ", "walk to ", "go to the ", "take me to the "):
                    P.add(pre + k)
        return sorted(P)

    def browse(self):
        """Walk the shelf with your hand: hear what is stocked here (no scan, no cart change)."""
        if self.heading_to is not None:
            self.say("Stop at an aisle to browse its shelf")
            return
        items = [x for x in self.data["skus"] if x["aisle"] == self.node]
        if not items:
            self.say(f"No shelves at {self._name(self.node)}")
            return
        bad = [x for x in items if self.avoid & set(x["allergens"])]
        out = [x for x in items if self.stock[x["sku_id"]] <= 0]
        names = ", ".join(x["name"] for x in items[:3])
        msg = f"{self._name(self.node)}: {len(items)} products, including {names}."
        if bad:
            msg += f" {len(bad)} contain allergens you avoid."
        if out:
            msg += f" {len(out)} out of stock."
        self.log("browse", f"{self.node}: {len(items)} products, {len(bad)} allergen, {len(out)} out")
        self.say(msg)

    def skip_item(self):
        if self.goal.kind == "item" and self.state != "COMPLETE":
            self.log("skip", self.goal.sku["name"])
            self._finish_goal("skipped")

    def scan(self):
        g = self.goal
        if self.state != "AT_SHELF" or g.kind != "item":
            self.say(f"Not at the target shelf yet. Keep walking to {g.node}")
            return
        g.attempts += 1
        self.m["scans"] += 1
        res = self._simulate_scan(g)
        self.last_scan = res
        self.scan_log.append({k: v for k, v in res.items() if k not in ("boxes",)})
        self.snap_flags.add("scan")
        tgt = g.sku
        oc = res["outcome"]
        self.log("scan", f"{tgt['name']}: {oc} sim={res['sim']:.2f} (try {g.attempts})")
        last_try = g.attempts >= self.data["retrieval"]["max_scan_attempts"]

        if oc == "VERIFIED":
            self.m["verified"] += 1
            actual = self.data["skus"][res["true_idx"]]
            if res["true_idx"] != res["target_idx"]:
                self.m["misread_picks"] += 1
            self.stock[actual["sku_id"]] -= 1
            self.cart.append({"requested": tgt, "actual": actual, "ok": res["true_idx"] == res["target_idx"]})
            meta = dict(tgt, price_inr=int(tgt["price_inr"]) if float(tgt["price_inr"]).is_integer() else tgt["price_inr"])
            self.say("Reach forward 40 centimeters at chest level", sku_meta=meta, novice=False)
            self._finish_goal("found" if res["true_idx"] == res["target_idx"] else "wrong")
        elif oc == "ALLERGEN_BLOCK":
            self.m["allergen_blocks"] += 1
            bad = sorted(self.avoid & set(tgt["allergens"]))
            alt = self._alternative(tgt)
            tail = f" Try {alt['name']} instead" if alt else ""
            self.say(f"Warning. {tgt['name']} contains {', '.join(bad)}, which you avoid. Do not pick it up.{tail}",
                     urgent=True)
            self._finish_goal("blocked")
        else:
            self.m["wrong_item" if oc == "WRONG_ITEM" else "low_conf"] += 1
            if last_try:
                self.m["missed"] += 1
                alt = self._alternative(tgt)
                tail = f" Closest alternative is {alt['name']}" if alt else ""
                self.say(f"I cannot find {tgt['name']}. It may be out of stock. Moving on.{tail}")
                self._finish_goal("missed")
            elif oc == "WRONG_ITEM":
                other = self.data["skus"][res["j_star"]]["name"]
                self.say(f"This looks like {other}, not {tgt['name']}. Sweep your hand along the shelf and scan again")
            else:
                self.say("Not confident about that product. Hold steady and scan again")

    def _alternative(self, tgt):
        cands = [s for s in self.data["skus"]
                 if s["aisle"] == tgt["aisle"] and s["macro_category"] == tgt["macro_category"]
                 and s["sku_id"] != tgt["sku_id"] and self.stock[s["sku_id"]] > 0
                 and not (self.avoid & set(s["allergens"]))]
        return min(cands, key=lambda s: s["price_inr"]) if cands else None

    def _simulate_scan(self, g: Goal) -> dict:
        cat, L = self.cat, self.data["lighting"][self.lighting]
        tgt_idx = cat.index[g.sku["sku_id"]]
        in_stock = self.stock[g.sku["sku_id"]] > 0
        pool = [i for i, s in enumerate(cat.skus)
                if s["aisle"] == g.node and self.stock[s["sku_id"]] > 0 and i != tgt_idx]
        n = int(self.rng.integers(4, 7)) - (1 if in_stock else 0)
        chosen = [int(x) for x in self.rng.choice(pool, size=min(len(pool), n), replace=False)] if pool else []
        if in_stock:
            chosen.append(tgt_idx)
        self.rng.shuffle(chosen)

        boxes = np.array([[60 + 110 * i, 140, 160 + 110 * i, 310] for i in range(len(chosen))], dtype=np.float32)
        conf = np.clip(self.rng.normal(L["det_conf"], 0.025, len(chosen)), 0.5, 0.99).astype(np.float32)
        z = cat.Z[chosen].copy()
        for r, idx in enumerate(chosen):   # look-alike packaging: drift toward a sibling SKU
            sibs = [j for j in cat.family[cat.skus[idx]["family"]] if j != idx]
            if sibs and self.rng.random() < L.get("lookalike_p", 0.0):
                a = float(self.rng.uniform(0.55, 0.75))
                z[r] = (1 - a) * z[r] + a * cat.Z[int(self.rng.choice(sibs))]
        z = z + L["embed_noise"] * self.rng.standard_normal((len(chosen), cat.Z.shape[1]))
        z = (z / np.linalg.norm(z, axis=-1, keepdims=True)).astype(np.float32)

        z_t = self.be.refine(z, boxes)
        sims = z_t @ cat.Z.T
        best_j, s_sim = sims.argmax(-1), sims.max(-1)
        # GoG relevance k_i from the *predicted* SKU only: goal match > same category > unrelated
        tgt_macro = g.sku["macro_category"]
        k_gog = np.array([1.0 if j == tgt_idx else 0.4 if cat.skus[j]["macro_category"] == tgt_macro else 0.1
                          for j in best_j], dtype=np.float32)
        q_sp = np.clip(1 - np.abs((boxes[:, 0] + boxes[:, 2]) / 2 - 385) / 385, 0, 1).astype(np.float32)
        prio = np.asarray(self.be.priority(conf, s_sim.astype(np.float32), q_sp, k_gog), dtype=np.float32)
        j_star, sim, ok = self.be.retrieve(z_t, cat.Z, prio)
        i_star = int(np.argmax(prio))
        true_idx = int(chosen[i_star])

        if not ok:
            outcome = "LOW_CONFIDENCE"
        elif j_star != tgt_idx:
            outcome = "WRONG_ITEM"
        elif self.avoid & set(g.sku["allergens"]):
            outcome = "ALLERGEN_BLOCK"
        else:
            outcome = "VERIFIED"
        return dict(attempt=g.attempts, aisle=g.node, target=g.sku["sku_id"], target_idx=tgt_idx,
                    target_in_stock=in_stock, outcome=outcome, sim=float(sim), i_star=i_star,
                    j_star=j_star if j_star is not None else int(best_j[i_star]),
                    true_idx=true_idx, boxes=boxes.tolist(), conf=conf.tolist(),
                    s_sim=s_sim.tolist(), prio=prio.tolist(), pred=[int(j) for j in best_j],
                    truth=[int(c) for c in chosen], lighting=self.lighting)

    # ---- per-frame update --------------------------------------------------
    def update(self, dt):
        if self.state == "COMPLETE":
            return
        self.t += dt
        self._tick_obstacles(dt)
        if self.chaos and self.state != "AT_SHELF":
            self.chaos_timer -= dt
            if self.chaos_timer <= 0:
                self.chaos_timer = float(self.rng.uniform(8, 16))
                self.spawn_obstacle(ttl=float(self.rng.uniform(8, 14)), moving=True)
        self._sense()
        self._hazard_logic()
        if self.replan_pending and self.manual:
            self.replan_pending = False
            self._hint()
            self.say("Path reopened")
        elif self.replan_pending and (self.no_route or (self.heading_to is None and not self.retreating)):
            self.replan_pending = False
            was_stuck = self.no_route
            self._replan()
            if was_stuck and not self.no_route:
                self.say("Path is clear. Resuming")
        if self.manual and self.travel and self.heading_to is None:
            if self.node == self.travel:
                self.travel = None
            else:
                r, c = self._route(self.node, self.travel)
                if len(r) < 2 or c >= 1e5 or frozenset((r[0], r[1])) in self.blocked:
                    self.say("No clear way there. Choose a path")
                    self.travel = None
                else:
                    self.go_to(r[1], _auto=True)
        self._refresh_state()
        if self.auto:
            if self.state == "AT_SHELF":
                self.auto_timer += dt
                if self.auto_timer >= 1.2:
                    self.auto_timer = 0.0
                    self.scan()
            elif self.state == "NAVIGATING":
                self.walk()
        self._move(dt)
        self._refresh_state()

    # ---- reporting ---------------------------------------------------------
    def summary(self):
        items = [g for g in self.goals if g.kind == "item"]
        total = sum(c["actual"]["price_inr"] for c in self.cart)
        return dict(list=self.list_name, profile=self.profile_name, lighting=self.lighting, seed=self.seed,
                    state=self.state, sim_seconds=round(self.t, 1),
                    items=[{"sku": g.sku["sku_id"], "name": g.sku["name"], "status": g.status,
                            "attempts": g.attempts} for g in items],
                    cart_total_inr=total, cart=[{"requested": c["requested"]["sku_id"],
                                                  "actual": c["actual"]["sku_id"], "ok": c["ok"]} for c in self.cart],
                    metrics={k: (round(v, 1) if isinstance(v, float) else v) for k, v in self.m.items()},
                    torch_backend=self.be.have_torch)

    def export_log(self):
        return {"summary": self.summary(), "scans": self.scan_log, "events": self.events}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
W, H = 1200, 800
HDR, MAP_W, MAP_H = 36, 780, 600
FONT = cv2.FONT_HERSHEY_SIMPLEX
BG, PANEL, GRID = (28, 28, 32), (40, 40, 46), (44, 44, 50)
TXT, DIM = (236, 236, 236), (150, 150, 158)
OK, WARN, BAD, INFO, ROUTE = (120, 220, 110), (0, 200, 255), (70, 70, 240), (250, 200, 70), (255, 215, 70)
CAT_COLORS = [(180, 120, 60), (90, 160, 90), (70, 130, 200), (160, 90, 160), (60, 170, 180),
              (120, 120, 200), (170, 150, 70), (90, 90, 170), (60, 150, 120), (150, 110, 110)]
STATE_COL = {"NAVIGATING": OK, "HAZARD_OVERRIDE": BAD, "AT_SHELF": WARN, "COMPLETE": INFO}
STATUS_COL = {"pending": DIM, "found": OK, "wrong": WARN, "missed": BAD, "blocked": BAD, "skipped": DIM}


def P(p):
    return (int(round(p[0])), int(round(p[1])))


def put(img, s, org, scale=0.45, col=TXT, th=1):
    cv2.putText(img, ascii_safe(s), P(org), FONT, scale, col, th, cv2.LINE_AA)


def trunc(s, n):
    s = ascii_safe(s)
    return s if len(s) <= n else s[: n - 1] + "."


def wrap(text, width_px, scale=0.5, th=1):
    lines, cur = [], ""
    for w in ascii_safe(text).split():
        trial = (cur + " " + w).strip()
        if cv2.getTextSize(trial, FONT, scale, th)[0][0] <= width_px:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    return lines + ([cur] if cur else [])


def dashed(img, p1, p2, col, th=2, dash=10):
    p1, p2 = np.array(p1, float), np.array(p2, float)
    n = max(1, int(np.linalg.norm(p2 - p1) // dash))
    for i in range(0, n, 2):
        a = p1 + (p2 - p1) * i / n
        b = p1 + (p2 - p1) * min(i + 1, n) / n
        cv2.line(img, P(a), P(b), col, th, cv2.LINE_AA)


def cat_color(cat):
    return CAT_COLORS[zlib.crc32(cat.encode()) % len(CAT_COLORS)]


class Renderer:
    def __init__(self, engine: SimEngine):
        self.e = engine

    def draw(self, ui) -> np.ndarray:
        e = self.e
        f = np.full((H, W, 3), BG, np.uint8)
        cv2.rectangle(f, (0, 0), (W, HDR), (50, 50, 58), -1)
        put(f, "SAARATHI AI  |  Live Supermarket Simulation", (14, 24), 0.6, TXT, 2)
        mode = "AUTOPILOT" if e.auto else ("FREE-ROAM" if e.free else "GUIDED")
        chaos = "  CHAOS ON" if e.chaos else ""
        put(f, f"{mode}{chaos}  |  light: {e.lighting}  |  profile: {e.profile_name}", (560, 24), 0.45, DIM)
        f[HDR:HDR + MAP_H, :MAP_W] = self._map(ui)
        self._right_panel(f, MAP_W, HDR)
        self._bottom(f, HDR + MAP_H, ui)
        return f

    # ---- store map ---------------------------------------------------------
    def _map(self, ui=None):
        e = self.e
        ui = ui or {}
        m = np.full((MAP_H, MAP_W, 3), (34, 34, 38), np.uint8)
        for x in range(0, MAP_W, 40):
            cv2.line(m, (x, 0), (x, MAP_H), GRID, 1)
        for y in range(0, MAP_H, 40):
            cv2.line(m, (0, y), (MAP_W, y), GRID, 1)
        nodes = e.data["layout"]["nodes"]
        for name, nd in nodes.items():
            if nd["label"]:
                x, y = nd["xy"]
                col = cat_color(nd["label"])
                cv2.rectangle(m, (x - 66, y - 50), (x + 66, y - 24), tuple(int(c * 0.45) for c in col), -1)
                cv2.rectangle(m, (x - 66, y - 50), (x + 66, y - 24), col, 1)
                put(m, nd["label"], (x - 60, y - 32), 0.4, TXT)
        for u, v, d in e.nav.graph.edges(data=True):
            p1, p2 = e.coords[u], e.coords[v]
            if d["weight"] > 900:
                dashed(m, p1, p2, BAD, 3)
            else:
                cv2.line(m, p1, p2, (105, 105, 112), 2, cv2.LINE_AA)
            mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            put(m, f"{e.base_w[frozenset((u, v))]:.1f}m", (mid[0] + 4, mid[1] - 4), 0.36, DIM)
        # planned route (a dashed hint in free-roam, solid arrows when guided)
        for a, b in zip(e.route, e.route[1:]):
            if e.manual:
                dashed(m, e.coords[a], e.coords[b], (90, 160, 190), 2, 14)
            else:
                cv2.arrowedLine(m, e.coords[a], e.coords[b], ROUTE, 3, cv2.LINE_AA, tipLength=0.06)
        # direction choices (free-roam)
        opts = e.options() if (e.manual and e.state != "COMPLETE") else []
        ux0, uy0 = e.user_xy()
        hover = ui.get("hover")
        for o in opts:
            tp = e.coords[o["node"]]
            col = BAD if o["blocked"] else (255, 255, 120) if o["node"] == hover else (200, 230, 90)
            cv2.line(m, P((ux0, uy0)), tp, col, 3, cv2.LINE_AA)
        for name, p in e.coords.items():
            is_dest = name == e.goal.node and e.state != "COMPLETE"
            cv2.circle(m, p, 13, OK if is_dest else (185, 185, 190), -1, cv2.LINE_AA)
            put(m, name, (p[0] - 28, p[1] + 30), 0.42, TXT)
        for o in opts:                      # numbered chips on each reachable node
            tp = e.coords[o["node"]]
            col = BAD if o["blocked"] else (255, 255, 120) if o["node"] == hover else (200, 230, 90)
            cv2.circle(m, tp, 20, col, 2, cv2.LINE_AA)
            cx, cy = tp[0] + 18, tp[1] - 20
            cv2.circle(m, (cx, cy), 10, col, -1, cv2.LINE_AA)
            put(m, str(o["key"]), (cx - 5, cy + 5), 0.45, (20, 20, 20), 2)
        # sensor wedge
        hv = e.heading_vec()
        ux, uy = e.user_xy()
        if hv and e.state not in ("COMPLETE", "AT_SHELF"):
            fused = e.sens["fused"]
            col = BAD if fused < e.fusion.tau_obs else WARN if fused < 2.4 else OK
            ln, ang = 110, math.atan2(hv[1], hv[0])
            pts = np.array([[ux, uy], [ux + ln * math.cos(ang - 0.35), uy + ln * math.sin(ang - 0.35)],
                            [ux + ln * math.cos(ang + 0.35), uy + ln * math.sin(ang + 0.35)]], np.int32)
            ov = m.copy()
            cv2.fillPoly(ov, [pts], col)
            m = cv2.addWeighted(ov, 0.25, m, 0.75, 0)
        # obstacles
        for ob in e.obstacles:
            (x0, y0), (x1, y1) = e.coords[ob.a], e.coords[ob.b]
            c = (x0 + (x1 - x0) * ob.t, y0 + (y1 - y0) * ob.t)
            cv2.circle(m, P(c), int(ob.w_real * 26), (40, 140, 255), -1, cv2.LINE_AA)
            cv2.circle(m, P(c), int(ob.w_real * 26) + 3, TXT, 1, cv2.LINE_AA)
            put(m, ob.label, (c[0] - 30, c[1] - 22), 0.38, (40, 160, 255))
        ob, d = e.sens["ob"], e.sens["d_true"]
        if ob is not None:
            (x0, y0), (x1, y1) = e.coords[ob.a], e.coords[ob.b]
            c = (x0 + (x1 - x0) * ob.t, y0 + (y1 - y0) * ob.t)
            cv2.line(m, P((ux, uy)), P(c), (40, 160, 255), 1, cv2.LINE_AA)
        # user
        cv2.circle(m, P((ux, uy)), 10, (255, 120, 20), -1, cv2.LINE_AA)
        cv2.circle(m, P((ux, uy)), 16, (255, 255, 255), 2, cv2.LINE_AA)
        if hv:
            cv2.arrowedLine(m, P((ux, uy)), P((ux + hv[0] * 26, uy + hv[1] * 26)), (255, 255, 255), 2,
                            cv2.LINE_AA, tipLength=0.5)
        put(m, "USER", (ux + 20, uy - 12), 0.45, (255, 140, 40), 2)
        v = ui.get("voice")
        if v is not None:                   # microphone status, bottom-left of the map
            live = v.backend != "off" and v.enabled
            cv2.circle(m, (18, MAP_H - 16), 6, OK if live else BAD, -1, cv2.LINE_AA)
            state = "listening" if live else ("muted [V]" if v.backend != "off" else "off")
            put(m, f"MIC ({v.backend}): {state}", (30, MAP_H - 12), 0.4, OK if live else DIM)
            if v.last_heard and time.time() - v.last_t < 6.0:
                put(m, f"heard: \"{trunc(v.last_heard, 40)}\"", (30, MAP_H - 32), 0.45, INFO)
        if opts:                            # choice box, top-left of the map
            title = "CHOOSE A WAY  (arrows / 1-9 / click)"
            lines = []
            for o in opts:
                tag = " BLOCKED" if o["blocked"] else ""
                lines.append((f"[{o['key']}] {o['word']} -> {e._name(o['node'])}  {o['dist']:.1f}m{tag}",
                              BAD if o["blocked"] else TXT))
            bh = 22 + 17 * len(lines) + 6
            cv2.rectangle(m, (8, 8), (372, 8 + bh), (24, 24, 28), -1)
            cv2.rectangle(m, (8, 8), (372, 8 + bh), (110, 110, 118), 1)
            put(m, title, (16, 25), 0.42, INFO)
            for i, (t, c) in enumerate(lines):
                put(m, t, (16, 44 + 17 * i), 0.4, c)
        return m

    # ---- right panel -------------------------------------------------------
    def _right_panel(self, f, x0, y0):
        e = self.e
        cv2.rectangle(f, (x0, y0), (W, y0 + MAP_H), PANEL, -1)
        x, y = x0 + 14, y0 + 20
        put(f, f"SHOPPING LIST - {e.list_name}", (x, y), 0.48, INFO, 1)
        y += 8
        items = [g for g in e.goals if g.kind == "item"]
        for g in items[:7]:
            y += 22
            cur = g is e.goal and e.state != "COMPLETE"
            cv2.circle(f, (x + 6, y - 4), 5, INFO if cur else STATUS_COL[g.status], -1 if (cur or g.status != "pending") else 1)
            put(f, trunc(g.sku["name"], 27), (x + 20, y), 0.42, TXT if cur else STATUS_COL[g.status])
            put(f, g.node, (x + 236, y), 0.38, DIM)
            put(f, f"Rs.{g.sku['price_inr']:.0f}", (x + 292, y), 0.4, DIM)
            if g.status != "pending":
                put(f, g.status.upper(), (x + 346, y), 0.33, STATUS_COL[g.status])
        total = sum(c["actual"]["price_inr"] for c in e.cart)
        y += 24
        n = len(e.cart)
        put(f, f"Cart: {n} item{'s' if n != 1 else ''}   Total: Rs.{total:.0f}", (x, y), 0.47, OK)

        # GoG plan
        y += 22
        put(f, "GoG PLAN", (x, y), 0.44, INFO)
        if e.goal.kind == "item":
            idx = 0 if e.state != "AT_SHELF" else (2 if e.goal.attempts else 1)
        else:
            idx = 0
        for i, step in enumerate(e.chain[:4]):
            y += 17
            col = OK if i < idx else (TXT if i == idx else DIM)
            put(f, f"{i + 1}. {trunc(step, 52)}", (x + 4, y), 0.38, col)

        # sensors
        y += 26
        put(f, f"SENSOR FUSION   tau_obs = {e.fusion.tau_obs} m", (x, y), 0.44, INFO)
        s = e.sens
        for label, v in (("RGB-D", s["rgbd"]), ("ToF", s["tof"]), ("Mono", min(s["tri"], 6.0)), ("FUSED", s["fused"])):
            y += 19
            put(f, label, (x, y), 0.38, TXT)
            bx, bw = x + 58, 250
            cv2.rectangle(f, (bx, y - 11), (bx + bw, y - 2), (60, 60, 66), -1)
            col = BAD if v < e.fusion.tau_obs else WARN if v < 2.4 else OK
            cv2.rectangle(f, (bx, y - 11), (bx + int(bw * min(v, 6.0) / 6.0), y - 2), col, -1)
            tx = bx + int(bw * e.fusion.tau_obs / 6.0)
            cv2.line(f, (tx, y - 14), (tx, y + 1), (255, 255, 255), 1)
            put(f, f"{v:.2f} m" if v < 5.99 else ">6 m", (bx + bw + 8, y), 0.38, col)

        # shelf camera
        y += 22
        put(f, "SHELF CAMERA / CAQE", (x, y), 0.44, INFO)
        self._shelf_cam(f, x, y + 6, 390, 118)

    def _shelf_cam(self, f, x, y, w, h):
        e, r = self.e, self.e.last_scan
        cv2.rectangle(f, (x, y), (x + w, y + h), (30, 30, 34), -1)
        cv2.rectangle(f, (x, y), (x + w, y + h), (90, 90, 96), 1)
        if not r:
            put(f, "idle - press S at the target aisle", (x + 70, y + 60), 0.42, DIM)
            return
        sx = w / 770.0
        for i, b in enumerate(r["boxes"]):
            x1, x2 = x + int(b[0] * sx), x + int(b[2] * sx)
            col = cat_color(e.cat.skus[r["pred"][i]]["macro_category"])
            cv2.rectangle(f, (x1, y + 12), (x2, y + 78), tuple(int(c * 0.55) for c in col), -1)
            chosen = i == r["i_star"]
            cv2.rectangle(f, (x1, y + 12), (x2, y + 78), (255, 255, 255) if chosen else col, 2 if chosen else 1)
            put(f, trunc(e.cat.skus[r["pred"][i]]["name"], 9), (x1 + 2, y + 28), 0.3, TXT)
            put(f, f"s={r['s_sim'][i]:.2f}", (x1 + 2, y + 92), 0.33, TXT)
            put(f, f"R={r['prio'][i]:.2f}", (x1 + 2, y + 106), 0.33, DIM)
            if r["pred"][i] != r["truth"][i]:
                cv2.line(f, (x2 - 12, y + 14), (x2 - 3, y + 24), BAD, 2)
                cv2.line(f, (x2 - 3, y + 14), (x2 - 12, y + 24), BAD, 2)
        col = {"VERIFIED": OK, "ALLERGEN_BLOCK": BAD}.get(r["outcome"], WARN)
        put(f, f"{r['outcome']}  sim={r['sim']:.2f} (tau {e.data['retrieval']['tau_sim']})  "
               f"try {r['attempt']}/{e.data['retrieval']['max_scan_attempts']}", (x, y + h + 18), 0.42, col)
        put(f, "white box = top priority   red X = SKU misread (ground truth)", (x, y + h + 34), 0.33, DIM)

    # ---- bottom panel ------------------------------------------------------
    def _bottom(self, f, y0, ui):
        e = self.e
        cv2.rectangle(f, (0, y0), (W, H), (46, 46, 52), -1)
        cv2.line(f, (0, y0), (W, y0), (110, 110, 118), 1)
        col = STATE_COL[e.state]
        put(f, f"System State: {e.state}", (16, y0 + 24), 0.62, col, 2)
        goal = e.goal
        tgt = goal.sku["name"] if goal.kind == "item" else "Checkout"
        put(f, f"Target: {tgt} @ {goal.node}   |   Obstacle (fused): {e.sens['fused']:.2f} m   "
               f"|   {'walking' if e.walking or e.retreating else 'standing'}", (16, y0 + 48), 0.45, (205, 205, 205))
        put(f, "Spoken audio output:", (16, y0 + 74), 0.42, DIM)
        for i, line in enumerate(wrap(e.last_speech or "...", 500, 0.5)[:4]):
            put(f, line, (16, y0 + 96 + i * 21), 0.5, TXT)
        # event log
        lx = 540
        put(f, "EVENT LOG", (lx, y0 + 22), 0.44, INFO)
        for i, ev in enumerate(e.events[-7:]):
            c = BAD if ev["type"] in ("hazard",) else WARN if ev["type"] in ("reroute", "scan") else DIM
            put(f, f"{ev['t']:6.1f}s {trunc(ev['type'], 8):8s} {trunc(ev['msg'], 44)}", (lx, y0 + 44 + i * 17), 0.36, c)
        # stats + keys
        rx = 940
        m = e.m
        put(f, "SESSION", (rx, y0 + 22), 0.44, INFO)
        put(f, f"t={e.t:5.1f}s  walked {m['dist_m']:.1f} m", (rx, y0 + 42), 0.38, TXT)
        put(f, f"hazards {m['hazards']}  reroutes {m['reroutes']}", (rx, y0 + 58), 0.38, TXT)
        put(f, f"scans {m['scans']}  ok {m['verified']}  wrong {m['wrong_item']}  low {m['low_conf']}", (rx, y0 + 74), 0.38, TXT)
        keys = ("Arrows/1-9/click: choose  [Space] stop-go",
                "[W] ahead [U] u-turn [B] browse [C] where?",
                "[S] scan  [N] skip  [O] obstacle  [X] chaos",
                "[F] free/guided [A] auto [G] light [V] mic",
                "[P] profile  [L] list  [R] reset  [M] mute  [Q] quit")
        if ui.get("help", True):
            for i, k in enumerate(keys):
                put(f, k, (rx - 10, y0 + 92 + i * 14), 0.32, DIM)
        if e.state == "COMPLETE":
            put(f, "TRIP COMPLETE - press R to restart", (rx - 10, y0 + 158), 0.5, INFO, 1)


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------
def write_log(engine, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(engine.export_log(), fh, indent=2)
    print(f"[Log] session written to {path}")


def print_summary(engine):
    s = engine.summary()
    print("\n--- Session summary ---")
    print(f"state={s['state']}  time={s['sim_seconds']}s  cart=Rs.{s['cart_total_inr']:.0f}  torch={s['torch_backend']}")
    for it in s["items"]:
        print(f"  {it['status']:8s} {it['name']}  (scans: {it['attempts']})")
    print("  metrics:", s["metrics"])


def run_headless(engine, args):
    engine.auto = True
    renderer = Renderer(engine)
    dt, saved = 1 / 30.0, set()
    while engine.t < args.max_seconds and engine.state != "COMPLETE":
        engine.update(dt)
        engine.speech_out.clear()
        if args.snapshots:
            for flag in list(engine.snap_flags - saved):
                os.makedirs(args.snapshots, exist_ok=True)
                cv2.imwrite(os.path.join(args.snapshots, f"{flag}.png"), renderer.draw({"help": True}))
                saved.add(flag)
    if args.snapshots and engine.state != "COMPLETE":
        cv2.imwrite(os.path.join(args.snapshots, "timeout.png"), renderer.draw({"help": True}))
    print_summary(engine)
    if not args.no_log:
        write_log(engine, args.log or os.path.join(ROOT, "logs", f"sim_session_{int(time.time())}.json"))
    return 0 if engine.state == "COMPLETE" else 1


ARROWS = {  # waitKeyEx codes differ per OS / GUI backend
    (0, -1): {2490368, 65362, 63232, 16777235},   # up
    (0, 1): {2621440, 65364, 63233, 16777237},    # down
    (-1, 0): {2424832, 65361, 63234, 16777234},   # left
    (1, 0): {2555904, 65363, 63235, 16777236},    # right
}


def run_interactive(engine, args):
    speaker = Speaker(enabled=not args.mute)
    muted = args.mute
    renderer = Renderer(engine)
    ui = {"help": True}
    listener = None
    if not args.no_voice:
        listener = VoiceListener(engine.voice_phrases(), speaker, args.voice_backend, args.vosk_model,
                                 args.mic_device)
        ui["voice"] = listener
    print("\n=======================================================")
    print(" Saarathi AI - Enhanced Visual & Audio Simulator (v2)")
    print("=======================================================")
    print(f" repo modules: {'real (torch)' if HAVE_TORCH else 'numpy fallbacks for GoG/CAQE/XXAI (' + str(_TORCH_ERR) + ')'}")
    print(f" speech backend: {speaker.backend}")
    print(" Keys (click the window first):")
    print(" FREE-ROAM (default): you choose the direction at every junction")
    print("   Arrow keys / 1-9 / mouse click : pick a path     [Space] stop / go     [U] u-turn")
    print("   [W] keep walking straight   [B] browse shelf   [C] where am I?   [F] free <-> guided")
    print(" Shopping / testing:")
    print("   [S] scan shelf   [N] skip item   [O] obstacle   [X] chaos   [A] autopilot")
    print("   [G] lighting   [P] profile   [L] list   [R] reset   [M] mute   [H] help   [Q] quit")
    if listener is None:
        print(" voice input: disabled (--no-voice)")
    elif listener.backend == "off":
        print(" voice input: OFF - " + "; ".join(listener.errors))
        print("   to enable offline voice: pip install vosk sounddevice, then download a small model from")
        print("   https://alphacephei.com/vosk/models and unzip it into ./models/  (or --vosk-model PATH)")
    else:
        print(f" voice input: ON ({listener.backend}). Say 'help' for commands, [V] to mute the mic. Headphones recommended.")
    print("=======================================================\n")
    engine.say("Saarathi AI initialized. Ready for shopping navigation")
    last_n = 0
    win = "Saarathi AI - Live Supermarket Simulation"
    clicks = []

    def nearest_node(x, y):
        y -= HDR
        if not (0 <= x < MAP_W and 0 <= y < MAP_H):
            return None
        n, d = min(((n, math.hypot(x - c[0], y - c[1])) for n, c in engine.coords.items()), key=lambda t: t[1])
        return n if d <= 28 else None

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            ui["hover"] = nearest_node(x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            clicks.append(nearest_node(x, y))

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        engine.update(0.04)
        if listener:
            while True:
                heard = listener.poll()
                if heard is None:
                    break
                ok, label = engine.handle_voice(heard)
                print(f"[Voice] \"{heard}\" -> {label if ok else 'not understood'}")
        for text, urgent in engine.speech_out:
            speaker.say(text, bool(urgent), force=(urgent == "force"))
        engine.speech_out.clear()
        for ev in engine.events[last_n:]:
            print(f"[{ev['t']:6.1f}s {ev['type']}] {ev['msg']}")
        last_n = len(engine.events)
        cv2.imshow(win, renderer.draw(ui))
        raw = cv2.waitKeyEx(40)
        while clicks:
            node = clicks.pop(0)
            if node:
                engine.choose_node(node)
        steered = False
        for (dx, dy), codes in ARROWS.items():
            if raw in codes:
                engine.steer(dx, dy)
                steered = True
        key = raw if 0 <= raw < 256 else 255
        if steered:
            key = 255
        if ord("A") <= key <= ord("Z"):
            key += 32          # caps lock
        if key == ord("q"):
            break
        elif key == ord(" "):
            engine.toggle_walk()
        elif key == ord("u"):
            engine.reverse()
        elif key == ord("b"):
            engine.browse()
        elif key == ord("c"):
            engine.describe()
        elif key == ord("f"):
            engine.auto = False
            engine.free = not engine.free
            engine.log("mode", "free-roam" if engine.free else "guided (follows route)")
            engine._hint()
        elif ord("1") <= key <= ord("9"):
            engine.choose_number(key - ord("0"))
        elif key == ord("w"):
            engine.walk()
        elif key == ord("a"):
            engine.auto = not engine.auto
            engine.log("mode", f"autopilot {'on' if engine.auto else 'off'}")
        elif key == ord("o"):
            engine.toggle_obstacle()
        elif key == ord("s"):
            engine.scan()
        elif key == ord("n"):
            engine.skip_item()
        elif key == ord("x"):
            engine.chaos = not engine.chaos
            engine.log("mode", f"chaos {'on' if engine.chaos else 'off'}")
        elif key == ord("g"):
            names = list(engine.data["lighting"])
            engine.lighting = names[(names.index(engine.lighting) + 1) % len(names)]
            engine.log("mode", f"lighting {engine.lighting}")
        elif key in (ord("p"), ord("l"), ord("r")):
            if key == ord("p"):
                names = engine.profile_names
                engine.profile_name = names[(names.index(engine.profile_name) + 1) % len(names)]
            elif key == ord("l"):
                names = engine.list_names
                engine.list_name = names[(names.index(engine.list_name) + 1) % len(names)]
            engine.reset()
            last_n = 0
        elif key == ord("m"):
            muted = not muted
            speaker.muted = muted
        elif key == ord("v") and listener:
            listener.enabled = not listener.enabled
            engine.log("mode", f"mic {'on' if listener.enabled else 'muted'}")
        elif key == ord("h"):
            ui["help"] = not ui["help"]
    cv2.destroyAllWindows()
    print_summary(engine)
    if not args.no_log:
        write_log(engine, args.log or os.path.join(ROOT, "logs", f"sim_session_{int(time.time())}.json"))


def main():
    ap = argparse.ArgumentParser(description="Saarathi AI interactive supermarket simulator")
    ap.add_argument("--data", default=DATA_PATH)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--list", dest="list_name", default=None, help="shopping list name")
    ap.add_argument("--profile", default="Default")
    ap.add_argument("--lighting", default="normal", choices=["bright", "normal", "dim"])
    ap.add_argument("--auto", action="store_true", help="start in autopilot")
    ap.add_argument("--guided", action="store_true", help="old behaviour: follow the planned route (W only)")
    ap.add_argument("--chaos", action="store_true", help="random moving obstacles")
    ap.add_argument("--mute", action="store_true", help="silence spoken output")
    ap.add_argument("--no-voice", action="store_true", help="disable voice input")
    ap.add_argument("--voice-backend", default="auto", choices=["auto", "vosk", "sr"])
    ap.add_argument("--vosk-model", default=None, help="path to an unzipped Vosk model folder")
    ap.add_argument("--mic-device", type=int, default=None, help="microphone device index")
    ap.add_argument("--list-mics", action="store_true", help="print audio input devices and exit")
    ap.add_argument("--headless", action="store_true", help="no window; run a full autopilot trip")
    ap.add_argument("--max-seconds", type=float, default=600.0)
    ap.add_argument("--snapshots", default=None, help="headless: save PNGs on hazard/scan/complete")
    ap.add_argument("--log", default=None)
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()
    if args.list_mics:
        try:
            import sounddevice as sd
            print(sd.query_devices())
        except Exception as exc:
            print(f"sounddevice not available: {exc}")
        return

    with open(args.data) as fh:
        data = json.load(fh)
    sc, rt = data["sensors"], data["retrieval"]
    backend = Backend(sc["tau_obs"], rt["tau_sim"], rt["caqe_radius"])
    engine = SimEngine(data, backend, seed=args.seed, list_name=args.list_name, profile_name=args.profile,
                       lighting=args.lighting, chaos=args.chaos, auto=args.auto,
                       free=not (args.guided or args.auto or args.headless))
    if args.headless:
        sys.exit(run_headless(engine, args))
    run_interactive(engine, args)


if __name__ == "__main__":
    main()