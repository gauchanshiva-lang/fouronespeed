import { useEffect, useRef, useState, useCallback } from "react";

// ============================================================
// MediaPipe Hands loader (CDN). Loaded lazily so SSR is safe.
// ============================================================
const MP_HANDS_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js";
const MP_BASE = "https://cdn.jsdelivr.net/npm/@mediapipe/hands/";

declare global {
  interface Window {
    Hands: any;
  }
}

function loadScript(src: string): Promise<void> {
  return new Promise((resolve, reject) => {
    if (document.querySelector(`script[src="${src}"]`)) return resolve();
    const s = document.createElement("script");
    s.src = src;
    s.crossOrigin = "anonymous";
    s.onload = () => resolve();
    s.onerror = () => reject(new Error("Failed to load " + src));
    document.head.appendChild(s);
  });
}

// ============================================================
// Game phases
// ============================================================
type Phase = "idle" | "countdown" | "playing" | "finished";

const GAME_DURATION = 20; // seconds
const REP_COOLDOWN_MS = 70;     // ~allows up to ~14 reps/sec — pure jitter guard
const MIN_MOVE_THRESHOLD = 0.012; // tiny movement still counts (was 0.025)
const DEAD_ZONE = 0.004;          // hands considered "level" only when nearly identical (was 0.015)
const SMOOTHING = 0.85;           // very responsive (was 0.55)
const HAND_LOST_MS = 200;         // drop hand quickly when not seen

interface HandPoint {
  x: number;
  y: number;
  visible: boolean;
  lastSeen: number;
}

export default function SpeedGame() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const handsRef = useRef<any>(null);
  const rafRef = useRef<number | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  // Tracked hands by spatial side (left side of frame / right side of frame).
  // We assign by x-position rather than MediaPipe's "Left/Right" handedness
  // because handedness can flip during fast motion. Spatial side is stable.
  const leftHand = useRef<HandPoint>({ x: 0.3, y: 0.5, visible: false, lastSeen: 0 });
  const rightHand = useRef<HandPoint>({ x: 0.7, y: 0.5, visible: false, lastSeen: 0 });

  // Rep detection state
  const lastRepTime = useRef<number>(0);
  // Sign of (leftY - rightY). When it flips, hands have crossed vertically.
  const lastSign = useRef<number>(0);
  // Track recent extremes to enforce minimum movement amplitude
  const leftExtreme = useRef<{ min: number; max: number }>({ min: 1, max: 0 });
  const rightExtreme = useRef<{ min: number; max: number }>({ min: 1, max: 0 });

  const [phase, setPhase] = useState<Phase>("idle");
  const [reps, setReps] = useState(0);
  const [timeLeft, setTimeLeft] = useState(GAME_DURATION);
  const [countdown, setCountdown] = useState(3);
  const [status, setStatus] = useState("Loading camera…");
  const [bothHandsVisible, setBothHandsVisible] = useState(false);
  const [popKey, setPopKey] = useState(0);
  const [highScore, setHighScore] = useState<number>(0);
  const [error, setError] = useState<string | null>(null);

  const phaseRef = useRef<Phase>("idle");
  useEffect(() => { phaseRef.current = phase; }, [phase]);

  // Load high score
  useEffect(() => {
    const hs = Number(localStorage.getItem("speedgame_hs") || "0");
    setHighScore(hs);
  }, []);

  // ============================================================
  // Rep detection — positional crossing logic
  // ============================================================
  const detectRep = useCallback(() => {
    const L = leftHand.current;
    const R = rightHand.current;
    if (!L.visible || !R.visible) return;

    // Update extremes for amplitude check
    leftExtreme.current.min = Math.min(leftExtreme.current.min, L.y);
    leftExtreme.current.max = Math.max(leftExtreme.current.max, L.y);
    rightExtreme.current.min = Math.min(rightExtreme.current.min, R.y);
    rightExtreme.current.max = Math.max(rightExtreme.current.max, R.y);

    const diff = L.y - R.y;
    const sign = diff > DEAD_ZONE ? 1 : diff < -DEAD_ZONE ? -1 : 0;

    if (sign === 0) return; // dead zone — hands roughly level

    if (lastSign.current === 0) {
      lastSign.current = sign;
      return;
    }

    if (sign !== lastSign.current) {
      // Crossing detected
      const now = performance.now();
      const sinceLast = now - lastRepTime.current;
      const leftAmp = leftExtreme.current.max - leftExtreme.current.min;
      const rightAmp = rightExtreme.current.max - rightExtreme.current.min;

      if (sinceLast > REP_COOLDOWN_MS && (leftAmp > MIN_MOVE_THRESHOLD || rightAmp > MIN_MOVE_THRESHOLD)) {
        if (phaseRef.current === "playing") {
          setReps((r) => r + 1);
          setPopKey((k) => k + 1);
          lastRepTime.current = now;
          // Reset amplitude window after counting
          leftExtreme.current = { min: L.y, max: L.y };
          rightExtreme.current = { min: R.y, max: R.y };
        }
      }
      lastSign.current = sign;
    }
  }, []);

  // ============================================================
  // MediaPipe results handler
  // ============================================================
  const onResults = useCallback((results: any) => {
    const now = performance.now();
    const landmarks = results.multiHandLandmarks || [];

    // Use wrist (landmark 0) — stable, near forearm, less affected by finger blur
    const points = landmarks.map((lm: any[]) => {
      // Average wrist + middle MCP for a slightly more stable "palm" point
      const wrist = lm[0];
      const mcp = lm[9];
      return {
        x: (wrist.x + mcp.x) / 2,
        y: (wrist.y + mcp.y) / 2,
      };
    });

    // Assignment: ALWAYS bind detected points to the two slots via
    // nearest-neighbor against last known position. If only one hand is
    // visible, ONLY that slot updates — the other slot is hidden, so we
    // never draw a stale dot on top of the visible hand.
    let leftDetect: { x: number; y: number } | null = null;
    let rightDetect: { x: number; y: number } | null = null;

    if (points.length >= 2) {
      // Two hands: assign to minimize total distance from last known slots.
      const [a, b] = points;
      const lastL = leftHand.current;
      const lastR = rightHand.current;
      // If neither slot has ever been seen, fall back to spatial x order.
      if (!lastL.visible && !lastR.visible) {
        const sorted = [a, b].sort((p, q) => p.x - q.x);
        leftDetect = sorted[0];
        rightDetect = sorted[1];
      } else {
        const costAA_BB =
          Math.hypot(a.x - lastL.x, a.y - lastL.y) +
          Math.hypot(b.x - lastR.x, b.y - lastR.y);
        const costAB_BA =
          Math.hypot(a.x - lastR.x, a.y - lastR.y) +
          Math.hypot(b.x - lastL.x, b.y - lastL.y);
        if (costAA_BB <= costAB_BA) {
          leftDetect = a; rightDetect = b;
        } else {
          leftDetect = b; rightDetect = a;
        }
      }
    } else if (points.length === 1) {
      const p = points[0];
      const lastL = leftHand.current;
      const lastR = rightHand.current;
      // Pick whichever slot is closer AND was recently visible.
      const distL = lastL.visible ? Math.hypot(p.x - lastL.x, p.y - lastL.y) : Infinity;
      const distR = lastR.visible ? Math.hypot(p.x - lastR.x, p.y - lastR.y) : Infinity;
      if (distL === Infinity && distR === Infinity) {
        // Cold start with one hand — assign by frame side.
        if (p.x < 0.5) leftDetect = p; else rightDetect = p;
      } else if (distL <= distR) {
        leftDetect = p;
      } else {
        rightDetect = p;
      }
    }

    // EWMA smoothing — high alpha for responsiveness
    if (leftDetect) {
      leftHand.current.x = leftHand.current.visible
        ? leftHand.current.x * (1 - SMOOTHING) + leftDetect.x * SMOOTHING
        : leftDetect.x;
      leftHand.current.y = leftHand.current.visible
        ? leftHand.current.y * (1 - SMOOTHING) + leftDetect.y * SMOOTHING
        : leftDetect.y;
      leftHand.current.visible = true;
      leftHand.current.lastSeen = now;
    } else if (now - leftHand.current.lastSeen > HAND_LOST_MS) {
      leftHand.current.visible = false;
    }

    if (rightDetect) {
      rightHand.current.x = rightHand.current.visible
        ? rightHand.current.x * (1 - SMOOTHING) + rightDetect.x * SMOOTHING
        : rightDetect.x;
      rightHand.current.y = rightHand.current.visible
        ? rightHand.current.y * (1 - SMOOTHING) + rightDetect.y * SMOOTHING
        : rightDetect.y;
      rightHand.current.visible = true;
      rightHand.current.lastSeen = now;
    } else if (now - rightHand.current.lastSeen > HAND_LOST_MS) {
      rightHand.current.visible = false;
    }

    const both = leftHand.current.visible && rightHand.current.visible;
    setBothHandsVisible(both);

    if (phaseRef.current === "playing") detectRep();

    drawOverlay();
  }, [detectRep]);

  // ============================================================
  // Draw hand indicators on canvas overlay
  // ============================================================
  const drawOverlay = useCallback(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;
    const w = canvas.width = video.videoWidth || 640;
    const h = canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, w, h);

    const drawHand = (p: HandPoint, color: string) => {
      if (!p.visible) return;
      // Mirror x for display (we mirror the video via CSS scaleX(-1))
      const x = (1 - p.x) * w;
      const y = p.y * h;
      const grad = ctx.createRadialGradient(x, y, 4, x, y, 50);
      grad.addColorStop(0, color);
      grad.addColorStop(1, "rgba(0,0,0,0)");
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(x, y, 50, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x, y, 14, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "white";
      ctx.lineWidth = 3;
      ctx.stroke();
    };

    drawHand(leftHand.current, "rgba(255, 60, 80, 0.95)");
    drawHand(rightHand.current, "rgba(80, 200, 255, 0.95)");
  }, []);

  // ============================================================
  // Camera + MediaPipe init
  // ============================================================
  useEffect(() => {
    let cancelled = false;
    let processing = false;

    async function init() {
      try {
        await loadScript(MP_HANDS_URL);
        if (cancelled) return;

        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, facingMode: "user", frameRate: { ideal: 60 } },
          audio: false,
        });
        streamRef.current = stream;
        if (cancelled) { stream.getTracks().forEach(t => t.stop()); return; }

        const video = videoRef.current!;
        video.srcObject = stream;
        await video.play();

        const Hands = window.Hands;
        const hands = new Hands({
          locateFile: (file: string) => MP_BASE + file,
        });
        hands.setOptions({
          maxNumHands: 2,
          modelComplexity: 0, // fastest model — prioritize latency
          minDetectionConfidence: 0.25,
          minTrackingConfidence: 0.2,
        });
        hands.onResults(onResults);
        handsRef.current = hands;

        setStatus("Show both hands");

        // Manual rAF loop (faster + more control than @mediapipe/camera_utils)
        const tick = async () => {
          if (cancelled) return;
          if (!processing && video.readyState >= 2) {
            processing = true;
            try {
              await hands.send({ image: video });
            } catch (_) { /* ignore intermittent send errors */ }
            processing = false;
          }
          rafRef.current = requestAnimationFrame(tick);
        };
        tick();
      } catch (e: any) {
        console.error(e);
        setError(e?.message || "Camera access failed. Please allow camera permissions.");
      }
    }

    init();

    return () => {
      cancelled = true;
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      streamRef.current?.getTracks().forEach(t => t.stop());
      handsRef.current?.close?.();
    };
  }, [onResults]);

  // ============================================================
  // Game flow: countdown + timer
  // ============================================================
  useEffect(() => {
    if (phase !== "countdown") return;
    setCountdown(3);
    let n = 3;
    const id = setInterval(() => {
      n -= 1;
      if (n <= 0) {
        clearInterval(id);
        setCountdown(0);
        // brief "GO!" then start
        setTimeout(() => {
          setReps(0);
          lastSign.current = 0;
          lastRepTime.current = 0;
          leftExtreme.current = { min: 1, max: 0 };
          rightExtreme.current = { min: 1, max: 0 };
          setTimeLeft(GAME_DURATION);
          setPhase("playing");
        }, 400);
      } else {
        setCountdown(n);
      }
    }, 800);
    return () => clearInterval(id);
  }, [phase]);

  useEffect(() => {
    if (phase !== "playing") return;
    const startedAt = performance.now();
    const id = setInterval(() => {
      const elapsed = (performance.now() - startedAt) / 1000;
      const left = Math.max(0, GAME_DURATION - elapsed);
      setTimeLeft(left);
      if (left <= 0) {
        clearInterval(id);
        setPhase("finished");
      }
    }, 50);
    return () => clearInterval(id);
  }, [phase]);

  // Save high score
  useEffect(() => {
    if (phase === "finished" && reps > highScore) {
      setHighScore(reps);
      localStorage.setItem("speedgame_hs", String(reps));
    }
  }, [phase, reps, highScore]);

  // Beep on rep
  useEffect(() => {
    if (popKey === 0) return;
    try {
      const ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
      const o = ctx.createOscillator();
      const g = ctx.createGain();
      o.frequency.value = 880;
      o.type = "triangle";
      g.gain.value = 0.08;
      o.connect(g).connect(ctx.destination);
      o.start();
      g.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.08);
      o.stop(ctx.currentTime + 0.09);
    } catch {}
  }, [popKey]);

  const start = () => {
    if (!bothHandsVisible) {
      setStatus("Show both hands first!");
      return;
    }
    setPhase("countdown");
  };

  const timerProgress = phase === "playing" ? timeLeft / GAME_DURATION : phase === "finished" ? 0 : 1;
  const timerColor = timeLeft <= 5 && phase === "playing" ? "stroke-[oklch(0.72_0.22_25)]" : "stroke-[oklch(0.82_0.18_200)]";

  return (
    <div className="fixed inset-0 bg-background overflow-hidden text-foreground">
      {/* Camera feed (mirrored) */}
      <video
        ref={videoRef}
        playsInline
        muted
        className="absolute inset-0 w-full h-full object-cover"
        style={{ transform: "scaleX(-1)" }}
      />
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full object-cover pointer-events-none"
      />
      {/* Vignette */}
      <div className="absolute inset-0 pointer-events-none bg-gradient-to-b from-black/60 via-transparent to-black/70" />

      {/* Top bar */}
      <div className="absolute top-0 left-0 right-0 p-6 flex items-start justify-between z-10">
        <div>
          <h1 className="text-2xl font-black tracking-tight">
            <span className="text-[oklch(0.72_0.22_25)]">41</span> SPEED
          </h1>
          <p className="text-xs text-white/60 mt-1">Hand crossing rep counter</p>
        </div>
        <div className="text-right">
          <div className="text-xs uppercase tracking-widest text-white/60">High</div>
          <div className="text-xl font-bold">{highScore}</div>
        </div>
      </div>

      {/* Status pill */}
      <div className="absolute top-24 left-1/2 -translate-x-1/2 z-10 flex items-center gap-2 px-4 py-2 rounded-full bg-black/50 backdrop-blur-md border border-white/10">
        <span className={`w-2 h-2 rounded-full ${bothHandsVisible ? "bg-emerald-400" : "bg-amber-400"}`} />
        <span className="text-sm font-medium">
          {error ? error : phase === "idle" ? (bothHandsVisible ? "Ready" : status) : phase === "playing" ? "GO!" : phase === "finished" ? "Done!" : "Get ready…"}
        </span>
      </div>

      {/* Center rep counter (during play) */}
      {(phase === "playing" || phase === "finished") && (
        <div className="absolute inset-0 flex items-center justify-center z-10 pointer-events-none">
          <div
            key={popKey}
            className="text-[14rem] leading-none font-black text-white drop-shadow-[0_0_40px_rgba(0,0,0,0.7)] rep-pop"
            style={{ textShadow: "0 0 60px oklch(0.72 0.22 25 / 0.5)" }}
          >
            {reps}
          </div>
        </div>
      )}

      {/* Countdown overlay */}
      {phase === "countdown" && (
        <div className="absolute inset-0 flex items-center justify-center z-20 bg-black/40 backdrop-blur-sm">
          <div className="text-[16rem] font-black text-white animate-pulse">
            {countdown > 0 ? countdown : "GO!"}
          </div>
        </div>
      )}

      {/* Bottom: timer */}
      {(phase === "playing" || phase === "countdown") && (
        <div className="absolute bottom-8 left-1/2 -translate-x-1/2 z-10 flex flex-col items-center">
          <div className="relative w-24 h-24">
            <svg className="w-full h-full -rotate-90" viewBox="0 0 100 100">
              <circle cx="50" cy="50" r="44" className="stroke-white/15 fill-none" strokeWidth="6" />
              <circle
                cx="50" cy="50" r="44"
                className={`fill-none transition-[stroke-dashoffset] duration-100 ${timerColor}`}
                strokeWidth="6"
                strokeLinecap="round"
                strokeDasharray={2 * Math.PI * 44}
                strokeDashoffset={2 * Math.PI * 44 * (1 - timerProgress)}
              />
            </svg>
            <div className="absolute inset-0 flex items-center justify-center text-2xl font-bold">
              {Math.ceil(timeLeft)}
            </div>
          </div>
        </div>
      )}

      {/* Idle / Finished start screen */}
      {(phase === "idle" || phase === "finished") && (
        <div className="absolute inset-0 flex items-center justify-center z-20">
          <div className="bg-black/60 backdrop-blur-xl border border-white/10 rounded-3xl p-10 text-center max-w-md mx-4 shadow-2xl">
            {phase === "finished" ? (
              <>
                <div className="text-sm uppercase tracking-widest text-white/60 mb-2">Final Score</div>
                <div className="text-7xl font-black mb-1 text-[oklch(0.72_0.22_25)]">{reps}</div>
                <div className="text-sm text-white/60 mb-6">reps in {GAME_DURATION}s</div>
                {reps >= highScore && reps > 0 && (
                  <div className="text-xs text-amber-300 mb-4">🏆 New high score!</div>
                )}
              </>
            ) : (
              <>
                <h2 className="text-3xl font-black mb-2">Ready to go?</h2>
                <p className="text-sm text-white/70 mb-6">
                  Hands in front of you, palms up. Swap them up & down as fast as you can for {GAME_DURATION} seconds.
                </p>
              </>
            )}
            <button
              onClick={start}
              disabled={!!error}
              className="w-full py-4 rounded-2xl bg-[oklch(0.72_0.22_25)] hover:bg-[oklch(0.78_0.24_30)] transition-all text-white font-bold text-lg shadow-[0_10px_40px_-10px_oklch(0.72_0.22_25_/_0.8)] disabled:opacity-40"
            >
              {phase === "finished" ? "PLAY AGAIN" : "START"}
            </button>
            {!bothHandsVisible && !error && (
              <p className="mt-4 text-xs text-amber-300">Show both hands to the camera</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
