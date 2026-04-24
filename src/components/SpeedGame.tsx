import { useEffect, useRef, useState, useCallback } from "react";

// ============================================================
// MediaPipe POSE loader (CDN). Pose tracks the whole upper body
// (shoulders, elbows, wrists) — these landmarks remain locked
// even during fast hand motion because the arm chain is large
// and high-contrast vs the background. Much more reliable than
// finger-level Hand tracking for high-speed rep counting.
// ============================================================
const MP_VERSION = "0.5.1675469404";
const MP_POSE_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/pose@${MP_VERSION}/pose.js`;
const MP_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/pose@${MP_VERSION}/`;

function waitForGlobal(name: string, timeoutMs = 8000): Promise<any> {
  return new Promise((resolve, reject) => {
    const start = Date.now();
    const check = () => {
      const v = (window as any)[name];
      if (typeof v === "function") return resolve(v);
      if (Date.now() - start > timeoutMs) return reject(new Error(`${name} not available`));
      setTimeout(check, 50);
    };
    check();
  });
}

declare global {
  interface Window {
    Pose: any;
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

const GAME_DURATION = 20;
const REP_COOLDOWN_MS = 50;
const MIN_MOVE_THRESHOLD = 0.005;
const DEAD_ZONE = 0.0015;
const SMOOTHING = 0.88;
const ARM_LOST_MS = 250;
const VISIBILITY_THRESHOLD = 0.3; // pose landmark visibility cutoff

// MediaPipe Pose landmark indices
const LM = {
  LEFT_SHOULDER: 11,
  RIGHT_SHOULDER: 12,
  LEFT_ELBOW: 13,
  RIGHT_ELBOW: 14,
  LEFT_WRIST: 15,
  RIGHT_WRIST: 16,
};

interface ArmPoint {
  x: number;
  y: number;
  visible: boolean;
  lastSeen: number;
}

export default function SpeedGame() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const poseRef = useRef<any>(null);
  const rafRef = useRef<number | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  // NOTE: MediaPipe Pose returns landmarks from the SUBJECT's perspective.
  // Subject's left wrist appears on the RIGHT side of an unmirrored frame.
  // We use the labels as-is (subject perspective). Display mirroring is
  // handled in the draw step.
  const leftArm = useRef<ArmPoint>({ x: 0.3, y: 0.5, visible: false, lastSeen: 0 });
  const rightArm = useRef<ArmPoint>({ x: 0.7, y: 0.5, visible: false, lastSeen: 0 });

  const lastRepTime = useRef<number>(0);
  const lastSign = useRef<number>(0);
  const leftExtreme = useRef<{ min: number; max: number }>({ min: 1, max: 0 });
  const rightExtreme = useRef<{ min: number; max: number }>({ min: 1, max: 0 });

  const [phase, setPhase] = useState<Phase>("idle");
  const [reps, setReps] = useState(0);
  const [timeLeft, setTimeLeft] = useState(GAME_DURATION);
  const [countdown, setCountdown] = useState(3);
  const [status, setStatus] = useState("Loading camera…");
  const [bothArmsVisible, setBothArmsVisible] = useState(false);
  const [popKey, setPopKey] = useState(0);
  const [highScore, setHighScore] = useState<number>(0);
  const [error, setError] = useState<string | null>(null);

  const phaseRef = useRef<Phase>("idle");
  useEffect(() => { phaseRef.current = phase; }, [phase]);

  useEffect(() => {
    const hs = Number(localStorage.getItem("speedgame_hs") || "0");
    setHighScore(hs);
  }, []);

  // ============================================================
  // Rep detection — vertical crossing of the two forearm points
  // ============================================================
  const detectRep = useCallback(() => {
    const L = leftArm.current;
    const R = rightArm.current;
    if (!L.visible || !R.visible) return;

    leftExtreme.current.min = Math.min(leftExtreme.current.min, L.y);
    leftExtreme.current.max = Math.max(leftExtreme.current.max, L.y);
    rightExtreme.current.min = Math.min(rightExtreme.current.min, R.y);
    rightExtreme.current.max = Math.max(rightExtreme.current.max, R.y);

    const diff = L.y - R.y;
    const sign = diff > DEAD_ZONE ? 1 : diff < -DEAD_ZONE ? -1 : 0;
    if (sign === 0) return;

    if (lastSign.current === 0) {
      lastSign.current = sign;
      return;
    }

    if (sign !== lastSign.current) {
      const now = performance.now();
      const sinceLast = now - lastRepTime.current;
      const leftAmp = leftExtreme.current.max - leftExtreme.current.min;
      const rightAmp = rightExtreme.current.max - rightExtreme.current.min;

      if (sinceLast > REP_COOLDOWN_MS && (leftAmp > MIN_MOVE_THRESHOLD || rightAmp > MIN_MOVE_THRESHOLD)) {
        if (phaseRef.current === "playing") {
          setReps((r) => r + 1);
          setPopKey((k) => k + 1);
          lastRepTime.current = now;
          leftExtreme.current = { min: L.y, max: L.y };
          rightExtreme.current = { min: R.y, max: R.y };
        }
      }
      lastSign.current = sign;
    }
  }, []);

  // ============================================================
  // Pose results handler
  //
  // We compute the FOREARM MIDPOINT (between elbow and wrist).
  // This is a large, stable region BELOW the hand — much easier
  // for the model to track during fast motion than the hand itself.
  // ============================================================
  const onResults = useCallback((results: any) => {
    const now = performance.now();
    const lm = results.poseLandmarks;

    if (lm) {
      // Subject's LEFT side
      const lWrist = lm[LM.LEFT_WRIST];
      const lElbow = lm[LM.LEFT_ELBOW];
      if (lWrist && lElbow &&
          (lWrist.visibility ?? 1) > VISIBILITY_THRESHOLD &&
          (lElbow.visibility ?? 1) > VISIBILITY_THRESHOLD) {
        // Forearm midpoint, biased slightly toward wrist (0.65 wrist, 0.35 elbow)
        const px = lWrist.x * 0.65 + lElbow.x * 0.35;
        const py = lWrist.y * 0.65 + lElbow.y * 0.35;
        leftArm.current.x = leftArm.current.visible
          ? leftArm.current.x * (1 - SMOOTHING) + px * SMOOTHING
          : px;
        leftArm.current.y = leftArm.current.visible
          ? leftArm.current.y * (1 - SMOOTHING) + py * SMOOTHING
          : py;
        leftArm.current.visible = true;
        leftArm.current.lastSeen = now;
      } else if (now - leftArm.current.lastSeen > ARM_LOST_MS) {
        leftArm.current.visible = false;
      }

      // Subject's RIGHT side
      const rWrist = lm[LM.RIGHT_WRIST];
      const rElbow = lm[LM.RIGHT_ELBOW];
      if (rWrist && rElbow &&
          (rWrist.visibility ?? 1) > VISIBILITY_THRESHOLD &&
          (rElbow.visibility ?? 1) > VISIBILITY_THRESHOLD) {
        const px = rWrist.x * 0.65 + rElbow.x * 0.35;
        const py = rWrist.y * 0.65 + rElbow.y * 0.35;
        rightArm.current.x = rightArm.current.visible
          ? rightArm.current.x * (1 - SMOOTHING) + px * SMOOTHING
          : px;
        rightArm.current.y = rightArm.current.visible
          ? rightArm.current.y * (1 - SMOOTHING) + py * SMOOTHING
          : py;
        rightArm.current.visible = true;
        rightArm.current.lastSeen = now;
      } else if (now - rightArm.current.lastSeen > ARM_LOST_MS) {
        rightArm.current.visible = false;
      }
    } else {
      if (now - leftArm.current.lastSeen > ARM_LOST_MS) leftArm.current.visible = false;
      if (now - rightArm.current.lastSeen > ARM_LOST_MS) rightArm.current.visible = false;
    }

    const both = leftArm.current.visible && rightArm.current.visible;
    setBothArmsVisible(both);

    if (phaseRef.current === "playing") detectRep();
    drawOverlay();
  }, [detectRep]);

  // ============================================================
  // Draw forearm trackers on canvas overlay
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

    const drawArm = (p: ArmPoint, color: string) => {
      if (!p.visible) return;
      // Mirror x (video is flipped via CSS scaleX(-1))
      const x = (1 - p.x) * w;
      const y = p.y * h;
      const grad = ctx.createRadialGradient(x, y, 4, x, y, 70);
      grad.addColorStop(0, color);
      grad.addColorStop(1, "rgba(0,0,0,0)");
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(x, y, 70, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x, y, 18, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "white";
      ctx.lineWidth = 4;
      ctx.stroke();
    };

    drawArm(leftArm.current, "rgba(255, 60, 80, 0.95)");
    drawArm(rightArm.current, "rgba(80, 200, 255, 0.95)");
  }, []);

  // ============================================================
  // Camera + MediaPipe Pose init
  // ============================================================
  useEffect(() => {
    let cancelled = false;
    let processing = false;

    async function init() {
      try {
        await loadScript(MP_POSE_URL);
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

        const Pose = await waitForGlobal("Pose");
        const pose = new Pose({
          locateFile: (file: string) => MP_BASE + file,
        });
        pose.setOptions({
          modelComplexity: 0,           // lightweight, fastest model
          smoothLandmarks: true,        // built-in smoothing helps fast motion
          enableSegmentation: false,
          minDetectionConfidence: 0.3,
          minTrackingConfidence: 0.3,
        });
        pose.onResults(onResults);
        poseRef.current = pose;

        setStatus("Show your upper body");

        const tick = async () => {
          if (cancelled) return;
          if (!processing && video.readyState >= 2) {
            processing = true;
            try {
              await pose.send({ image: video });
            } catch (_) { /* ignore */ }
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
      poseRef.current?.close?.();
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

  useEffect(() => {
    if (phase === "finished" && reps > highScore) {
      setHighScore(reps);
      localStorage.setItem("speedgame_hs", String(reps));
    }
  }, [phase, reps, highScore]);

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
    setStatus("Get in position!");
    setPhase("countdown");
  };

  const timerProgress = phase === "playing" ? timeLeft / GAME_DURATION : phase === "finished" ? 0 : 1;
  const timerColor = timeLeft <= 5 && phase === "playing" ? "stroke-[oklch(0.72_0.22_25)]" : "stroke-[oklch(0.82_0.18_200)]";

  return (
    <div className="fixed inset-0 bg-background overflow-hidden text-foreground">
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
      <div className="absolute inset-0 pointer-events-none bg-gradient-to-b from-black/60 via-transparent to-black/70" />

      <div className="absolute top-0 left-0 right-0 p-6 flex items-start justify-between z-10">
        <div>
          <h1 className="text-2xl font-black tracking-tight">
            <span className="text-[oklch(0.72_0.22_25)]">41</span> SPEED
          </h1>
          <p className="text-xs text-white/60 mt-1">Forearm crossing rep counter</p>
        </div>
        <div className="text-right">
          <div className="text-xs uppercase tracking-widest text-white/60">High</div>
          <div className="text-xl font-bold">{highScore}</div>
        </div>
      </div>

      <div className="absolute top-24 left-1/2 -translate-x-1/2 z-10 flex items-center gap-2 px-4 py-2 rounded-full bg-black/50 backdrop-blur-md border border-white/10">
        <span className={`w-2 h-2 rounded-full ${bothArmsVisible ? "bg-emerald-400" : "bg-amber-400"}`} />
        <span className="text-sm font-medium">
          {error ? error : phase === "idle" ? (bothArmsVisible ? "Ready" : status) : phase === "playing" ? "GO!" : phase === "finished" ? "Done!" : "Get ready…"}
        </span>
      </div>

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

      {phase === "countdown" && (
        <div className="absolute inset-0 flex flex-col items-center justify-center z-20 bg-black/50 backdrop-blur-sm">
          <div className="text-sm uppercase tracking-widest text-white/70 mb-4">Get in position</div>
          <div
            key={countdown}
            className="text-[16rem] leading-none font-black text-white countdown-flash"
            style={{ textShadow: "0 0 80px oklch(0.72 0.22 25 / 0.8)" }}
          >
            {countdown > 0 ? countdown : "GO!"}
          </div>
        </div>
      )}

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
                  Stand back so your shoulders & arms are visible. Swap your arms up & down as fast as you can for {GAME_DURATION} seconds.
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
            {!bothArmsVisible && !error && (
              <p className="mt-4 text-xs text-amber-300">Show your upper body to the camera</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
