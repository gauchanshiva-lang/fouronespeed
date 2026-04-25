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
    <div className="fixed inset-0 overflow-hidden text-foreground flex flex-col items-center"
      style={{
        background:
          "radial-gradient(ellipse at top, oklch(0.22 0.05 270) 0%, oklch(0.10 0.03 270) 60%, oklch(0.06 0.02 270) 100%)",
      }}
    >
      {/* Ambient glow blobs */}
      <div className="pointer-events-none absolute -top-40 -left-40 w-[500px] h-[500px] rounded-full opacity-30 blur-3xl"
        style={{ background: "var(--brand)" }} />
      <div className="pointer-events-none absolute -bottom-40 -right-40 w-[600px] h-[600px] rounded-full opacity-20 blur-3xl"
        style={{ background: "var(--accent-cyan)" }} />

      {/* Top bar */}
      <header className="relative z-20 w-full max-w-5xl px-6 pt-6 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl flex items-center justify-center font-black text-white shadow-lg"
            style={{ background: "linear-gradient(135deg, var(--brand), var(--brand-glow))" }}>
            41
          </div>
          <div>
            <h1 className="text-lg font-black tracking-tight leading-none">SPEED</h1>
            <p className="text-[10px] uppercase tracking-[0.2em] text-white/50 mt-1">Hand Swap Challenge</p>
          </div>
        </div>
        <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 backdrop-blur-md border border-white/10">
          <span className="text-[10px] uppercase tracking-widest text-white/50">High</span>
          <span className="text-base font-black tabular-nums" style={{ color: "var(--brand-glow)" }}>{highScore}</span>
        </div>
      </header>

      {/* Centered camera stage */}
      <main className="relative z-10 flex-1 w-full flex items-center justify-center px-6 py-6">
        <div className="relative w-full max-w-3xl aspect-video rounded-3xl overflow-hidden border border-white/10 shadow-[0_30px_80px_-20px_rgba(0,0,0,0.8)]"
          style={{
            background: "oklch(0.10 0.02 270)",
            boxShadow:
              "0 30px 80px -20px rgba(0,0,0,0.8), 0 0 0 1px oklch(1 0 0 / 0.05), 0 0 60px -10px oklch(0.72 0.22 25 / 0.25)",
          }}
        >
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
          <div className="absolute inset-0 pointer-events-none"
            style={{ background: "radial-gradient(ellipse at center, transparent 50%, rgba(0,0,0,0.55) 100%)" }} />

          {/* Status pill */}
          <div className="absolute top-4 left-1/2 -translate-x-1/2 z-10 flex items-center gap-2 px-4 py-2 rounded-full bg-black/60 backdrop-blur-md border border-white/10">
            <span className={`w-2 h-2 rounded-full ${bothArmsVisible ? "bg-emerald-400 shadow-[0_0_10px_rgb(52,211,153)]" : "bg-amber-400 shadow-[0_0_10px_rgb(251,191,36)]"}`} />
            <span className="text-xs font-medium tracking-wide">
              {error ? error : phase === "idle" ? (bothArmsVisible ? "Ready" : status) : phase === "playing" ? "GO!" : phase === "finished" ? "Done!" : "Get ready…"}
            </span>
          </div>

          {/* Live rep counter */}
          {(phase === "playing" || phase === "finished") && (
            <div className="absolute inset-0 flex items-center justify-center z-10 pointer-events-none">
              <div
                key={popKey}
                className="font-black text-white rep-pop tabular-nums leading-none"
                style={{
                  fontSize: "clamp(6rem, 22vw, 16rem)",
                  textShadow: "0 0 60px oklch(0.72 0.22 25 / 0.7), 0 4px 24px rgba(0,0,0,0.8)",
                }}
              >
                {reps}
              </div>
            </div>
          )}

          {/* Countdown overlay */}
          {phase === "countdown" && (
            <div className="absolute inset-0 flex flex-col items-center justify-center z-20 bg-black/60 backdrop-blur-md">
              <div className="text-xs uppercase tracking-[0.4em] text-white/70 mb-6">Get in position</div>
              <div
                key={countdown}
                className="font-black text-white countdown-flash leading-none"
                style={{
                  fontSize: "clamp(8rem, 26vw, 18rem)",
                  textShadow: "0 0 100px oklch(0.72 0.22 25 / 0.9)",
                }}
              >
                {countdown > 0 ? countdown : "GO!"}
              </div>
            </div>
          )}

          {/* Idle / Finished modal */}
          {(phase === "idle" || phase === "finished") && (
            <div className="absolute inset-0 flex items-center justify-center z-20 bg-black/55 backdrop-blur-md p-4">
              <div className="w-full max-w-sm rounded-3xl p-8 text-center border border-white/10"
                style={{
                  background: "linear-gradient(160deg, oklch(0.18 0.03 270 / 0.95), oklch(0.10 0.02 270 / 0.95))",
                  boxShadow: "0 20px 60px -10px rgba(0,0,0,0.6), inset 0 1px 0 oklch(1 0 0 / 0.06)",
                }}
              >
                {phase === "finished" ? (
                  <>
                    <div className="text-[10px] uppercase tracking-[0.3em] text-white/50 mb-3">Final Score</div>
                    <div className="text-7xl font-black mb-1 tabular-nums"
                      style={{
                        background: "linear-gradient(135deg, var(--brand), var(--brand-glow))",
                        WebkitBackgroundClip: "text",
                        WebkitTextFillColor: "transparent",
                      }}>
                      {reps}
                    </div>
                    <div className="text-xs text-white/50 mb-5">reps in {GAME_DURATION}s</div>
                    {reps >= highScore && reps > 0 && (
                      <div className="text-xs text-amber-300 mb-4 font-medium">🏆 New high score!</div>
                    )}
                  </>
                ) : (
                  <>
                    <div className="text-[10px] uppercase tracking-[0.3em] text-white/50 mb-3">Challenge</div>
                    <h2 className="text-3xl font-black mb-3 leading-tight">Ready to go?</h2>
                    <p className="text-sm text-white/60 mb-6 leading-relaxed">
                      Stand back so your shoulders &amp; arms are visible. Swap your arms up &amp; down as fast as you can for <span className="text-white font-semibold">{GAME_DURATION}s</span>.
                    </p>
                  </>
                )}
                <button
                  onClick={start}
                  disabled={!!error}
                  className="w-full py-4 rounded-2xl text-white font-black text-base tracking-wide transition-all hover:scale-[1.02] active:scale-[0.98] disabled:opacity-40 disabled:hover:scale-100"
                  style={{
                    background: "linear-gradient(135deg, var(--brand), var(--brand-glow))",
                    boxShadow: "0 10px 40px -10px oklch(0.72 0.22 25 / 0.8), inset 0 1px 0 oklch(1 0 0 / 0.2)",
                  }}
                >
                  {phase === "finished" ? "PLAY AGAIN" : "START"}
                </button>
                {!bothArmsVisible && !error && (
                  <p className="mt-4 text-xs text-amber-300/80">Show your upper body to the camera</p>
                )}
              </div>
            </div>
          )}
        </div>
      </main>

      {/* Bottom timer */}
      {(phase === "playing" || phase === "countdown") && (
        <div className="relative z-10 pb-8 flex flex-col items-center">
          <div className="relative w-24 h-24">
            <svg className="w-full h-full -rotate-90" viewBox="0 0 100 100">
              <circle cx="50" cy="50" r="44" className="stroke-white/10 fill-none" strokeWidth="6" />
              <circle
                cx="50" cy="50" r="44"
                className={`fill-none transition-[stroke-dashoffset] duration-100 ${timerColor}`}
                strokeWidth="6"
                strokeLinecap="round"
                strokeDasharray={2 * Math.PI * 44}
                strokeDashoffset={2 * Math.PI * 44 * (1 - timerProgress)}
                style={{ filter: "drop-shadow(0 0 8px currentColor)" }}
              />
            </svg>
            <div className="absolute inset-0 flex items-center justify-center text-2xl font-black tabular-nums">
              {Math.ceil(timeLeft)}
            </div>
          </div>
          <div className="text-[10px] uppercase tracking-[0.3em] text-white/40 mt-2">Seconds left</div>
        </div>
      )}

      {/* Footer hint when idle */}
      {phase === "idle" && (
        <footer className="relative z-10 pb-6 text-[10px] uppercase tracking-[0.3em] text-white/30">
          Powered by real-time pose tracking
        </footer>
      )}
    </div>
  );
}
