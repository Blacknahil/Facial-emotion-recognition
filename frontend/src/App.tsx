import { useCallback, useEffect, useRef, useState } from "react";
import "./App.css";

type PredictResponse = {
  dominant: string;
  scores: Record<string, number>;
};

type ModelOption = "deepface" | "hsemotion" | "finetuned";

const MODEL_OPTIONS: { value: ModelOption; label: string }[] = [
  { value: "deepface", label: "DeepFace (baseline)" },
  { value: "hsemotion", label: "HSEmotion (AffectNet)" },
  { value: "finetuned", label: "Fine-tuned (FER+)" },
];

const API_BASE = "/api";

async function postFrame(blob: Blob, model: ModelOption): Promise<PredictResponse> {
  const form = new FormData();
  form.append("file", blob, "frame.jpg");

  const res = await fetch(`${API_BASE}/predict?model=${model}`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = (await res.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail))
        detail = body.detail.map((d) => String(d)).join(", ");
    } catch {
      /* ignore */
    }
    throw new Error(detail || `Request failed (${res.status})`);
  }

  return res.json() as Promise<PredictResponse>;
}

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const [cameraOn, setCameraOn] = useState(false);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [model, setModel] = useState<ModelOption>("deepface");

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraOn(false);
  }, []);

  const startCamera = useCallback(async () => {
    setError(null);
    setStatus("Starting camera…");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setCameraOn(true);
      setStatus("Camera ready. Capture a frame when your face is visible.");
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Could not access camera.";
      setError(msg);
      setStatus("");
    }
  }, []);

  useEffect(() => {
    return () => stopCamera();
  }, [stopCamera]);

  const captureAndAnalyze = useCallback(async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !cameraOn) return;

    const w = video.videoWidth;
    const h = video.videoHeight;
    if (!w || !h) {
      setError("Video not ready yet.");
      return;
    }

    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, w, h);

    setLoading(true);
    setError(null);
    setStatus("Analyzing…");

    await new Promise<void>((resolve) =>
      canvas.toBlob(
        async (blob) => {
          if (!blob) {
            setError("Could not capture image.");
            setLoading(false);
            setStatus("");
            resolve();
            return;
          }
          try {
            const data = await postFrame(blob, model);
            setResult(data);
            setStatus("Done.");
          } catch (e) {
            setResult(null);
            setError(e instanceof Error ? e.message : "Request failed.");
            setStatus("");
          } finally {
            setLoading(false);
          }
          resolve();
        },
        "image/jpeg",
        0.92,
      ),
    );
  }, [cameraOn, model]);

  const onFile = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      e.target.value = "";
      if (!file || !file.type.startsWith("image/")) return;

      setLoading(true);
      setError(null);
      setStatus("Analyzing upload…");
      try {
        const data = await postFrame(file, model);
        setResult(data);
        setStatus("Done.");
      } catch (err) {
        setResult(null);
        setError(err instanceof Error ? err.message : "Request failed.");
        setStatus("");
      } finally {
        setLoading(false);
      }
    },
    [model],
  );

  const sortedScores = result
    ? Object.entries(result.scores).sort((a, b) => b[1] - a[1])
    : [];

  return (
    <div className="page">
      <h1>Facial emotion recognition</h1>
      <p className="subtitle">
        Webcam or image upload — choose a model below, then analyze.
      </p>

      <div className="panel">
        <div className="model-selector">
          <label htmlFor="model-select">Model:</label>
          <select
            id="model-select"
            value={model}
            onChange={(e) => setModel(e.target.value as ModelOption)}
            disabled={loading}
          >
            {MODEL_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>

        <div className="video-wrap">
          <video ref={videoRef} playsInline muted />
        </div>
        <canvas ref={canvasRef} className="hidden-canvas" />

        <div className="toolbar">
          {!cameraOn ? (
            <button type="button" className="primary" onClick={startCamera} disabled={loading}>
              Start camera
            </button>
          ) : (
            <>
              <button type="button" className="primary" onClick={captureAndAnalyze} disabled={loading}>
                {loading ? "Working…" : "Analyze frame"}
              </button>
              <button type="button" onClick={stopCamera} disabled={loading}>
                Stop camera
              </button>
            </>
          )}

          <label className="file-input">
            Or upload:{" "}
            <input
              type="file"
              accept="image/*"
              onChange={onFile}
              disabled={loading}
            />
          </label>
        </div>

        <p className={`status ${error ? "error" : ""}`}>{error || status}</p>
      </div>

      {result && (
        <div className="panel result-block">
          <h2>Prediction</h2>
          <div className="dominant">{result.dominant}</div>
          <div className="scores">
            {sortedScores.map(([label, value]) => (
              <div key={label} className="score-row">
                <span>{label}</span>
                <span>{(value * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
