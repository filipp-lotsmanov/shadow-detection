"use client";

import { useState, useEffect, useCallback } from "react";
import ImageUploader from "@/components/ImageUploader";
import PredictionViewer from "@/components/PredictionViewer";
import PredictionStats from "@/components/PredictionStats";
import styles from "./page.module.css";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function HomePage() {
  const [selection, setSelection] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [health, setHealth] = useState({ checked: false, status: "unknown", device: "" });

  useEffect(() => {
    fetch(`${API_URL}/health`)
      .then((r) => r.json())
      .then((d) =>
        setHealth({
          checked: true,
          status: d.model_loaded ? "ok" : d.status,
          device: d.device,
        })
      )
      .catch(() =>
        setHealth({ checked: true, status: "unreachable", device: "" })
      );
  }, []);

  const runPrediction = useCallback(async (sel) => {
    setLoading(true);
    setError(null);
    setPrediction(null);
    try {
      const formData = new FormData();
      if (sel.file) {
        formData.append("file", sel.file);
      } else {
        const res = await fetch(sel.previewUrl);
        const blob = await res.blob();
        formData.append("file", blob, `${sel.id}.png`);
      }
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        const detail = await res.json().catch(() => null);
        throw new Error(detail?.detail || `Backend returned HTTP ${res.status}`);
      }
      const data = await res.json();
      setPrediction(data);
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  function handleSelect(sel) {
    setSelection(sel);
    runPrediction(sel);
  }

  return (
    <main className={styles.container}>
      <header className={styles.header}>
        <div className={styles.eyebrow}>BrabantHack 2026 - DEMCON Deep Tech Track Winner</div>
        <h1 className={styles.title}>Shadow Detection</h1>
        <p className={styles.subtitle}>
          A ResNet-50 with hand-crafted geometric features predicts where a pedestrian
          is standing off-screen, using only their shadow as a visual cue. Achieved IoU 0.626
          on the official test set.
        </p>
        <div className={styles.status}>
          <span
            className={`${styles.statusDot} ${
              health.status === "ok"
                ? styles.statusOk
                : health.status === "unreachable" || health.status === "model_missing"
                ? styles.statusBad
                : ""
            }`}
          />
          {!health.checked
            ? "Checking backend..."
            : health.status === "ok"
            ? `Backend online (${health.device})`
            : health.status === "unreachable"
            ? `Backend unreachable at ${API_URL}`
            : `Backend running but model not loaded`}
        </div>
      </header>

      {(health.status === "unreachable" || health.status === "model_missing") && (
        <div className={styles.errorBanner}>
          <strong>Backend not ready.</strong>{" "}
          {health.status === "unreachable" ? (
            <>
              Start it with <code>cd backend && uv run uvicorn app.main:app --port 8000</code>.
            </>
          ) : (
            <>
              The server is running but no model is loaded. Run the export script first:{" "}
              <code>uv run python scripts/export_model.py ...</code>
            </>
          )}
        </div>
      )}

      <div className={styles.layout}>
        <div>
          <ImageUploader onSelect={handleSelect} activeId={selection?.id} />
          <PredictionViewer
            selection={selection}
            prediction={prediction}
            loading={loading}
            error={error}
          />
        </div>
        <PredictionStats prediction={prediction} selection={selection} />
      </div>
    </main>
  );
}
