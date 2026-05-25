"use client";

import { useEffect, useRef } from "react";
import styles from "./PredictionViewer.module.css";

// Canvas sizing - render bigger than the source so off-screen bboxes have
// room to show. The image is centered with margins on all sides.
const CANVAS_W = 1080;
const CANVAS_H = 720;

export default function PredictionViewer({ selection, prediction, loading, error }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!selection?.previewUrl) return;
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    canvas.width = CANVAS_W;
    canvas.height = CANVAS_H;

    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      ctx.fillStyle = "#000";
      ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);

      // Fit the image inside the canvas while preserving aspect ratio,
      // leaving 180 px of margin on each side for off-screen bboxes.
      const margin = 180;
      const maxW = CANVAS_W - 2 * margin;
      const maxH = CANVAS_H - 2 * margin;
      const scale = Math.min(maxW / img.width, maxH / img.height);
      const drawW = img.width * scale;
      const drawH = img.height * scale;
      const offX = (CANVAS_W - drawW) / 2;
      const offY = (CANVAS_H - drawH) / 2;
      ctx.drawImage(img, offX, offY, drawW, drawH);

      // Frame outline
      ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
      ctx.lineWidth = 1;
      ctx.strokeRect(offX, offY, drawW, drawH);

      function drawBBox(bbox, color, label) {
        const x = offX + bbox.xmin * scale;
        const y = offY + bbox.ymin * scale;
        const w = (bbox.xmax - bbox.xmin) * scale;
        const h = (bbox.ymax - bbox.ymin) * scale;
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, w, h);
        ctx.fillStyle = color + "33";
        ctx.fillRect(x, y, w, h);
        ctx.fillStyle = color;
        ctx.font = "13px ui-monospace, Menlo, monospace";
        ctx.fillText(label, x + 6, y - 8);
      }

      if (selection.groundTruth) {
        drawBBox(selection.groundTruth, "#4ade80", "Ground truth");
      }
      if (prediction?.bbox) {
        drawBBox(prediction.bbox, "#f87171", "Predicted");
      }
    };
    img.src = selection.previewUrl;
  }, [selection, prediction]);

  return (
    <div className={styles.viewer}>
      <div className={styles.toolbar}>
        <div className={styles.label}>Visualization</div>
        <div className={styles.legend}>
          {selection?.groundTruth && (
            <span className={styles.legendItem}>
              <span className={`${styles.swatch} ${styles.swatchGt}`} />
              Ground truth
            </span>
          )}
          <span className={styles.legendItem}>
            <span className={`${styles.swatch} ${styles.swatchPred}`} />
            Predicted
          </span>
        </div>
      </div>

      <div className={styles.canvasWrap}>
        <canvas ref={canvasRef} className={styles.canvas} />
        {!selection && (
          <div className={styles.placeholder}>
            Upload or select a sample image to see a prediction
          </div>
        )}
        {error && !loading && (
          <div className={styles.placeholder}>{error}</div>
        )}
        {loading && (
          <div className={styles.spinner}>
            <div className={styles.spinnerRing} />
            <div>Running inference...</div>
          </div>
        )}
      </div>
    </div>
  );
}
