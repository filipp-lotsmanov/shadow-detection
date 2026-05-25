"use client";

import styles from "./PredictionStats.module.css";

const SIDE_LABEL = { 0: "Left", 1: "Right" };
const DIRECTION_LABEL = {
  0: "Out of frame",
  1: "Into frame",
  "-1": "Abstain",
};

function fmt(n, digits = 1) {
  return Number(n).toFixed(digits);
}

function ConfidenceBar({ value }) {
  return (
    <div className={styles.bar}>
      <div
        className={styles.barFill}
        style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }}
      />
    </div>
  );
}

export default function PredictionStats({ prediction, selection }) {
  if (!prediction) {
    return (
      <aside className={styles.panel}>
        <div className={styles.panelTitle}>Prediction</div>
        <div className={styles.empty}>No prediction yet.</div>
      </aside>
    );
  }

  const gt = selection?.groundTruth;
  const sideMatch = gt ? prediction.side === inferSideFromBBox(gt) : null;

  return (
    <aside className={styles.panel}>
      <div className={styles.panelTitle}>Prediction</div>

      <div className={styles.section}>
        <div className={styles.sectionLabel}>Side</div>
        <div className={styles.bigValue}>{SIDE_LABEL[prediction.side]}</div>
        <div className={styles.confidence}>
          {fmt(prediction.side_confidence * 100, 1)}% confidence
        </div>
        <ConfidenceBar value={prediction.side_confidence} />
        {sideMatch !== null && (
          <div
            className={sideMatch ? styles.match : styles.mismatch}
            style={{ marginTop: 8 }}
          >
            {sideMatch ? "matches ground truth" : "does not match ground truth"}
          </div>
        )}
      </div>

      <div className={styles.section}>
        <div className={styles.sectionLabel}>Direction</div>
        {prediction.direction === -1 ? (
          <div className={`${styles.bigValue} ${styles.abstain}`}>
            Abstain
          </div>
        ) : (
          <div className={styles.bigValue}>{DIRECTION_LABEL[prediction.direction]}</div>
        )}
        <div className={styles.confidence}>
          {fmt(prediction.direction_confidence * 100, 1)}% confidence
        </div>
        <ConfidenceBar value={prediction.direction_confidence} />
      </div>

      <div className={styles.section}>
        <div className={styles.sectionLabel}>Predicted bounding box</div>
        <div className={styles.row}>
          <span className={styles.key}>x range</span>
          <span className={styles.value}>
            {fmt(prediction.bbox.xmin)} - {fmt(prediction.bbox.xmax)}
          </span>
        </div>
        <div className={styles.row}>
          <span className={styles.key}>y range</span>
          <span className={styles.value}>
            {fmt(prediction.bbox.ymin)} - {fmt(prediction.bbox.ymax)}
          </span>
        </div>
        <div className={styles.row}>
          <span className={styles.key}>width</span>
          <span className={styles.value}>
            {fmt(prediction.bbox.xmax - prediction.bbox.xmin)} px
          </span>
        </div>
        <div className={styles.row}>
          <span className={styles.key}>height</span>
          <span className={styles.value}>
            {fmt(prediction.bbox.ymax - prediction.bbox.ymin)} px
          </span>
        </div>
      </div>

      {gt && (
        <div className={styles.section}>
          <div className={styles.sectionLabel}>Ground truth bounding box</div>
          <div className={styles.row}>
            <span className={styles.key}>x range</span>
            <span className={styles.value}>
              {fmt(gt.xmin)} - {fmt(gt.xmax)}
            </span>
          </div>
          <div className={styles.row}>
            <span className={styles.key}>y range</span>
            <span className={styles.value}>
              {fmt(gt.ymin)} - {fmt(gt.ymax)}
            </span>
          </div>
        </div>
      )}

      <div className={styles.section}>
        <div className={styles.sectionLabel}>Inference</div>
        <div className={styles.row}>
          <span className={styles.key}>latency</span>
          <span className={styles.value}>{fmt(prediction.inference_ms, 0)} ms</span>
        </div>
        <div className={styles.row}>
          <span className={styles.key}>image size</span>
          <span className={styles.value}>
            {prediction.image_width} x {prediction.image_height}
          </span>
        </div>
      </div>
    </aside>
  );
}

// Infer which side the GT bbox is off-screen on, for a side-accuracy display.
function inferSideFromBBox(bbox) {
  if (bbox.xmin < 0) return 0;
  if (bbox.xmax > 720) return 1;
  // Edge case: bbox fully on-screen (shouldn't happen for this dataset but
  // be defensive). Pick the closer edge.
  return bbox.xmin < 720 - bbox.xmax ? 0 : 1;
}
