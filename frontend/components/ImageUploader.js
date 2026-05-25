"use client";

import { useState, useRef, useEffect } from "react";
import styles from "./ImageUploader.module.css";

export default function ImageUploader({ onSelect, activeId }) {
  const inputRef = useRef(null);
  const [dragActive, setDragActive] = useState(false);
  const [samples, setSamples] = useState([]);

  useEffect(() => {
    fetch("/samples/samples.json")
      .then((r) => (r.ok ? r.json() : []))
      .then(setSamples)
      .catch(() => setSamples([]));
  }, []);

  function handleFile(file) {
    if (!file || !file.type.startsWith("image/")) return;
    onSelect({
      id: `upload-${Date.now()}`,
      kind: "upload",
      file,
      previewUrl: URL.createObjectURL(file),
      groundTruth: null,
      label: file.name,
    });
  }

  function handleDrop(e) {
    e.preventDefault();
    setDragActive(false);
    handleFile(e.dataTransfer.files?.[0]);
  }

  function handleSample(sample) {
    onSelect({
      id: sample.id,
      kind: "sample",
      file: null,
      previewUrl: `/samples/${sample.file}`,
      groundTruth: sample.bbox
        ? {
            xmin: sample.bbox.xmin,
            ymin: sample.bbox.ymin,
            xmax: sample.bbox.xmax,
            ymax: sample.bbox.ymax,
            direction: sample.direction,
          }
        : null,
      label: sample.id,
    });
  }

  return (
    <div className={styles.uploader}>
      <div
        className={`${styles.dropzone} ${dragActive ? styles.dropzoneActive : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragActive(true);
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
      >
        <div className={styles.dropzoneText}>
          Drop an image here or click to upload
        </div>
        <div className={styles.dropzoneHint}>
          PNG or JPG, up to 10 MB
        </div>
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className={styles.hidden}
          onChange={(e) => handleFile(e.target.files?.[0])}
        />
      </div>

      {samples.length > 0 && (
        <>
          <div className={styles.galleryTitle}>
            Or pick a held-out sample (with ground truth)
          </div>
          <div className={styles.gallery}>
            {samples.map((s) => (
              <button
                key={s.id}
                type="button"
                className={`${styles.galleryItem} ${
                  activeId === s.id ? styles.galleryItemActive : ""
                }`}
                onClick={() => handleSample(s)}
              >
                <img src={`/samples/${s.file}`} alt={s.id} />
                <span className={styles.galleryLabel}>{s.id}</span>
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
