'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { featureKeys, imageMeta as fallbackMeta, sampleDetections, stages } from './sampleData';

const MAX_SIDE = 1024;
const placeholderImage = createPlaceholderImage();

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function formatPercent(value) {
  return `${Math.round(value * 100)}%`;
}

function formatNumber(value, digits = 1) {
  return Number.isFinite(value) ? value.toFixed(digits) : '0.0';
}

function createPlaceholderImage() {
  const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 720" width="960" height="720">
      <defs>
        <linearGradient id="bg" x1="0" x2="1" y1="0" y2="1">
          <stop offset="0%" stop-color="#0f172a" />
          <stop offset="55%" stop-color="#1e293b" />
          <stop offset="100%" stop-color="#020617" />
        </linearGradient>
        <radialGradient id="glow1" cx="30%" cy="32%" r="32%">
          <stop offset="0%" stop-color="rgba(59,130,246,0.88)" />
          <stop offset="100%" stop-color="rgba(59,130,246,0)" />
        </radialGradient>
        <radialGradient id="glow2" cx="68%" cy="62%" r="28%">
          <stop offset="0%" stop-color="rgba(245,158,11,0.82)" />
          <stop offset="100%" stop-color="rgba(245,158,11,0)" />
        </radialGradient>
        <radialGradient id="glow3" cx="50%" cy="50%" r="24%">
          <stop offset="0%" stop-color="rgba(239,68,68,0.5)" />
          <stop offset="100%" stop-color="rgba(239,68,68,0)" />
        </radialGradient>
      </defs>
      <rect width="960" height="720" fill="url(#bg)" />
      <rect width="960" height="720" fill="url(#glow1)" />
      <rect width="960" height="720" fill="url(#glow2)" />
      <rect width="960" height="720" fill="url(#glow3)" />
      <g opacity="0.22" stroke="white">
        <path d="M0 90H960M0 180H960M0 270H960M0 360H960M0 450H960M0 540H960M0 630H960" />
        <path d="M96 0V720M192 0V720M288 0V720M384 0V720M480 0V720M576 0V720M672 0V720M768 0V720M864 0V720" />
      </g>
      <text x="50%" y="48%" text-anchor="middle" fill="white" font-family="Arial, sans-serif" font-size="34" font-weight="700">Upload a LWIR image</text>
      <text x="50%" y="54%" text-anchor="middle" fill="rgba(255,255,255,0.72)" font-family="Arial, sans-serif" font-size="18">The full pipeline becomes interactive after file upload</text>
    </svg>
  `;
  return `data:image/svg+xml;charset=UTF-8,${encodeURIComponent(svg)}`;
}

function readFileAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error || new Error('Failed to read file'));
    reader.readAsDataURL(file);
  });
}

function loadImage(dataUrl) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error('Could not load image'));
    image.src = dataUrl;
  });
}

function canvasToDataUrl(canvas) {
  return canvas.toDataURL('image/jpeg', 0.92);
}

function cropCanvasDataUrl(sourceCanvas, x, y, width, height) {
  const cropCanvas = document.createElement('canvas');
  cropCanvas.width = Math.max(1, Math.round(width));
  cropCanvas.height = Math.max(1, Math.round(height));
  const context = cropCanvas.getContext('2d');
  context.drawImage(sourceCanvas, x, y, width, height, 0, 0, cropCanvas.width, cropCanvas.height);
  return cropCanvas.toDataURL('image/jpeg', 0.9);
}

async function analyzeFile(file) {
  const dataUrl = await readFileAsDataUrl(file);
  const image = await loadImage(dataUrl);
  const sourceWidth = image.naturalWidth || image.width;
  const sourceHeight = image.naturalHeight || image.height;
  const scale = Math.min(1, MAX_SIDE / Math.max(sourceWidth, sourceHeight));
  const width = Math.max(1, Math.round(sourceWidth * scale));
  const height = Math.max(1, Math.round(sourceHeight * scale));

  const sourceCanvas = document.createElement('canvas');
  sourceCanvas.width = width;
  sourceCanvas.height = height;
  const sourceContext = sourceCanvas.getContext('2d');
  sourceContext.drawImage(image, 0, 0, width, height);

  const originalPreview = canvasToDataUrl(sourceCanvas);
  const imageData = sourceContext.getImageData(0, 0, width, height);
  const grayValues = new Uint8ClampedArray(width * height);
  const processedCanvas = document.createElement('canvas');
  processedCanvas.width = width;
  processedCanvas.height = height;
  const processedContext = processedCanvas.getContext('2d');
  const processedData = processedContext.createImageData(width, height);

  for (let index = 0; index < imageData.data.length; index += 4) {
    const red = imageData.data[index];
    const green = imageData.data[index + 1];
    const blue = imageData.data[index + 2];
    const gray = Math.round(red * 0.299 + green * 0.587 + blue * 0.114);
    const pixelIndex = index / 4;
    grayValues[pixelIndex] = gray;
    processedData.data[index] = gray;
    processedData.data[index + 1] = gray;
    processedData.data[index + 2] = gray;
    processedData.data[index + 3] = 255;
  }

  processedContext.putImageData(processedData, 0, 0);
  const processedPreview = canvasToDataUrl(processedCanvas);

  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch('http://localhost:8000/detect', {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) throw new Error('Failed to fetch detections from API');
  const data = await response.json();
  
  const detections = data.detections.map(det => {
    const boxWidth = (det.x2 - det.x1) * width / 100;
    const boxHeight = (det.y2 - det.y1) * height / 100;
    const startX = det.x1 * width / 100;
    const startY = det.y1 * height / 100;
    const cropDataUrl = cropCanvasDataUrl(sourceCanvas, startX, startY, boxWidth, boxHeight);
    return { ...det, cropDataUrl };
  });

  return {
    fileName: file.name,
    fileSizeKB: file.size / 1024,
    sourceWidth,
    sourceHeight,
    width,
    height,
    originalPreview,
    processedPreview,
    detections,
    fallback: false,
  };
}

function initialAnalysis() {
  return {
    fileName: 'demo-lwir.png',
    fileSizeKB: fallbackMeta.fileSizeKB,
    sourceWidth: fallbackMeta.width,
    sourceHeight: fallbackMeta.height,
    width: fallbackMeta.width,
    height: fallbackMeta.height,
    originalPreview: placeholderImage,
    processedPreview: placeholderImage,
    detections: sampleDetections,
    fallback: true,
  };
}

function stageLabel(stage) {
  return stages.find((item) => item.id === stage)?.title || 'Upload';
}

export default function Dashboard() {
  const [stage, setStage] = useState(1);
  const [analysis, setAnalysis] = useState(() => initialAnalysis());
  const [activeBox, setActiveBox] = useState(sampleDetections[0]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const inputRef = useRef(null);
  const imageRef = useRef(null);

  useEffect(() => {
    if (!analysis.detections.length) return;
    setActiveBox((current) => {
      if (current && analysis.detections.some((box) => box.id === current.id)) return current;
      return analysis.detections[0];
    });
  }, [analysis.detections]);

  const showBoxes = stage >= 4 && analysis.detections.length > 0;

  const navigate = (delta) => {
    setStage((current) => clamp(current + delta, 1, 7));
  };

  const handleFile = async (file) => {
    if (!file) return;
    setLoading(true);
    setError('');
    try {
      const result = await analyzeFile(file);
      setAnalysis(result);
      setActiveBox(result.detections[0] || null);
      setStage(4);
    } catch (exception) {
      setError(exception?.message || 'Failed to process image');
      setAnalysis(initialAnalysis());
      setActiveBox(sampleDetections[0]);
    } finally {
      setLoading(false);
    }
  };

  const openPicker = () => inputRef.current?.click();
  const currentStage = useMemo(() => stages.find((item) => item.id === stage) || stages[0], [stage]);
  const imageAspectRatio = analysis.width / analysis.height;

  return (
    <div className="page" style={{ height: '100vh', display: 'flex', flexDirection: 'column', padding: '12px', overflow: 'hidden' }}>
      <style jsx global>{`
        body, html { height: 100vh; overflow: hidden; background-color: var(--bg0); }
        .shell { flex: 1; display: flex; flex-direction: column; overflow: hidden; min-height: 0; }
        .content { flex: 1; display: flex; flex-direction: column; overflow: hidden; min-height: 0; padding: 12px 24px 24px; }
        .grid-2 { flex: 1; display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(320px, 0.75fr); gap: 18px; overflow: hidden; min-height: 0; }
        .panel { display: flex; flex-direction: column; overflow: hidden; }
        .panel-scroll { flex: 1; overflow-y: auto; padding-right: 4px; }
        .panel-scroll::-webkit-scrollbar { width: 6px; }
        .panel-scroll::-webkit-scrollbar-thumb { background: var(--border); border-radius: 10px; }
        .thermal-stage-wrap { flex: 1; display: flex; align-items: center; justify-content: center; overflow: hidden; min-height: 0; padding: 10px; }
        .thermal-stage { position: relative; width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; overflow: hidden; background: transparent; }
        .thermal-stage-inner { position: relative; max-width: 100%; max-height: 100%; box-shadow: 0 0 40px rgba(0,0,0,0.5); border-radius: 12px; overflow: hidden; }
        .thermal-stage-inner img { width: 100%; height: 100%; display: block; object-fit: contain; }
      `}</style>

      <div className="shell">
        <div className="header" style={{ padding: '12px 24px', alignItems: 'center', borderBottom: '1px solid var(--border)' }}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: '12px', flexWrap: 'wrap' }}>
            <h1 style={{ fontSize: '1.5rem', margin: 0 }}>Mine Explainer</h1>
            <span style={{ fontSize: '0.85rem', color: 'var(--muted)', fontWeight: 500 }}>A Hybrid Deep Learning & Statistical Analysis Pipeline for Detecting Buried Landmines</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
             <div className="chip" style={{ padding: '4px 12px', fontSize: '0.85rem' }}>
              <strong>Stage {stage}</strong>: {stageLabel(stage)}
            </div>
          </div>
        </div>

        <div className="stage-strip" style={{ padding: '8px 24px', gap: '4px', borderBottom: '1px solid var(--border)' }}>
          <button className="stage-arrow" style={{ width: '24px', height: '24px', fontSize: '0.8rem' }} onClick={() => navigate(-1)} type="button">‹</button>
          {stages.map((item) => (
            <button
              key={item.id}
              className={`stage-pill ${stage === item.id ? 'active' : ''}`}
              style={{ padding: '4px 12px', fontSize: '0.75rem' }}
              onClick={() => setStage(item.id)}
              type="button"
            >
              {item.id}. {item.title}
            </button>
          ))}
          <button className="stage-arrow" style={{ width: '24px', height: '24px', fontSize: '0.8rem' }} onClick={() => navigate(1)} type="button">›</button>
        </div>

        <div className="content" style={{ padding: '12px 24px', overflow: 'hidden' }}>
          
          {stage === 1 && (
            <section className="panel" style={{ flex: 1, width: '100%', overflow: 'hidden', padding: '0px', border: 'none', background: 'transparent', boxShadow: 'none' }}>
              <div className="panel-title" style={{ padding: '12px 0', marginBottom: '0' }}>
                <div>
                  <h3 style={{ fontSize: '1.2rem', margin: 0 }}>{currentStage.title}</h3>
                  <p style={{ fontSize: '0.9rem', margin: 0 }}>{currentStage.subtitle}</p>
                </div>
                <span className="badge" style={{ padding: '4px 12px', fontSize: '0.8rem' }}>S{stage}</span>
              </div>
              <div className="panel-scroll" style={{ overflow: 'auto', display: 'flex', flexDirection: 'column' }}>
                <div className="hero-grid" style={{ gridTemplateColumns: '1fr', padding: '16px 0' }}>
                  <div className="upload-box card" style={{ padding: '16px' }}>
                    <div className="upload-area" style={{ padding: '24px', textAlign: 'center' }}>
                      <div className="small-label">Upload Image</div>
                      <div className="controls" style={{ marginTop: 20, justifyContent: 'center' }}>
                        <button className="button" style={{ padding: '8px 16px', fontSize: '1rem' }} type="button" onClick={openPicker} disabled={loading}>
                          {loading ? 'Analyzing...' : 'Choose image'}
                        </button>
                        {analysis.fileName ? <span className="chip" style={{ fontSize: '0.85rem', padding: '4px 12px' }}>{analysis.fileName}</span> : null}
                      </div>
                      <input
                        ref={inputRef}
                        type="file"
                        accept="image/*"
                        style={{ display: 'none' }}
                        onChange={(event) => handleFile(event.target.files?.[0])}
                      />
                      <div className="upload-meta" style={{ marginTop: 20, gap: '12px', justifyContent: 'center', display: 'flex' }}>
                        <div className="metric" style={{ padding: '8px 16px' }}><div className="k" style={{ fontSize: '0.7rem' }}>Res</div><div className="v" style={{ fontSize: '0.9rem' }}>{analysis.sourceWidth}×{analysis.sourceHeight}</div></div>
                        <div className="metric" style={{ padding: '8px 16px' }}><div className="k" style={{ fontSize: '0.7rem' }}>Size</div><div className="v" style={{ fontSize: '0.9rem' }}>{formatNumber(analysis.fileSizeKB, 0)}KB</div></div>
                        <div className="metric" style={{ padding: '8px 16px' }}><div className="k" style={{ fontSize: '0.7rem' }}>Mines</div><div className="v" style={{ fontSize: '0.9rem' }}>{analysis.detections.length}</div></div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </section>
          )}

          {stage === 2 && (
             <section className="panel" style={{ flex: 1, width: '100%', overflow: 'hidden', padding: '0px', border: 'none', background: 'transparent', boxShadow: 'none' }}>
                <div className="panel-title" style={{ padding: '12px 0', marginBottom: '0' }}>
                  <div>
                    <h3 style={{ fontSize: '1.2rem', margin: 0 }}>{currentStage.title}</h3>
                    <p style={{ fontSize: '0.9rem', margin: 0 }}>{currentStage.subtitle}</p>
                  </div>
                  <span className="badge" style={{ padding: '4px 12px', fontSize: '0.8rem' }}>S{stage}</span>
                </div>
                <div className="grid-2" style={{ padding: '16px 0' }}>
                  <div className="panel">
                    <div className="panel-title"><h3 style={{ fontSize: '1rem' }}>Original Frame</h3></div>
                    <div style={{ flex: 1, display: 'flex', alignItems: 'center', background: '#000', borderRadius: '12px', overflow: 'hidden' }}>
                      <img src={analysis.originalPreview} style={{ width: '100%', objectFit: 'contain' }} />
                    </div>
                  </div>
                  <div className="panel">
                    <div className="panel-title"><h3 style={{ fontSize: '1rem' }}>Grayscale Preprocessed</h3></div>
                    <div style={{ flex: 1, display: 'flex', alignItems: 'center', background: '#000', borderRadius: '12px', overflow: 'hidden' }}>
                      <img src={analysis.processedPreview} style={{ width: '100%', objectFit: 'contain' }} />
                    </div>
                  </div>
                </div>
             </section>
          )}

          {stage === 3 && (
             <section className="panel" style={{ flex: 1, width: '100%', overflow: 'hidden', padding: '0px', border: 'none', background: 'transparent', boxShadow: 'none' }}>
                <div className="panel-title" style={{ padding: '12px 0', marginBottom: '0' }}>
                  <div>
                    <h3 style={{ fontSize: '1.2rem', margin: 0 }}>{currentStage.title}</h3>
                    <p style={{ fontSize: '0.9rem', margin: 0 }}>{currentStage.subtitle}</p>
                  </div>
                  <span className="badge" style={{ padding: '4px 12px', fontSize: '0.8rem' }}>S{stage}</span>
                </div>
                <div className="panel" style={{ flex: 1, position: 'relative', overflow: 'hidden', background: '#000', borderRadius: '16px', marginTop: '16px' }}>
                  <div style={{ position: 'relative', width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                     <div style={{ position: 'relative', maxWidth: '100%', maxHeight: '100%', aspectRatio: imageAspectRatio }}>
                        <div className="scan-line" style={{ animation: 'scan 2s linear infinite' }} />
                        <img src={analysis.originalPreview} style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
                     </div>
                  </div>
                  <style>{`@keyframes scan { 0% { top: 0%; } 100% { top: 100%; } }`}</style>
                </div>
             </section>
          )}

          {stage >= 4 && (
            <div className="grid-2">
              <div className="panel" style={{ padding: '0', background: 'transparent', border: 'none', boxShadow: 'none' }}>
                <div className="thermal-stage-wrap">
                  <div className="thermal-stage">
                    <div className="thermal-stage-inner" style={{ aspectRatio: imageAspectRatio }}>
                      {stage === 3 ? <div className="scan-line" /> : null}
                      <img ref={imageRef} src={analysis.originalPreview} alt="Pipeline preview" />
                      {showBoxes && analysis.detections.map((box) => (
                        <button
                          key={box.id}
                          className={`box ${box.ensemble_pred === 1 ? '' : 'safe'}`}
                          style={{
                            left: `${box.x1}%`,
                            top: `${box.y1}%`,
                            width: `${box.x2 - box.x1}%`,
                            height: `${box.y2 - box.y1}%`,
                            borderWidth: activeBox?.id === box.id ? '3px' : '2px',
                            borderColor: activeBox?.id === box.id ? 'var(--accent)' : (box.ensemble_pred === 1 ? 'var(--danger)' : 'var(--success)'),
                            background: activeBox?.id === box.id ? 'rgba(255,255,255,0.2)' : ''
                          }}
                          onClick={() => {
                            setActiveBox(box);
                          }}
                          type="button"
                        >
                          <span className="tag" style={{ fontSize: '0.65rem', padding: '2px 6px', top: '-12px', background: activeBox?.id === box.id ? 'var(--accent)' : '' }}>#{box.id} {box.label}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              <div className="panel panel-scroll">
                <div className="panel-title" style={{ padding: '12px 0', marginBottom: '12px' }}>
                  <div>
                    <h3 style={{ fontSize: '1.2rem', margin: 0 }}>{currentStage.title}</h3>
                    <p style={{ fontSize: '0.9rem', margin: 0 }}>{currentStage.subtitle}</p>
                  </div>
                  <span className="badge" style={{ padding: '4px 12px', fontSize: '0.8rem' }}>S{stage}</span>
                </div>

                <div style={{ padding: '0' }}>
                {activeBox ? (
                  <div className="details">
                    <div className="card" style={{ marginBottom: '16px', overflow: 'hidden' }}>
                      <div className="crop-preview">
                        <img src={activeBox.cropDataUrl} style={{ width: '100%', maxHeight: '200px', objectFit: 'cover', imageRendering: 'pixelated' }} />
                      </div>
                      <div style={{ padding: '12px' }}>
                        <div className="badge">Selected: Detection #{activeBox.id}</div>
                      </div>
                    </div>

                    {stage === 4 && (
                      <div className="list">
                        <h4 style={{ margin: '0 0 12px 0', fontSize: '1rem' }}>Detections List</h4>
                        {analysis.detections.map(box => (
                          <div 
                            key={box.id} 
                            className={`metric ${activeBox?.id === box.id ? 'active' : ''}`}
                            onClick={() => setActiveBox(box)}
                            style={{ cursor: 'pointer', marginBottom: '8px', border: activeBox?.id === box.id ? '1px solid var(--accent)' : '', background: activeBox?.id === box.id ? 'var(--accent-soft)' : '' }}
                          >
                            <div className="k" style={{ fontSize: '0.75rem' }}>#{box.id} {box.label}</div>
                            <div className="v" style={{ fontSize: '0.9rem' }}>Confidence: {formatPercent(box.conf)}</div>
                          </div>
                        ))}
                      </div>
                    )}

                    {stage === 5 && (
                      <div className="feature-grid">
                        {featureKeys.map(([label, key, unit]) => (
                          <div key={key} className="feature-pill">
                            <div className="k" style={{ fontSize: '0.7rem' }}>{label}</div>
                            <div className="v" style={{ fontSize: '0.95rem' }}>{formatNumber(activeBox.features[key])}{unit}</div>
                          </div>
                        ))}
                      </div>
                    )}

                    {stage === 6 && (
                      <div className="bar-group">
                        <div className="bar-item">
                          <div className="bar-header"><span>YOLO Confidence</span><span>{formatPercent(activeBox.conf)}</span></div>
                          <div className="bar-bg"><div className="bar-fill lr" style={{ width: formatPercent(activeBox.conf) }} /></div>
                        </div>
                        <div className="bar-item">
                          <div className="bar-header"><span>Logistic Regression</span><span>{formatPercent(activeBox.lr_prob)}</span></div>
                          <div className="bar-bg"><div className="bar-fill lr" style={{ width: formatPercent(activeBox.lr_prob) }} /></div>
                        </div>
                        <div className="bar-item">
                          <div className="bar-header"><span>Random Forest</span><span>{formatPercent(activeBox.rf_prob)}</span></div>
                          <div className="bar-bg"><div className="bar-fill rf" style={{ width: formatPercent(activeBox.rf_prob) }} /></div>
                        </div>
                      </div>
                    )}

                    {stage === 7 && (
                      <div className="vote-grid" style={{ gridTemplateColumns: '1fr' }}>
                        <div className={`final-verdict ${activeBox.ensemble_pred === 1 ? 'mine' : 'safe'}`} style={{ textAlign: 'center', padding: '32px 24px' }}>
                          <div style={{ fontSize: '3rem', marginBottom: '12px' }}>{activeBox.ensemble_pred === 1 ? '💣' : '✅'}</div>
                          <div style={{ fontSize: '1.8rem', fontWeight: 'bold' }}>{activeBox.ensemble_pred === 1 ? 'DANGER: MINE' : 'CLEAR: SAFE'}</div>
                          <div style={{ marginTop: '12px', opacity: 0.8, fontSize: '1.1rem' }}>Ensemble Probability: {formatPercent(activeBox.ensemble_prob)}</div>
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="empty-state">No detections selected or available.</div>
                )}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
