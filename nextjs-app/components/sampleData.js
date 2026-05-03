export const stages = [
  { id: 1, title: 'Upload', subtitle: 'Input frame and metadata' },
  { id: 2, title: 'Preprocess', subtitle: 'Resize and normalize' },
  { id: 3, title: 'Grid Scan', subtitle: 'Thermal sweep' },
  { id: 4, title: 'BBoxes', subtitle: 'Candidate detections' },
  { id: 5, title: 'Inspector', subtitle: 'Crop and features' },
  { id: 6, title: 'Classify', subtitle: 'LR and RF scores' },
  { id: 7, title: 'Vote', subtitle: 'Final verdict' },
];

export const sampleDetections = [
  {
    id: 0,
    label: 'Mine',
    conf: 0.91,
    lr_prob: 0.88,
    rf_prob: 0.94,
    ensemble_prob: 0.93,
    ensemble_pred: 1,
    x1: 18,
    y1: 16,
    x2: 43,
    y2: 38,
    features: {
      area: 845.2,
      circularity: 0.78,
      mean_intensity: 168.4,
      thermal_contrast: 27.2,
      edge_density: 0.0412,
    },
  },
  {
    id: 1,
    label: 'Background',
    conf: 0.77,
    lr_prob: 0.21,
    rf_prob: 0.28,
    ensemble_prob: 0.53,
    ensemble_pred: 1,
    x1: 56,
    y1: 48,
    x2: 79,
    y2: 71,
    features: {
      area: 642.8,
      circularity: 0.41,
      mean_intensity: 142.1,
      thermal_contrast: 12.7,
      edge_density: 0.0328,
    },
  },
  {
    id: 2,
    label: 'Safe',
    conf: 0.69,
    lr_prob: 0.19,
    rf_prob: 0.26,
    ensemble_prob: 0.48,
    ensemble_pred: 0,
    x1: 68,
    y1: 17,
    x2: 90,
    y2: 39,
    features: {
      area: 521.7,
      circularity: 0.29,
      mean_intensity: 119.2,
      thermal_contrast: 8.5,
      edge_density: 0.0189,
    },
  },
];

export const featureKeys = [
  ['Area', 'area', 'px²'],
  ['Circularity', 'circularity', ''],
  ['Mean intensity', 'mean_intensity', ''],
  ['Thermal contrast', 'thermal_contrast', ''],
  ['Edge density', 'edge_density', ''],
];

export const imageMeta = {
  width: 1280,
  height: 720,
  fileSizeKB: 812.4,
};
