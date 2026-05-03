import { create } from 'zustand';

export const usePipelineStore = create((set) => ({
  stage: 1,
  image: null,
  originalPreview: null,
  processedPreview: null,
  detections: [],
  selectedBox: null,
  loading: false,
  error: null,
  analysis: null,

  setStage: (stage) => set({ stage }),
  setImage: (data) => set({ ...data }),
  setDetections: (detections) => set({ detections }),
  setSelectedBox: (box) => set({ selectedBox: box }),
  setLoading: (loading) => set({ loading }),
  setError: (error) => set({ error }),
  
  reset: () => set({
    stage: 1,
    image: null,
    detections: [],
    selectedBox: null,
    loading: false,
    error: null
  })
}));
