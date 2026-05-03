'use client';

import { usePipelineStore } from '@/store/pipeline';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, ChevronRight, ChevronLeft, Target, Cpu, Activity, ShieldCheck } from 'lucide-react';
import { useRef } from 'react';
import { stages } from '@/components/sampleData';

export default function ProfessionalDashboard() {
  const { stage, setStage, loading, setLoading, originalPreview, detections, setImage, setDetections } = usePipelineStore();
  const fileInputRef = useRef(null);

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    const reader = new FileReader();
    reader.onload = (e) => setImage({ originalPreview: e.target.result });
    reader.readAsDataURL(file);

    try {
      const res = await fetch('http://localhost:8000/detect', { method: 'POST', body: formData });
      const data = await res.json();
      setDetections(data.detections);
      setStage(2);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-[#020617] text-slate-200 overflow-hidden font-sans">
      {/* Compact Header */}
      <header className="flex items-center justify-between px-6 py-2 border-b border-slate-800 bg-[#020617]/80 backdrop-blur-md z-50">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-red-600 rounded flex items-center justify-center shadow-lg shadow-red-900/20">
            <Target className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-tight text-white">MINE EXPLORER <span className="text-red-500 text-xs ml-1">v2.0</span></h1>
          </div>
        </div>
        
        <div className="flex items-center gap-4">
          <div className="flex bg-slate-900/50 rounded-full p-1 border border-slate-800">
            {[1, 2, 3, 4, 5, 6, 7].map((s) => (
              <button
                key={s}
                onClick={() => setStage(s)}
                className={`w-7 h-7 rounded-full text-xs font-bold transition-all ${
                  stage === s ? 'bg-red-600 text-white' : 'text-slate-500 hover:text-slate-300'
                }`}
              >
                {s}
              </button>
            ))}
          </div>
        </div>
      </header>

      {/* Main Analysis Area */}
      <main className="flex-1 relative flex items-center justify-center p-4 overflow-hidden">
        <AnimatePresence mode="wait">
          {stage === 1 ? (
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="max-w-xl w-full"
            >
              <div 
                onClick={() => fileInputRef.current.click()}
                className="group relative border-2 border-dashed border-slate-800 hover:border-red-500/50 rounded-3xl p-12 transition-all cursor-pointer bg-slate-900/20 hover:bg-red-500/5"
              >
                <input type="file" ref={fileInputRef} onChange={handleUpload} className="hidden" />
                <div className="flex flex-col items-center gap-4">
                  <div className="w-16 h-16 rounded-full bg-slate-900 flex items-center justify-center group-hover:scale-110 transition-transform">
                    <Upload className="w-8 h-8 text-slate-400 group-hover:text-red-500" />
                  </div>
                  <div className="text-center">
                    <h2 className="text-xl font-bold text-white mb-2">Initialize LWIR Pipeline</h2>
                    <p className="text-slate-400 text-sm">Upload a thermal frame to begin neural scanning</p>
                  </div>
                </div>
              </div>
            </motion.div>
          ) : (
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="relative w-full h-full flex items-center justify-center"
            >
              <div className="relative max-h-full max-w-full shadow-2xl shadow-black/50 border border-slate-800 rounded-lg overflow-hidden bg-black">
                <img 
                  src={originalPreview} 
                  alt="Thermal analysis" 
                  className="max-h-[80vh] w-auto block"
                />
                
                {/* YOLO Bounding Boxes Layer */}
                {stage >= 2 && (
                  <div className="absolute inset-0 pointer-events-none">
                    {detections.map((det, i) => (
                      <motion.div
                        key={i}
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: i * 0.1 }}
                        style={{
                          left: `${det.x1}%`,
                          top: `${det.y1}%`,
                          width: `${det.x2 - det.x1}%`,
                          height: `${det.y2 - det.y1}%`,
                        }}
                        className="absolute border-2 border-red-500 bg-red-500/10"
                      >
                        <div className="absolute -top-6 left-0 bg-red-600 text-white text-[10px] font-bold px-1.5 py-0.5 rounded-sm flex items-center gap-1">
                          <Activity className="w-3 h-3" />
                          MINE {Math.round(det.conf * 100)}%
                        </div>
                      </motion.div>
                    ))}
                  </div>
                )}

                {/* Scan Line Animation */}
                {stage === 3 && (
                  <motion.div 
                    initial={{ top: '0%' }}
                    animate={{ top: '100%' }}
                    transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                    className="absolute left-0 right-0 h-1 bg-red-500 shadow-[0_0_15px_rgba(239,68,68,0.8)] z-10"
                  />
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Floating Stage Controls */}
        <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex items-center gap-4 bg-slate-900/80 backdrop-blur-xl border border-slate-800 p-2 rounded-2xl shadow-2xl">
          <button 
            onClick={() => setStage(Math.max(1, stage - 1))}
            className="p-3 hover:bg-slate-800 rounded-xl transition-colors text-slate-400 hover:text-white"
          >
            <ChevronLeft className="w-6 h-6" />
          </button>
          
          <div className="px-4 py-1 text-center border-x border-slate-800">
            <div className="text-[10px] uppercase tracking-widest text-slate-500 font-bold mb-0.5">Stage {stage}</div>
            <div className="text-sm font-bold text-white">{stages[stage - 1]?.title}</div>
          </div>

          <button 
            onClick={() => setStage(Math.min(7, stage + 1))}
            className="p-3 hover:bg-slate-800 rounded-xl transition-colors text-slate-400 hover:text-white"
          >
            <ChevronRight className="w-6 h-6" />
          </button>
        </div>
      </main>

      {/* Global Processing State (Bottom Bar) */}
      <footer className="px-6 py-2 bg-[#020617] border-t border-slate-800 flex justify-between items-center text-[10px] font-mono text-slate-500 uppercase tracking-tighter">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse"></span>
            BACKEND: CONNECTED (PORT 8000)
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-blue-500"></span>
            YOLOv26: ACTIVE
          </div>
        </div>
        <div className="flex items-center gap-4">
           {loading && <div className="text-red-500 animate-pulse">PROCESSING_NEURAL_FRAME...</div>}
           <div>LATENCY: 4.9MS</div>
           <div>DEVICE: CUDA_0</div>
        </div>
      </footer>
    </div>
  );
}
