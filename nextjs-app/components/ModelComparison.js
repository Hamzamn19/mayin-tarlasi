'use client';

import React, { useMemo } from 'react';
import { AnimatePresence } from 'framer-motion';
import LinearStage from './pipeline/LinearStage';
import ForestStage from './pipeline/ForestStage';

export default function ModelComparison({ activeBox, featureKeys, mode }) {
  const lrWeights = {
    area: 0.0009, circularity: 3.4431, thermal_contrast: 0.0952,
    aspect_ratio: 0.7994, rts: 1.4648,
    log_zero_crossings: -0.8601, lbp_ri: 7.1652, dct_high_energy: 305.6818,
    wavelet_approx: -1.7360, hole_count: -0.0598, bias: -1.5325
  };
  
  const lrScore = useMemo(() => {
    let score = lrWeights.bias;
    featureKeys.forEach(([label, key]) => {
      const val = activeBox.features[key] || 0;
      score += val * lrWeights[key];
    });
    return score;
  }, [activeBox, featureKeys]);

  return (
    <div className="h-full w-full p-8 bg-white text-slate-900">
      <AnimatePresence mode="wait">
        {mode === 'lr' && (
          <LinearStage activeBox={activeBox} featureKeys={featureKeys} lrWeights={lrWeights} lrScore={lrScore} />
        )}
        {mode === 'rf' && (
          <ForestStage activeBox={activeBox} />
        )}
      </AnimatePresence>
    </div>
  );
}
