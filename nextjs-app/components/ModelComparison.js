'use client';

import React, { useMemo } from 'react';
import { AnimatePresence } from 'framer-motion';
import LinearStage from './pipeline/LinearStage';
import ForestStage from './pipeline/ForestStage';

export default function ModelComparison({ activeBox, featureKeys, mode }) {
  const lrWeights = { area: 1.2, circularity: 2.5, mean_intensity: 0.8, thermal_contrast: 3.1, edge_density: 1.5, bias: -4.2 };
  
  const lrScore = useMemo(() => {
    let score = lrWeights.bias;
    featureKeys.forEach(([label, key]) => {
      const val = (key === 'area' ? activeBox.features[key] / 1000 : key === 'mean_intensity' ? activeBox.features[key] / 255 : activeBox.features[key]);
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
