'use client';

import React, { useMemo } from 'react';
import { motion } from 'framer-motion';

// Correct sigmoid function
function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

const FEATURE_CONFIG = [
  { label: 'Area',             key: 'area',              color: '#ef4444' },
  { label: 'Circularity',      key: 'circularity',       color: '#f97316' },
  { label: 'Mean Intensity',   key: 'mean_intensity',    color: '#8b5cf6' },
  { label: 'Thermal Contrast', key: 'thermal_contrast',  color: '#3b82f6' },
  { label: 'Edge Density',     key: 'edge_density',      color: '#10b981' },
];

const DEFAULT_WEIGHTS = {
  area: 1.2,
  circularity: 2.5,
  mean_intensity: 0.8,
  thermal_contrast: 3.1,
  edge_density: 1.5,
};

export default function LinearStage({ activeBox, lrWeights = DEFAULT_WEIGHTS }) {
  const features = activeBox?.features ?? {
    area: 850,
    circularity: 0.71,
    mean_intensity: 162,
    thermal_contrast: 0.44,
    edge_density: 0.22,
  };

  // Normalize features to 0–1 range
  const normalizedFeatures = {
    area:             features.area / 1000,
    circularity:      features.circularity,
    mean_intensity:   features.mean_intensity / 255,
    thermal_contrast: features.thermal_contrast,
    edge_density:     features.edge_density,
  };

  const contributions = useMemo(() =>
    FEATURE_CONFIG.map(({ label, key, color }) => {
      const val = normalizedFeatures[key] ?? 0;
      const weight = lrWeights[key] ?? 1;
      const contribution = val * weight;
      return { label, key, val, weight, contribution, color };
    }),
    [features, lrWeights]
  );

  const bias = -4.2;
  const lrScore = contributions.reduce((sum, f) => sum + f.contribution, bias);
  const lrProb = sigmoid(lrScore);
  const isLandmine = lrProb > 0.5;

  // Max contribution for bar scaling
  const maxContrib = Math.max(...contributions.map(f => Math.abs(f.contribution)), 1);

  // Sigmoid curve points (SVG viewBox 0 0 200 100)
  const sigmoidPoints = useMemo(() => {
    return Array.from({ length: 80 }, (_, i) => {
      const x = (i / 79) * 200;
      const logOdds = (i / 79) * 12 - 6; // range -6 to +6
      const y = 100 - sigmoid(logOdds) * 80 - 10;
      return `${x},${y}`;
    }).join(' ');
  }, []);

  // Dot position on the curve
  const dotX = Math.min(Math.max(((lrScore + 6) / 12) * 200, 4), 196);
  const dotY = 100 - lrProb * 80 - 10;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="grid grid-cols-2 gap-10 h-full"
    >
      {/* LEFT — Feature Contribution Chart */}
      <div className="flex flex-col gap-5">
        <div>
          <h3 className="text-lg font-semibold text-slate-800 mb-1">
            Step 1 — Weighted Feature Contributions
          </h3>
          <p className="text-sm text-slate-500">
            Each feature multiplied by its learned weight. Red bars push toward LANDMINE.
          </p>
        </div>

        <div className="space-y-4 flex-1">
          {contributions.map((f, i) => {
            const barWidth = (Math.abs(f.contribution) / maxContrib) * 45; // max 45% of half-width
            const isPositive = f.contribution >= 0;

            return (
              <div key={f.key}>
                {/* Feature label + formula */}
                <div className="flex justify-between text-xs text-slate-500 mb-1">
                  <span className="font-medium text-slate-700">{f.label}</span>
                  <span>
                    {f.val.toFixed(2)} × {f.weight.toFixed(1)} =
                    <strong className={isPositive ? ' text-red-600' : ' text-green-600'}>
                      {' '}{f.contribution.toFixed(2)}
                    </strong>
                  </span>
                </div>

                {/* Bar — grows from center */}
                <div className="relative h-6 bg-slate-100 rounded-md overflow-hidden border border-slate-200">
                  {/* Center line */}
                  <div className="absolute left-1/2 top-0 bottom-0 w-px bg-slate-300 z-10" />

                  {/* Bar itself */}
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${barWidth}%` }}
                    transition={{ delay: i * 0.1 + 0.2, duration: 0.5, ease: 'easeOut' }}
                    className="absolute top-1 bottom-1 rounded-sm"
                    style={{
                      left: isPositive ? '50%' : undefined,
                      right: isPositive ? undefined : '50%',
                      backgroundColor: isPositive ? '#ef4444' : '#22c55e',
                    }}
                  />
                </div>
              </div>
            );
          })}
        </div>

        {/* Total score */}
        <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 text-sm">
          <div className="flex justify-between">
            <span className="text-slate-500">Bias (baseline)</span>
            <span className="font-mono">{bias.toFixed(1)}</span>
          </div>
          <div className="flex justify-between font-bold text-slate-800 border-t border-slate-200 mt-2 pt-2">
            <span>Total log-odds score</span>
            <span className="font-mono">{lrScore.toFixed(2)}</span>
          </div>
        </div>
      </div>

      {/* RIGHT — Sigmoid Curve */}
      <div className="flex flex-col gap-5">
        <div>
          <h3 className="text-lg font-semibold text-slate-800 mb-1">
            Step 2 — Sigmoid Transform
          </h3>
          <p className="text-sm text-slate-500">
            The score is mapped to a probability between 0% and 100%.
          </p>
        </div>

        <div className="flex-1 flex flex-col justify-between">
          {/* SVG Sigmoid */}
          <div className="bg-slate-50 rounded-2xl border border-slate-200 p-4">
            <svg viewBox="0 0 200 100" className="w-full h-40">
              {/* Grid lines */}
              <line x1="0" y1="50" x2="200" y2="50" stroke="#e2e8f0" strokeWidth="1" />
              <line x1="100" y1="0" x2="100" y2="100" stroke="#e2e8f0" strokeWidth="1" />

              {/* Decision threshold line at y=0.5 → svg y = 100 - 0.5*80 - 10 = 50 */}
              <line x1="0" y1="50" x2="200" y2="50" stroke="#94a3b8" strokeWidth="0.8" strokeDasharray="3 2" />
              <text x="3" y="47" fontSize="5" fill="#94a3b8">50% threshold</text>

              {/* Y axis labels */}
              <text x="2" y="14" fontSize="5" fill="#94a3b8">100%</text>
              <text x="2" y="94" fontSize="5" fill="#94a3b8">0%</text>

              {/* X axis labels */}
              <text x="4" y="99" fontSize="5" fill="#94a3b8">-6</text>
              <text x="92" y="99" fontSize="5" fill="#94a3b8">0</text>
              <text x="187" y="99" fontSize="5" fill="#94a3b8">+6</text>

              {/* Sigmoid curve */}
              <polyline
                points={sigmoidPoints}
                fill="none"
                stroke="#3b82f6"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />

              {/* Current point dot */}
              <motion.circle
                cx={dotX}
                cy={dotY}
                r="4"
                fill={isLandmine ? '#ef4444' : '#22c55e'}
                stroke="white"
                strokeWidth="1.5"
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: 0.8, type: 'spring' }}
              />

              {/* Dashed drop lines from dot */}
              <line x1={dotX} y1={dotY} x2={dotX} y2="95" stroke="#94a3b8" strokeWidth="0.5" strokeDasharray="2 2" />
              <line x1={dotX} y1={dotY} x2="5" y2={dotY} stroke="#94a3b8" strokeWidth="0.5" strokeDasharray="2 2" />
            </svg>
          </div>

          {/* Result */}
          <div className="text-center space-y-3 mt-4">
            <div className="text-slate-500 text-sm">
              sigmoid({lrScore.toFixed(2)}) =
            </div>
            <motion.div
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ delay: 1, type: 'spring' }}
              className="text-5xl font-black text-slate-900"
            >
              {(lrProb * 100).toFixed(1)}%
            </motion.div>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 1.2 }}
              className={`inline-block px-6 py-2 rounded-full font-bold text-white text-sm ${
                isLandmine ? 'bg-red-600' : 'bg-green-600'
              }`}
            >
              {isLandmine ? '⚠ LANDMINE' : '✓ SAFE'}
            </motion.div>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
