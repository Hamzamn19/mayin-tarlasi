'use client';

import React, { useMemo } from 'react';
import { motion } from 'framer-motion';

export default function ForestStage({ activeBox }) {
  const rfProb = activeBox?.rf_prob ?? 0.62;
  const voteCount = Math.round(rfProb * 100);
  const isLandmine = rfProb > 0.5;

  // Generate votes ONCE using useMemo — not Math.random() on every render
  const votes = useMemo(() => {
    const arr = Array.from({ length: 20 }, (_, i) => i < Math.round(rfProb * 20));
    // Shuffle deterministically based on rfProb
    return arr.sort(() => Math.sin(rfProb * 999) - 0.5);
  }, [rfProb]);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="grid grid-cols-2 gap-10 h-full"
    >
      {/* LEFT — Vote Grid */}
      <div className="flex flex-col gap-6">
        <div>
          <h3 className="text-lg font-semibold text-slate-800 mb-1">Collective Vote</h3>
          <p className="text-sm text-slate-500">
            Each cell = 1 tree out of 100 casting its vote independently
          </p>
        </div>

        {/* 4×5 grid of vote cards */}
        <div className="grid grid-cols-5 gap-2">
          {votes.map((voted, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, scale: 0.6 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: i * 0.04, duration: 0.2 }}
              className={`h-12 rounded-lg flex items-center justify-center text-xs font-bold border ${
                voted
                  ? 'bg-red-100 border-red-300 text-red-700'
                  : 'bg-green-100 border-green-300 text-green-700'
              }`}
            >
              {voted ? 'MINE' : 'SAFE'}
            </motion.div>
          ))}
        </div>

        {/* Progress bar */}
        <div className="space-y-2">
          <div className="flex justify-between text-sm font-medium">
            <span className="text-slate-600">{voteCount} / 100 voted</span>
            <span className={isLandmine ? 'text-red-600 font-bold' : 'text-green-600 font-bold'}>
              {isLandmine ? 'LANDMINE' : 'SAFE'}
            </span>
          </div>
          <div className="h-3 bg-slate-100 rounded-full overflow-hidden border border-slate-200">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${voteCount}%` }}
              transition={{ duration: 1, delay: 0.8, ease: 'easeOut' }}
              className={`h-full rounded-full ${isLandmine ? 'bg-red-500' : 'bg-green-500'}`}
            />
          </div>
          <div className="flex justify-between text-xs text-slate-400">
            <span>0%</span>
            <span className="text-slate-500">Threshold: 50%</span>
            <span>100%</span>
          </div>
        </div>
      </div>

      {/* RIGHT — One Tree Expanded */}
      <div className="flex flex-col gap-6">
        <div>
          <h3 className="text-lg font-semibold text-slate-800 mb-1">Example: Tree #73</h3>
          <p className="text-sm text-slate-500">
            Decision path taken for this detection
          </p>
        </div>

        <div className="flex-1 flex flex-col items-center gap-0 pt-2">
          {/* Node 1 — Root */}
          <TreeNode
            question="Area > 800px²?"
            answer="YES"
            delay={0.2}
            isActive
          />

          <Connector delay={0.5} />

          {/* Node 2 */}
          <TreeNode
            question="Intensity > 150?"
            answer="YES"
            delay={0.6}
            isActive
          />

          <Connector delay={0.9} />

          {/* Node 3 */}
          <TreeNode
            question="Circularity > 0.5?"
            answer="YES"
            delay={1.0}
            isActive
          />

          <Connector delay={1.3} />

          {/* Leaf */}
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.4 }}
            className="px-8 py-3 bg-red-600 text-white rounded-xl font-bold text-sm shadow-sm"
          >
            → LANDMINE
          </motion.div>
        </div>

        <p className="text-xs text-slate-400 text-center">
          This single tree predicted: <strong className="text-red-600">LANDMINE</strong>
        </p>
      </div>
    </motion.div>
  );
}

function TreeNode({ question, answer, delay, isActive }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay }}
      className={`w-56 px-4 py-3 rounded-xl border-2 text-center ${
        isActive
          ? 'border-blue-400 bg-blue-50'
          : 'border-slate-200 bg-white'
      }`}
    >
      <p className="text-sm font-medium text-slate-700">{question}</p>
      <p className={`text-xs font-bold mt-1 ${answer === 'YES' ? 'text-green-600' : 'text-red-600'}`}>
        ↓ {answer}
      </p>
    </motion.div>
  );
}

function Connector({ delay }) {
  return (
    <motion.div
      initial={{ scaleY: 0 }}
      animate={{ scaleY: 1 }}
      transition={{ delay, duration: 0.2 }}
      className="w-px h-8 bg-blue-300 origin-top"
    />
  );
}
