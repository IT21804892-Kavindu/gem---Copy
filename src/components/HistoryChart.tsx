import React from 'react';
import type { Prediction } from '../App';

interface HistoryChartProps {
  predictions: Prediction[];
}

const HistoryChart: React.FC<HistoryChartProps> = ({ predictions }) => {
  if (predictions.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500">
        <div className="text-center">
          <div className="w-16 h-16 mx-auto mb-4 bg-gray-100 rounded-full flex items-center justify-center">
            <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </div>
          <p>No prediction history available</p>
          <p className="text-sm mt-1">Generate predictions to see trends</p>
        </div>
      </div>
    );
  }

  // SVG dimensions and margin convention
  const svgWidth = 500; // An arbitrary width for the coordinate system
  const svgHeight = 250; // An arbitrary height
  const margin = { top: 20, right: 20, bottom: 30, left: 40 };
  const innerWidth = svgWidth - margin.left - margin.right;
  const innerHeight = svgHeight - margin.top - margin.bottom;

  return (
    <div className="space-y-4">
      {/* Chart */}
      <div className="relative h-64 bg-gray-50 rounded-lg p-4">
        <svg width="100%" height="100%" viewBox={`0 0 ${svgWidth} ${svgHeight}`} preserveAspectRatio="xMidYMid meet">
          <g transform={`translate(${margin.left}, ${margin.top})`}>
            {/* Y-axis Grid lines and Labels */}
            {[0, 25, 50, 75, 100].map(value => {
              const y = innerHeight - (value / 100) * innerHeight;
              return (
                <g key={value} className="text-gray-400">
                  <line x1={0} y1={y} x2={innerWidth} y2={y} stroke="currentColor" strokeWidth="0.5" />
                  <text
                    x={-margin.left + 10} // Position in the left margin
                    y={y + 4}
                    fontSize="12"
                    fill="#6b7280"
                    textAnchor="start"
                  >
                    {value}%
                  </text>
                </g>
              );
            })}

            {/* Data line */}
            {predictions.length > 1 && (
              <polyline
                fill="none"
                stroke="#2563eb"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                points={predictions
                  .slice()
                  .reverse()
                  .map((prediction, index) => {
                    const x = (index / (predictions.length - 1)) * innerWidth;
                    const y = innerHeight - (prediction.premiseIndex / 100) * innerHeight;
                    return `${x},${y}`;
                  })
                  .join(' ')}
              />
            )}

            {/* Data points */}
            {predictions.slice().reverse().map((prediction, index) => {
              const x = (index / Math.max(predictions.length - 1, 1)) * innerWidth;
              const y = innerHeight - (prediction.premiseIndex / 100) * innerHeight;
              const color = prediction.riskLevel === 'high' ? '#dc2626' :
                            prediction.riskLevel === 'medium' ? '#d97706' : '#059669';

              return (
                <g key={prediction.id}>
                  <circle
                    cx={x}
                    cy={y}
                    r="4"
                    fill={color}
                    stroke="white"
                    strokeWidth="1.5"
                  />
                  <title>{`${new Date(prediction.timestamp).toLocaleString()}: ${prediction.premiseIndex}%`}</title>
                </g>
              );
            })}
          </g>
        </svg>
      </div>

      {/* Recent Predictions List */}
      <div className="space-y-2 max-h-48 overflow-y-auto">
        {predictions.slice(0, 5).map((prediction) => (
          <div
            key={prediction.id}
            className="flex items-center justify-between p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
          >
            <div className="flex items-center space-x-3">
              <div
                className={`w-3 h-3 rounded-full ${
                  prediction.riskLevel === 'high' ? 'bg-red-600' :
                  prediction.riskLevel === 'medium' ? 'bg-yellow-600' : 'bg-green-600'
                }`}
              />
              <div>
                <p className="font-medium text-gray-800">{prediction.premiseIndex}%</p>
                <p className="text-sm text-gray-500">{new Date(prediction.timestamp).toLocaleString()}</p>
              </div>
            </div>
            <div className="text-right text-sm text-gray-600">
              <p>{prediction.sensorData.temperature}°C</p>
              <p>{prediction.sensorData.rainfall}mm</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default HistoryChart;
