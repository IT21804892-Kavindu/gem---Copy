// import React from 'react';
// import type { Prediction } from '../App';

// interface HistoryChartProps {
//   predictions: Prediction[];
// }

// const HistoryChart: React.FC<HistoryChartProps> = ({ predictions }) => {
//   if (predictions.length === 0) {
//     return (
//       <div className="flex items-center justify-center h-64 text-gray-500">
//         <div className="text-center">
//           <div className="w-16 h-16 mx-auto mb-4 bg-gray-100 rounded-full flex items-center justify-center">
//             <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
//               <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
//             </svg>
//           </div>
//           <p>No prediction history available</p>
//           <p className="text-sm mt-1">Generate predictions to see trends</p>
//         </div>
//       </div>
//     );
//   }

//   const maxIndex = Math.max(...predictions.map(p => p.premiseIndex));
//   const chartHeight = 200;
//   const chartWidth = 100;

//   return (
//     <div className="space-y-4">
//       {/* Chart */}
//       <div className="relative h-64 bg-gray-50 rounded-lg p-4">
//         <svg width="100%" height={chartHeight} className="overflow-visible">
//           {/* Grid lines */}
//           {[0, 25, 50, 75, 100].map(value => (
//             <g key={value}>
//               <line
//                 x1="0"
//                 y1={chartHeight - (value / 100) * chartHeight}
//                 x2="100%"
//                 y2={chartHeight - (value / 100) * chartHeight}
//                 stroke="#e5e7eb"
//                 strokeWidth="1"
//               />
//               <text
//                 x="-10"
//                 y={chartHeight - (value / 100) * chartHeight + 4}
//                 fontSize="12"
//                 fill="#6b7280"
//                 textAnchor="end"
//               >
//                 {value}%
//               </text>
//             </g>
//           ))}
          
//           {/* Data line */}
//           {predictions.length > 1 && (
//             <polyline
//               fill="none"
//               stroke="#2563eb"
//               strokeWidth="3"
//               strokeLinecap="round"
//               strokeLinejoin="round"
//               points={predictions
//                 .slice()
//                 .reverse()
//                 .map((prediction, index) => {
//                   const x = (index / (predictions.length - 1)) * chartWidth;
//                   const y = chartHeight - (prediction.premiseIndex / 100) * chartHeight;
//                   return `${x}%,${y}`;
//                 })
//                 .join(' ')}
//             />
//           )}
          
//           {/* Data points */}
//           {predictions.slice().reverse().map((prediction, index) => {
//             const x = (index / Math.max(predictions.length - 1, 1)) * chartWidth;
//             const y = chartHeight - (prediction.premiseIndex / 100) * chartHeight;
//             const color = prediction.riskLevel === 'high' ? '#dc2626' : 
//                          prediction.riskLevel === 'medium' ? '#d97706' : '#059669';
            
//             return (
//               <g key={prediction.id}>
//                 <circle
//                   cx={`${x}%`}
//                   cy={y}
//                   r="6"
//                   fill={color}
//                   stroke="white"
//                   strokeWidth="2"
//                 />
//                 <title>{`${prediction.premiseIndex}% - ${prediction.timestamp}`}</title>
//               </g>
//             );
//           })}
//         </svg>
//       </div>

//       {/* Recent Predictions List */}
//       <div className="space-y-2 max-h-48 overflow-y-auto">
//         {predictions.slice(0, 5).map((prediction, index) => (
//           <div
//             key={prediction.id}
//             className="flex items-center justify-between p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
//           >
//             <div className="flex items-center space-x-3">
//               <div
//                 className={`w-3 h-3 rounded-full ${
//                   prediction.riskLevel === 'high' ? 'bg-red-600' :
//                   prediction.riskLevel === 'medium' ? 'bg-yellow-600' : 'bg-green-600'
//                 }`}
//               />
//               <div>
//                 <p className="font-medium text-gray-800">{prediction.premiseIndex}%</p>
//                 <p className="text-sm text-gray-500">{prediction.timestamp}</p>
//               </div>
//             </div>
//             <div className="text-right text-sm text-gray-600">
//               <p>{prediction.temperature}°C</p>
//               <p>{prediction.rainfall}mm</p>
//             </div>
//           </div>
//         ))}
//       </div>
//     </div>
//   );
// };

// export default HistoryChart;



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

  // Define the internal coordinate system dimensions for the SVG viewBox.
  const chartHeight = 200;
  const chartWidth = 100; // Using 100 makes percentage-like calculations easy.

  return (
    <div className="space-y-4">
      {/* Chart */}
      <div className="relative h-64 bg-gray-50 rounded-lg p-4">
        {/* FIX: Added viewBox and preserveAspectRatio to create a scalable coordinate system */}
        <svg width="100%" height="100%" viewBox={`0 0 ${chartWidth} ${chartHeight}`} preserveAspectRatio="none" className="overflow-visible">
          {/* Grid lines */}
          {[0, 25, 50, 75, 100].map(value => (
            <g key={value}>
              <line
                x1="0"
                y1={chartHeight - (value / 100) * chartHeight}
                /* FIX: Changed x2 from "100%" to the chartWidth variable */
                x2={chartWidth}
                y2={chartHeight - (value / 100) * chartHeight}
                stroke="#e5e7eb"
                strokeWidth="1"
              />
              <text
                x="-4" // Adjusted position for viewBox
                y={chartHeight - (value / 100) * chartHeight + 4}
                fontSize="12"
                fill="#6b7280"
                textAnchor="end"
              >
                {/* Keep the % here for display purposes only */}
                {value}%
              </text>
            </g>
          ))}
          
          {/* Data line */}
          {predictions.length > 1 && (
            <polyline
              fill="none"
              stroke="#2563eb"
              strokeWidth="3"
              strokeLinecap="round"
              strokeLinejoin="round"
              points={predictions
                .slice()
                .reverse()
                .map((prediction, index) => {
                  const x = (index / (predictions.length - 1)) * chartWidth;
                  const y = chartHeight - (prediction.premiseIndex / 100) * chartHeight;
                  /* FIX: Removed the "%" from the x-coordinate */
                  return `${x},${y}`;
                })
                .join(' ')}
            />
          )}
          
          {/* Data points */}
          {predictions.slice().reverse().map((prediction, index) => {
            const x = (index / Math.max(predictions.length - 1, 1)) * chartWidth;
            const y = chartHeight - (prediction.premiseIndex / 100) * chartHeight;
            const color = prediction.riskLevel === 'high' ? '#dc2626' : 
                          prediction.riskLevel === 'medium' ? '#d97706' : '#059669';
            
            return (
              <g key={prediction.id}>
                <circle
                  /* FIX: Removed the "%" from cx */
                  cx={x}
                  cy={y}
                  r="6"
                  fill={color}
                  stroke="white"
                  strokeWidth="2"
                />
                <title>{`${new Date(prediction.timestamp).toLocaleString()}: ${prediction.premiseIndex}%`}</title>
              </g>
            );
          })}
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
