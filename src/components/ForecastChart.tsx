// import React from 'react';
// import type { ForecastData } from '../App';

// interface ForecastChartProps {
//   forecast: ForecastData[];
// }

// const ForecastChart: React.FC<ForecastChartProps> = ({ forecast }) => {
//   if (forecast.length === 0) {
//     return (
//       <div className="flex items-center justify-center h-64 text-gray-500">
//         <div className="text-center">
//           <div className="w-16 h-16 mx-auto mb-4 bg-gray-100 rounded-full flex items-center justify-center">
//             <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
//               <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
//             </svg>
//           </div>
//           <p>Loading forecast data...</p>
//         </div>
//       </div>
//     );
//   }

//   const chartHeight = 200;
//   const chartWidth = 100;
//   const maxIndex = Math.max(...forecast.map(f => f.premiseIndex));
//   const minIndex = Math.min(...forecast.map(f => f.premiseIndex));

//   // Group forecast by weeks for better visualization
//   const getWeekStartDate = (date: Date) => {
//     const d = new Date(date);
//     const day = d.getDay();
//     const diff = d.getDate() - day + (day === 0 ? -6 : 1); // adjust when day is sunday
//     return new Date(d.setDate(diff));
//   };

//   const today = new Date();
//   const currentWeekStart = getWeekStartDate(today);

//   const futureForecasts = forecast.filter(f => new Date(f.date) >= currentWeekStart);

//   const weeklyForecast = Array.from({ length: 12 }, (_, i) => {
//     const weekStart = new Date(currentWeekStart);
//     weekStart.setDate(weekStart.getDate() + i * 7);
//     const weekEnd = new Date(weekStart);
//     weekEnd.setDate(weekEnd.getDate() + 6);

//     const weekData = futureForecasts.filter(f => {
//       const fDate = new Date(f.date);
//       return fDate >= weekStart && fDate <= weekEnd;
//     });

//     const avgPremiseIndex = weekData.length > 0
//       ? weekData.reduce((acc, curr) => acc + curr.premiseIndex, 0) / weekData.length
//       : 0;

//     return {
//       date: weekStart.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
//       premiseIndex: avgPremiseIndex,
//     };
//   });

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

//           {/* Forecast area */}
//           <defs>
//             <linearGradient id="forecastGradient" x1="0%" y1="0%" x2="0%" y2="100%">
//               <stop offset="0%" stopColor="#6366f1" stopOpacity={0.3} />
//               <stop offset="100%" stopColor="#6366f1" stopOpacity={0.1} />
//             </linearGradient>
//           </defs>

//           <polygon
//             fill="url(#forecastGradient)"
//             points={[
//               `0,${chartHeight}`,
//               ...weeklyForecast.map((data, index) => {
//                 const x = (index / (weeklyForecast.length - 1)) * chartWidth;
//                 const y = chartHeight - (data.premiseIndex / 100) * chartHeight;
//                 return `${x}%,${y}`;
//               }),
//               `${chartWidth}%,${chartHeight}`
//             ].join(' ')}
//           />

//           {/* Forecast line */}
//           <polyline
//             fill="none"
//             stroke="#6366f1"
//             strokeWidth="3"
//             strokeLinecap="round"
//             strokeLinejoin="round"
//             points={weeklyForecast
//               .map((data, index) => {
//                 const x = (index / (weeklyForecast.length - 1)) * chartWidth;
//                 const y = chartHeight - (data.premiseIndex / 100) * chartHeight;
//                 return `${x}%,${y}`;
//               })
//               .join(' ')}
//           />

//           {/* Data points and labels */}
//           {weeklyForecast.map((data, index) => {
//             const x = (index / (weeklyForecast.length - 1)) * chartWidth;
//             const y = chartHeight - (data.premiseIndex / 100) * chartHeight;

//             return (
//               <g key={index}>
//                 <circle
//                   cx={`${x}%`}
//                   cy={y}
//                   r="4"
//                   fill="#6366f1"
//                   stroke="white"
//                   strokeWidth="2"
//                 />
//                 <text
//                   x={`${x}%`}
//                   y={chartHeight + 20}
//                   fontSize="12"
//                   fill="#6b7280"
//                   textAnchor="middle"
//                 >
//                   {data.date}
//                 </text>
//               </g>
//             );
//           })}
//         </svg>
//       </div>

//       {/* Forecast Summary */}
//       <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
//         <div className="text-center p-3 bg-blue-50 rounded-lg">
//           <p className="text-2xl font-bold text-blue-600">
//             {Math.round(forecast.reduce((acc, f) => acc + f.premiseIndex, 0) / forecast.length)}%
//           </p>
//           <p className="text-sm text-blue-800">Average Risk</p>
//         </div>
//         <div className="text-center p-3 bg-red-50 rounded-lg">
//           <p className="text-2xl font-bold text-red-600">
//             {Math.round(maxIndex)}%
//           </p>
//           <p className="text-sm text-red-800">Peak Risk</p>
//         </div>
//         <div className="text-center p-3 bg-green-50 rounded-lg">
//           <p className="text-2xl font-bold text-green-600">
//             {Math.round(minIndex)}%
//           </p>
//           <p className="text-sm text-green-800">Lowest Risk</p>
//         </div>
//         <div className="text-center p-3 bg-yellow-50 rounded-lg">
//           <p className="text-2xl font-bold text-yellow-600">
//             {weeklyForecast.filter(f => f.premiseIndex > 60).length}
//           </p>
//           <p className="text-sm text-yellow-800">High Risk Weeks</p>
//         </div>
//       </div>

//       {/* Forecast Legend */}
//       <div className="text-center text-sm text-gray-600">
//         <p>Predicted forecast showing weekly predictions • Higher values indicate increased dengue breeding conditions</p>
//       </div>
//     </div>
//   );
// };

// export default ForecastChart;


import React from 'react';
import type { ForecastData } from '../App';

interface ForecastChartProps {
  forecast: ForecastData[];
}

const ForecastChart: React.FC<ForecastChartProps> = ({ forecast = [] }) => {
  if (forecast.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500">
        <div className="text-center">
          <div className="w-16 h-16 mx-auto mb-4 bg-gray-100 rounded-full flex items-center justify-center">
            <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
          </div>
          <p>Loading forecast data...</p>
        </div>
      </div>
    );
  }

  // Define the internal coordinate system dimensions for the SVG viewBox.
  const chartHeight = 200;
  const chartWidth = 100; // Using 100 makes percentage-like calculations easy.
  const maxIndex = Math.max(...forecast.map(f => f.premiseIndex), 0);
  const minIndex = Math.min(...forecast.map(f => f.premiseIndex), 100);

  // Group forecast by weeks for better visualization
  const getWeekStartDate = (date: Date) => {
    const d = new Date(date);
    const day = d.getDay();
    const diff = d.getDate() - day + (day === 0 ? -6 : 1); // adjust when day is sunday
    return new Date(d.setDate(diff));
  };

  const today = new Date();
  const currentWeekStart = getWeekStartDate(today);

  const futureForecasts = forecast.filter(f => new Date(f.date) >= currentWeekStart);

  const weeklyForecast = Array.from({ length: 12 }, (_, i) => {
    const weekStart = new Date(currentWeekStart);
    weekStart.setDate(weekStart.getDate() + i * 7);
    const weekEnd = new Date(weekStart);
    weekEnd.setDate(weekEnd.getDate() + 6);

    const weekData = futureForecasts.filter(f => {
      const fDate = new Date(f.date);
      return fDate >= weekStart && fDate <= weekEnd;
    });

    const avgPremiseIndex = weekData.length > 0
      ? weekData.reduce((acc, curr) => acc + curr.premiseIndex, 0) / weekData.length
      : 0;

    return {
      date: weekStart.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
      premiseIndex: avgPremiseIndex,
    };
  });

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
                {value}
              </text>
            </g>
          ))}
          
          {/* Forecast area */}
          <defs>
            <linearGradient id="forecastGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#6366f1" stopOpacity={0.3} />
              <stop offset="100%" stopColor="#6366f1" stopOpacity={0.1} />
            </linearGradient>
          </defs>
          
          {weeklyForecast.length > 1 && (
            <polygon
              fill="url(#forecastGradient)"
              points={[
                `0,${chartHeight}`,
                ...weeklyForecast.map((data, index) => {
                  const x = (index / (weeklyForecast.length - 1)) * chartWidth;
                  const y = chartHeight - (data.premiseIndex / 100) * chartHeight;
                  /* FIX: Removed the "%" from the x-coordinate */
                  return `${x},${y}`;
                }),
                `${chartWidth},${chartHeight}`
              ].join(' ')}
            />
          )}
          
          {/* Forecast line */}
          {weeklyForecast.length > 1 && (
            <polyline
              fill="none"
              stroke="#6366f1"
              strokeWidth="3"
              strokeLinecap="round"
              strokeLinejoin="round"
              points={weeklyForecast
                .map((data, index) => {
                  const x = (index / (weeklyForecast.length - 1)) * chartWidth;
                  const y = chartHeight - (data.premiseIndex / 100) * chartHeight;
                  /* FIX: Removed the "%" from the x-coordinate */
                  return `${x},${y}`;
                })
                .join(' ')}
            />
          )}
          
          {/* Data points and labels */}
          {weeklyForecast.map((data, index) => {
            if (weeklyForecast.length <= 1) return null;
            const x = (index / (weeklyForecast.length - 1)) * chartWidth;
            const y = chartHeight - (data.premiseIndex / 100) * chartHeight;
            
            return (
              <g key={index}>
                <circle
                  /* FIX: Removed the "%" from cx */
                  cx={x}
                  cy={y}
                  r="4"
                  fill="#6366f1"
                  stroke="white"
                  strokeWidth="2"
                />
                <text
                  /* FIX: Removed the "%" from x */
                  x={x}
                  y={chartHeight + 20} // Position below the chart line
                  fontSize="12"
                  fill="#6b7280"
                  textAnchor="middle"
                >
                  {data.date}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      {/* Forecast Summary */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="text-center p-3 bg-blue-50 rounded-lg">
          <p className="text-2xl font-bold text-blue-600">
            {forecast.length > 0 ? Math.round(forecast.reduce((acc, f) => acc + f.premiseIndex, 0) / forecast.length) : 0}%
          </p>
          <p className="text-sm text-blue-800">Average Risk</p>
        </div>
        <div className="text-center p-3 bg-red-50 rounded-lg">
          <p className="text-2xl font-bold text-red-600">
            {Math.round(maxIndex)}%
          </p>
          <p className="text-sm text-red-800">Peak Risk</p>
        </div>
        <div className="text-center p-3 bg-green-50 rounded-lg">
          <p className="text-2xl font-bold text-green-600">
            {Math.round(minIndex)}%
          </p>
          <p className="text-sm text-green-800">Lowest Risk</p>
        </div>
        <div className="text-center p-3 bg-yellow-50 rounded-lg">
          <p className="text-2xl font-bold text-yellow-600">
            {weeklyForecast.filter(f => f.premiseIndex > 60).length}
          </p>
          <p className="text-sm text-yellow-800">High Risk Weeks</p>
        </div>
      </div>

      {/* Forecast Legend */}
      <div className="text-center text-sm text-gray-600">
        <p>Predicted forecast showing weekly predictions • Higher values indicate increased dengue breeding conditions</p>
      </div>
    </div>
  );
};

export default ForecastChart;
