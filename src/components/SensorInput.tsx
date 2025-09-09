// import React, { useState } from 'react';
// import { Droplets, Thermometer, Waves, Send, CalendarDays } from 'lucide-react';
// import type { SensorData } from '../App';

// interface SensorInputProps {
//   onSubmit: (data: SensorData) => void;
//   isLoading: boolean;
// }

// const SensorInput: React.FC<SensorInputProps> = ({ onSubmit, isLoading }) => {
//   const [formData, setFormData] = useState<SensorData>({
//     rainfall: 0,
//     temperature: 25,
//     waterContent: 0,
//     Rainfall_7d_avg: 0,
//     WaterContent_7d_avg: 0,
//   });

//   const [errors, setErrors] = useState<Partial<SensorData>>({});

//   const validateForm = (): boolean => {
//     const newErrors: Partial<SensorData> = {};

//     if (formData.rainfall < 0 || formData.rainfall > 500) {
//       newErrors.rainfall = formData.rainfall;
//     }

//     if (formData.temperature < -10 || formData.temperature > 50) {
//       newErrors.temperature = formData.temperature;
//     }

//     if (formData.waterContent < 0 || formData.waterContent > 100) {
//       newErrors.waterContent = formData.waterContent;
//     }

//     if (formData.Rainfall_7d_avg && (formData.Rainfall_7d_avg < 0 || formData.Rainfall_7d_avg > 500)) {
//         newErrors.Rainfall_7d_avg = formData.Rainfall_7d_avg;
//     }

//     if (formData.WaterContent_7d_avg && (formData.WaterContent_7d_avg < 0 || formData.WaterContent_7d_avg > 100)) {
//         newErrors.WaterContent_7d_avg = formData.WaterContent_7d_avg;
//     }

//     setErrors(newErrors);
//     return Object.keys(newErrors).length === 0;
//   };

//   const handleSubmit = (e: React.FormEvent) => {
//     e.preventDefault();
//     if (validateForm()) {
//       onSubmit(formData);
//     }
//   };

//   const handleInputChange = (field: keyof SensorData, value: number) => {
//     setFormData(prev => ({ ...prev, [field]: value }));
//     if (errors[field] !== undefined) {
//       setErrors(prev => ({ ...prev, [field]: undefined }));
//     }
//   };

//   return (
//     <div className="bg-white rounded-xl shadow-lg p-6">
//       <h2 className="text-2xl font-semibold text-gray-800 mb-6 flex items-center">
//         <Send className="w-6 h-6 text-blue-600 mr-3" />
//         Sensor Data Input
//       </h2>

//       <form onSubmit={handleSubmit} className="space-y-6">
//         {/* Rainfall Input */}
//         <div>
//           <label className="flex items-center text-sm font-medium text-gray-700 mb-2">
//             <Droplets className="w-4 h-4 text-blue-500 mr-2" />
//             Average Rainfall (mm)
//           </label>
//           <input
//             type="number"
//             step="0.1"
//             min="0"
//             max="500"
//             value={formData.rainfall}
//             onChange={(e) => handleInputChange('rainfall', parseFloat(e.target.value) || 0)}
//             className={`w-full px-4 py-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all ${
//               errors.rainfall !== undefined ? 'border-red-500 bg-red-50' : 'border-gray-300'
//             }`}
//             placeholder="Enter rainfall amount..."
//           />
//           {errors.rainfall !== undefined && (
//             <p className="text-red-500 text-sm mt-1">Rainfall must be between 0-500mm</p>
//           )}
//         </div>

//         {/* Temperature Input */}
//         <div>
//           <label className="flex items-center text-sm font-medium text-gray-700 mb-2">
//             <Thermometer className="w-4 h-4 text-orange-500 mr-2" />
//             Temperature (°C)
//           </label>
//           <input
//             type="number"
//             step="0.1"
//             min="-10"
//             max="50"
//             value={formData.temperature}
//             onChange={(e) => handleInputChange('temperature', parseFloat(e.target.value) || 0)}
//             className={`w-full px-4 py-3 border rounded-lg focus:ring-2 focus:ring-orange-500 focus:border-transparent transition-all ${
//               errors.temperature !== undefined ? 'border-red-500 bg-red-50' : 'border-gray-300'
//             }`}
//             placeholder="Enter temperature..."
//           />
//           {errors.temperature !== undefined && (
//             <p className="text-red-500 text-sm mt-1">Temperature must be between -10°C to 50°C</p>
//           )}
//         </div>

//         {/* Water Content Input */}
//         <div>
//           <label className="flex items-center text-sm font-medium text-gray-700 mb-2">
//             <Waves className="w-4 h-4 text-cyan-500 mr-2" />
//             Water Content (0-100)
//           </label>
//           <input
//             type="number"
//             step="0.1"
//             min="0"
//             max="100"
//             value={formData.waterContent}
//             onChange={(e) => handleInputChange('waterContent', parseFloat(e.target.value) || 0)}
//             className={`w-full px-4 py-3 border rounded-lg focus:ring-2 focus:ring-cyan-500 focus:border-transparent transition-all ${
//               errors.waterContent !== undefined ? 'border-red-500 bg-red-50' : 'border-gray-300'
//             }`}
//             placeholder="Enter water content..."
//           />
//           {errors.waterContent !== undefined && (
//             <p className="text-red-500 text-sm mt-1">Water content must be between 0-100</p>
//           )}
//         </div>

//         {/* 7-day Average Rainfall Input */}
//         <div>
//           <label className="flex items-center text-sm font-medium text-gray-700 mb-2">
//             <CalendarDays className="w-4 h-4 text-blue-500 mr-2" />
//             7-day Average Rainfall (mm)
//           </label>
//           <input
//             type="number"
//             step="0.1"
//             min="0"
//             max="500"
//             value={formData.Rainfall_7d_avg}
//             onChange={(e) => handleInputChange('Rainfall_7d_avg', parseFloat(e.target.value) || 0)}
//             className={`w-full px-4 py-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all ${
//               errors.Rainfall_7d_avg !== undefined ? 'border-red-500 bg-red-50' : 'border-gray-300'
//             }`}
//             placeholder="Enter 7-day average rainfall..."
//           />
//           {errors.Rainfall_7d_avg !== undefined && (
//             <p className="text-red-500 text-sm mt-1">Rainfall must be between 0-500mm</p>
//           )}
//         </div>

//         {/* 7-day Average Water Content Input */}
//         <div>
//           <label className="flex items-center text-sm font-medium text-gray-700 mb-2">
//             <CalendarDays className="w-4 h-4 text-cyan-500 mr-2" />
//             7-day Average Water Content (0-100)
//           </label>
//           <input
//             type="number"
//             step="0.1"
//             min="0"
//             max="100"
//             value={formData.WaterContent_7d_avg}
//             onChange={(e) => handleInputChange('WaterContent_7d_avg', parseFloat(e.target.value) || 0)}
//             className={`w-full px-4 py-3 border rounded-lg focus:ring-2 focus:ring-cyan-500 focus:border-transparent transition-all ${
//               errors.WaterContent_7d_avg !== undefined ? 'border-red-500 bg-red-50' : 'border-gray-300'
//             }`}
//             placeholder="Enter 7-day average water content..."
//           />
//           {errors.WaterContent_7d_avg !== undefined && (
//             <p className="text-red-500 text-sm mt-1">Water content must be between 0-100</p>
//           )}
//         </div>

//         {/* Submit Button */}
//         <button
//           type="submit"
//           disabled={isLoading}
//           className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white font-semibold py-3 px-6 rounded-lg transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
//         >
//           {isLoading ? (
//             <>
//               <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
//               Processing...
//             </>
//           ) : (
//             <>
//               <Send className="w-5 h-5 mr-2" />
//               Generate Prediction
//             </>
//           )}
//         </button>
//       </form>

//       {/* Input Guidelines */}
//       <div className="mt-6 p-4 bg-gray-50 rounded-lg">
//         <h3 className="text-sm font-semibold text-gray-700 mb-2">Input Guidelines:</h3>
//         <ul className="text-sm text-gray-600 space-y-1">
//           <li>• Rainfall: Typical range 0-200mm for daily readings</li>
//           <li>• Temperature: Normal range 20-35°C for tropical climates</li>
//           <li>• Water Content: Scale of 0-100 representing moisture percentage</li>
//           <li>• 7-day Averages: Provide historical averages for more accurate predictions</li>
//         </ul>
//       </div>
//     </div>
//   );
// };

// export default SensorInput;


import React, { useState } from 'react';
import { Droplets, Thermometer, Waves, Send, CalendarDays } from 'lucide-react';
import type { SensorData } from '../App';

interface SensorInputProps {
  onSubmit: (data: SensorData) => void;
  isLoading: boolean;
}

const SensorInput: React.FC<SensorInputProps> = ({ onSubmit, isLoading }) => {
  const [formData, setFormData] = useState<SensorData>({
    rainfall: 0,
    temperature: 25,
    waterContent: 0,
    Rainfall_7d_avg: 0,
    WaterContent_7d_avg: 0,
  });

  const [errors, setErrors] = useState<Partial<SensorData>>({});

  const validateForm = (): boolean => {
    const newErrors: Partial<SensorData> = {};

    if (formData.rainfall < 0 || formData.rainfall > 500) {
      newErrors.rainfall = formData.rainfall;
    }

    if (formData.temperature < -10 || formData.temperature > 50) {
      newErrors.temperature = formData.temperature;
    }

    if (formData.waterContent < 0 || formData.waterContent > 100) {
      newErrors.waterContent = formData.waterContent;
    }

    // Always validate the 7-day average fields
    if (formData.Rainfall_7d_avg === undefined || formData.Rainfall_7d_avg < 0 || formData.Rainfall_7d_avg > 500) {
        newErrors.Rainfall_7d_avg = formData.Rainfall_7d_avg;
    }

    if (formData.WaterContent_7d_avg === undefined || formData.WaterContent_7d_avg < 0 || formData.WaterContent_7d_avg > 100) {
        newErrors.WaterContent_7d_avg = formData.WaterContent_7d_avg;
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (validateForm()) {
      // Always submit the full formData object
      onSubmit(formData);
    }
  };

  const handleInputChange = (field: keyof SensorData, value: number) => {
    setFormData(prev => ({ ...prev, [field]: value }));
    if (errors[field] !== undefined) {
      setErrors(prev => ({ ...prev, [field]: undefined }));
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <h2 className="text-2xl font-semibold text-gray-800 mb-6 flex items-center">
        <Send className="w-6 h-6 text-blue-600 mr-3" />
        Sensor Data Input
      </h2>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Rainfall Input */}
        <div>
          <label className="flex items-center text-sm font-medium text-gray-700 mb-2">
            <Droplets className="w-4 h-4 text-blue-500 mr-2" />
            Average Rainfall (mm)
          </label>
          <input
            type="number"
            step="0.1"
            min="0"
            max="500"
            value={formData.rainfall}
            onChange={(e) => handleInputChange('rainfall', parseFloat(e.target.value) || 0)}
            className={`w-full px-4 py-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all ${
              errors.rainfall !== undefined ? 'border-red-500 bg-red-50' : 'border-gray-300'
            }`}
            placeholder="Enter rainfall amount..."
          />
          {errors.rainfall !== undefined && (
            <p className="text-red-500 text-sm mt-1">Rainfall must be between 0-500mm</p>
          )}
        </div>

        {/* Temperature Input */}
        <div>
          <label className="flex items-center text-sm font-medium text-gray-700 mb-2">
            <Thermometer className="w-4 h-4 text-orange-500 mr-2" />
            Temperature (°C)
          </label>
          <input
            type="number"
            step="0.1"
            min="-10"
            max="50"
            value={formData.temperature}
            onChange={(e) => handleInputChange('temperature', parseFloat(e.target.value) || 0)}
            className={`w-full px-4 py-3 border rounded-lg focus:ring-2 focus:ring-orange-500 focus:border-transparent transition-all ${
              errors.temperature !== undefined ? 'border-red-500 bg-red-50' : 'border-gray-300'
            }`}
            placeholder="Enter temperature..."
          />
          {errors.temperature !== undefined && (
            <p className="text-red-500 text-sm mt-1">Temperature must be between -10°C to 50°C</p>
          )}
        </div>

        {/* Water Content Input */}
        <div>
          <label className="flex items-center text-sm font-medium text-gray-700 mb-2">
            <Waves className="w-4 h-4 text-cyan-500 mr-2" />
            Water Content (0-100)
          </label>
          <input
            type="number"
            step="0.1"
            min="0"
            max="100"
            value={formData.waterContent}
            onChange={(e) => handleInputChange('waterContent', parseFloat(e.target.value) || 0)}
            className={`w-full px-4 py-3 border rounded-lg focus:ring-2 focus:ring-cyan-500 focus:border-transparent transition-all ${
              errors.waterContent !== undefined ? 'border-red-500 bg-red-50' : 'border-gray-300'
            }`}
            placeholder="Enter water content..."
          />
          {errors.waterContent !== undefined && (
            <p className="text-red-500 text-sm mt-1">Water content must be between 0-100</p>
          )}
        </div>
        
        {/* 7-day Average Rainfall Input */}
        <div>
          <label className="flex items-center text-sm font-medium text-gray-700 mb-2">
            <CalendarDays className="w-4 h-4 text-blue-500 mr-2" />
            7-day Average Rainfall (mm)
          </label>
          <input
            type="number"
            step="0.1"
            min="0"
            max="500"
            value={formData.Rainfall_7d_avg}
            onChange={(e) => handleInputChange('Rainfall_7d_avg', parseFloat(e.target.value) || 0)}
            className={`w-full px-4 py-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all ${
              errors.Rainfall_7d_avg !== undefined ? 'border-red-500 bg-red-50' : 'border-gray-300'
            }`}
            placeholder="Enter 7-day average rainfall..."
          />
          {errors.Rainfall_7d_avg !== undefined && (
            <p className="text-red-500 text-sm mt-1">Rainfall must be between 0-500mm</p>
          )}
        </div>

        {/* 7-day Average Water Content Input */}
        <div>
          <label className="flex items-center text-sm font-medium text-gray-700 mb-2">
            <CalendarDays className="w-4 h-4 text-cyan-500 mr-2" />
            7-day Average Water Content (0-100)
          </label>
          <input
            type="number"
            step="0.1"
            min="0"
            max="100"
            value={formData.WaterContent_7d_avg}
            onChange={(e) => handleInputChange('WaterContent_7d_avg', parseFloat(e.target.value) || 0)}
            className={`w-full px-4 py-3 border rounded-lg focus:ring-2 focus:ring-cyan-500 focus:border-transparent transition-all ${
              errors.WaterContent_7d_avg !== undefined ? 'border-red-500 bg-red-50' : 'border-gray-300'
            }`}
            placeholder="Enter 7-day average water content..."
          />
          {errors.WaterContent_7d_avg !== undefined && (
            <p className="text-red-500 text-sm mt-1">Water content must be between 0-100</p>
          )}
        </div>

        {/* Submit Button */}
        <button
          type="submit"
          disabled={isLoading}
          className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white font-semibold py-3 px-6 rounded-lg transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
        >
          {isLoading ? (
            <>
              <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
              Processing...
            </>
          ) : (
            <>
              <Send className="w-5 h-5 mr-2" />
              Generate Prediction
            </>
          )}
        </button>
      </form>

      {/* Input Guidelines */}
      <div className="mt-6 p-4 bg-gray-50 rounded-lg">
        <h3 className="text-sm font-semibold text-gray-700 mb-2">Input Guidelines:</h3>
        <ul className="text-sm text-gray-600 space-y-1">
          <li>• Rainfall: Typical range 0-200mm for daily readings</li>
          <li>• Temperature: Normal range 20-35°C for tropical climates</li>
          <li>• Water Content: Scale of 0-100 representing moisture percentage</li>
          <li>• 7-day Averages: Provide historical averages for more accurate predictions</li>
        </ul>
      </div>
    </div>
  );
};

export default SensorInput;

