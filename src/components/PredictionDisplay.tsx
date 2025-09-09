import React, { useState, useEffect } from 'react';
import { Activity, AlertTriangle, CheckCircle, Clock } from 'lucide-react';
import type { Prediction } from '../App';
import { databaseService } from '../services/database';

interface PredictionDisplayProps {
  prediction: Prediction | null;
  isLoading: boolean;
}

const PredictionDisplay: React.FC<PredictionDisplayProps> = ({ prediction: initialPrediction, isLoading: initialIsLoading }) => {
  const [prediction, setPrediction] = useState<Prediction | null>(initialPrediction);
  const [isLoading, setIsLoading] = useState(initialIsLoading);

  useEffect(() => {
    if (!initialPrediction) {
      const fetchLatestPrediction = async () => {
        setIsLoading(true);
        try {
          const latestPrediction = await databaseService.getLatestPrediction();
          if (latestPrediction) {
            setPrediction(latestPrediction);
          }
        } catch (error) {
          console.error('Error fetching latest prediction:', error);
        } finally {
          setIsLoading(false);
        }
      };
      fetchLatestPrediction();
    } else {
      setPrediction(initialPrediction);
    }
  }, [initialPrediction]);

  useEffect(() => {
    setIsLoading(initialIsLoading);
  }, [initialIsLoading]);

  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel) {
      case 'low': return 'text-green-600 bg-green-100';
      case 'medium': return 'text-yellow-600 bg-yellow-100';
      case 'high': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getRiskIcon = (riskLevel: string) => {
    switch (riskLevel) {
      case 'low': return <CheckCircle className="w-5 h-5" />;
      case 'medium': return <Clock className="w-5 h-5" />;
      case 'high': return <AlertTriangle className="w-5 h-5" />;
      default: return <Activity className="w-5 h-5" />;
    }
  };

  if (isLoading) {
    return (
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h2 className="text-2xl font-semibold text-gray-800 mb-6 flex items-center">
          <Activity className="w-6 h-6 text-blue-600 mr-3" />
          Current Prediction
        </h2>
        
        <div className="flex items-center justify-center py-12">
          <div className="text-center">
            <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-600 mx-auto mb-4"></div>
            <p className="text-gray-600">Analyzing sensor data...</p>
            <p className="text-sm text-gray-500 mt-2">AI models are processing your input</p>
          </div>
        </div>
      </div>
    );
  }

  if (!prediction) {
    return (
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h2 className="text-2xl font-semibold text-gray-800 mb-6 flex items-center">
          <Activity className="w-6 h-6 text-blue-600 mr-3" />
          Current Prediction
        </h2>
        
        <div className="text-center py-12">
          <div className="bg-gray-100 rounded-full p-6 w-24 h-24 mx-auto mb-4 flex items-center justify-center">
            <Activity className="w-12 h-12 text-gray-400" />
          </div>
          <p className="text-gray-600 text-lg">No prediction available</p>
          <p className="text-sm text-gray-500 mt-2">Submit sensor data to generate a prediction</p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-semibold text-gray-800 flex items-center">
          <Activity className="w-6 h-6 text-blue-600 mr-3" />
          Current Prediction
        </h2>
        <span className="text-sm text-gray-500">{prediction.timestamp}</span>
      </div>

      {/* Main Prediction Display */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        <div className="text-center">
          <div className="relative">
            <div className="w-32 h-32 mx-auto mb-4">
              <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
                <circle
                  cx="50"
                  cy="50"
                  r="40"
                  stroke="currentColor"
                  strokeWidth="8"
                  fill="transparent"
                  className="text-gray-200"
                />
                <circle
                  cx="50"
                  cy="50"
                  r="40"
                  stroke="currentColor"
                  strokeWidth="8"
                  fill="transparent"
                  strokeDasharray={`${2 * Math.PI * 40}`}
                  strokeDashoffset={`${2 * Math.PI * 40 * (1 - prediction.premiseIndex / 100)}`}
                  className={
                    prediction.riskLevel === 'high' ? 'text-red-600' :
                    prediction.riskLevel === 'medium' ? 'text-yellow-600' : 'text-green-600'
                  }
                  strokeLinecap="round"
                />
              </svg>
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-center">
                  <div className="text-3xl font-bold text-gray-800">
                    {prediction.premiseIndex}%
                  </div>
                  <div className="text-sm text-gray-600">Premise Index</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="space-y-4">
          <div className={`inline-flex items-center px-4 py-2 rounded-full ${getRiskColor(prediction.riskLevel)}`}>
            {getRiskIcon(prediction.riskLevel)}
            <span className="ml-2 font-semibold capitalize">{prediction.riskLevel} Risk</span>
          </div>

          <div className="space-y-3">
            <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
              <span className="text-gray-600">Temperature</span>
              <span className="font-semibold text-gray-800">{prediction.temperature}°C</span>
            </div>
            <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
              <span className="text-gray-600">Rainfall</span>
              <span className="font-semibold text-gray-800">{prediction.rainfall}mm</span>
            </div>
            <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
              <span className="text-gray-600">Water Content</span>
              <span className="font-semibold text-gray-800">{prediction.waterContent}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Risk Assessment */}
      <div className="border-t pt-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-3">Risk Assessment</h3>
        <div className="space-y-2">
          {prediction.riskLevel === 'high' && (
            <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-red-800 font-medium">High Breeding Risk Detected!</p>
              <p className="text-red-700 text-sm mt-1">
                Environmental conditions are highly favorable for dengue breeding. Immediate action recommended.
              </p>
            </div>
          )}
          {prediction.riskLevel === 'medium' && (
            <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
              <p className="text-yellow-800 font-medium">Moderate Breeding Risk</p>
              <p className="text-yellow-700 text-sm mt-1">
                Conditions are moderately favorable. Monitor closely and consider preventive measures.
              </p>
            </div>
          )}
          {prediction.riskLevel === 'low' && (
            <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
              <p className="text-green-800 font-medium">Low Breeding Risk</p>
              <p className="text-green-700 text-sm mt-1">
                Current conditions are not favorable for dengue breeding. Continue regular monitoring.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default PredictionDisplay;