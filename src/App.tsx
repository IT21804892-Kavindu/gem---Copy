import React, { useState, useEffect } from 'react';
import { 
  Activity, 
  TrendingUp, 
  Calendar, 
  Database, 
  AlertTriangle, 
  Thermometer, 
  Droplets 
} from 'lucide-react';
import SensorInput from './components/SensorInput';
import PredictionDisplay from './components/PredictionDisplay';
import HistoryChart from './components/HistoryChart';
import ForecastChart from './components/ForecastChart';
import AlertSystem from './components/AlertSystem';
import ReportButton from './components/ReportButton';
import { databaseService } from './services/database';
import { extendedApiService as apiService } from './services/api';

export interface SensorData {
  rainfall: number;
  temperature: number;
  waterContent: number;
  Rainfall_7d_avg?: number;
  WaterContent_7d_avg?: number;
}

export interface Prediction {
  id: string;
  timestamp: string;
  premiseIndex: number;
  rainfall: number;
  temperature: number;
  waterContent: number;
  Rainfall_7d_avg?: number;
  WaterContent_7d_avg?: number;
  riskLevel: 'low' | 'medium' | 'high';
  confidence?: number;
}

export interface ForecastData {
  date: string;
  premiseIndex: number;
}

const App: React.FC = () => {
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [displayPredictions, setDisplayPredictions] = useState<Prediction[]>([]);
  const [currentPrediction, setCurrentPrediction] = useState<Prediction | null>(null);
  const [forecast, setForecast] = useState<ForecastData[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [alerts, setAlerts] = useState<string[]>([]);
  const [backendStatus, setBackendStatus] = useState<'connected' | 'disconnected' | 'checking'>('checking');
  const [isSavingToDb, setIsSavingToDb] = useState(false);

  useEffect(() => {
    loadInitialData();
    loadForecast();
    checkBackendStatus();
  }, []);

  const loadInitialData = async () => {
    try {
      const allPredictions = await databaseService.getAllPredictions();
      setPredictions(allPredictions);

      const thirtyDaysAgo = new Date();
      thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30);

      const filtered = allPredictions.filter(prediction => {
        const predictionDate = new Date(prediction.timestamp);
        return predictionDate >= thirtyDaysAgo;
      });

      setDisplayPredictions(filtered);
      console.log(`Loaded ${allPredictions.length} predictions from Firebase`);
    } catch (error) {
      console.error('Error loading initial data:', error);
    }
  };


  const checkBackendStatus = async () => {
    try {
      await apiService.checkHealth();
      setBackendStatus('connected');
    } catch (error) {
      setBackendStatus('disconnected');
    }
  };

  const handlePrediction = async (data: SensorData) => {
    setIsLoading(true);
    setIsSavingToDb(true);
    
    try {
      // Try to get prediction from backend
      const apiResult = await apiService.getPrediction(data);
      
      // Convert API response to Prediction format
      const prediction: Prediction = {
        id: Date.now().toString(),
        timestamp: new Date().toLocaleString(),
        premiseIndex: parseFloat(apiResult.premiseIndex.toFixed(2)),
        rainfall: parseFloat(data.rainfall.toFixed(2)),
        temperature: parseFloat(data.temperature.toFixed(2)),
        waterContent: parseFloat(data.waterContent.toFixed(2)),
        Rainfall_7d_avg: data.Rainfall_7d_avg ? parseFloat(data.Rainfall_7d_avg.toFixed(2)) : undefined,
        WaterContent_7d_avg: data.WaterContent_7d_avg ? parseFloat(data.WaterContent_7d_avg.toFixed(2)) : undefined,
        riskLevel: apiResult.riskLevel,
        confidence: parseFloat((apiResult.confidence || 0).toFixed(2))
      };
      
      setCurrentPrediction(prediction);
      const newPredictions = [prediction, ...predictions];
      setPredictions(newPredictions);
      const thirtyDaysAgo = new Date();
      thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30);
      const filtered = newPredictions.filter(p => new Date(p.timestamp) >= thirtyDaysAgo);
      setDisplayPredictions(filtered);
      
      // Save to database
      try {
        await databaseService.savePrediction(prediction);
        setIsSavingToDb(false);
      } catch (dbError) {
        console.error('Database save error:', dbError);
        setAlerts(prev => ['Failed to save to database', ...prev.slice(0, 4)]);
        setIsSavingToDb(false);
      }
      
      // Generate alert if high risk
      if (prediction.riskLevel === 'high') {
        const alertMessage = `High breeding risk detected! Premise Index: ${prediction.premiseIndex}% (Confidence: ${((prediction.confidence || 0) * 100).toFixed(1)}%)`;
        setAlerts(prev => [alertMessage, ...prev.slice(0, 4)]);
      }
      
    } catch (error) {
      console.error('Prediction error:', error);
      setAlerts(prev => [`Prediction failed: ${error instanceof Error ? error.message : 'Unknown error'}`, ...prev.slice(0, 4)]);
      
      // Fallback to local calculation if backend fails
      const fallbackPrediction = calculateFallbackPrediction(data);
      setCurrentPrediction(fallbackPrediction);
      const newPredictions = [fallbackPrediction, ...predictions];
      setPredictions(newPredictions);
      const thirtyDaysAgo = new Date();
      thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30);
      const filtered = newPredictions.filter(p => new Date(p.timestamp) >= thirtyDaysAgo);
      setDisplayPredictions(filtered);
    } finally {
      setIsLoading(false);
    }
  };

  const calculateFallbackPrediction = (data: SensorData): Prediction => {
    // Fallback calculation when backend is unavailable
    const rainfallWeight = 0.4;
    const temperatureWeight = 0.35;
    const waterContentWeight = 0.25;
    
    const normalizedRainfall = Math.min(data.rainfall / 200 * 100, 100);
    const normalizedTemp = Math.min((data.temperature - 20) / 15 * 100, 100);
    const normalizedWater = Math.min(data.waterContent, 100);
    
    const premiseIndex = (
      normalizedRainfall * rainfallWeight +
      normalizedTemp * temperatureWeight +
      normalizedWater * waterContentWeight
    );
    
    const variation = (Math.random() - 0.5) * 20;
    const finalIndex = Math.max(0, Math.min(100, premiseIndex + variation));
    
    const getRiskLevel = (index: number): 'low' | 'medium' | 'high' => {
      if (index < 30) return 'low';
      if (index < 60) return 'medium';
      return 'high';
    };

    return {
      id: Date.now().toString(),
      timestamp: new Date().toLocaleString(),
      premiseIndex: parseFloat(finalIndex.toFixed(2)),
      rainfall: parseFloat(data.rainfall.toFixed(2)),
      temperature: parseFloat(data.temperature.toFixed(2)),
      waterContent: parseFloat(data.waterContent.toFixed(2)),
      Rainfall_7d_avg: data.Rainfall_7d_avg ? parseFloat(data.Rainfall_7d_avg.toFixed(2)) : undefined,
      WaterContent_7d_avg: data.WaterContent_7d_avg ? parseFloat(data.WaterContent_7d_avg.toFixed(2)) : undefined,
      riskLevel: getRiskLevel(finalIndex),
      confidence: 0.60 // Lower confidence for fallback
    };
  };

  const loadForecast = async () => {
    try {
      const result = await apiService.getTimeSeriesForecast(90);
      const forecastData: ForecastData[] = result.dates.map((date, index) => ({
        date,
        premiseIndex: parseFloat(result.predictions[index].toFixed(2))
      }));
      setForecast(forecastData);
    } catch (error) {
      console.warn('Backend forecast unavailable, using fallback data:', error);
      // Generate fallback forecast
      generateFallbackForecast();
    }
  };

  const generateFallbackForecast = () => {
    const forecastData: ForecastData[] = [];
    const today = new Date();
    
    for (let i = 0; i < 90; i++) {
      const date = new Date(today);
      date.setDate(date.getDate() + i);
      
      const seasonalFactor = Math.sin((i / 365) * 2 * Math.PI) * 15 + 45;
      const randomVariation = (Math.random() - 0.5) * 20;
      const premiseIndex = Math.max(10, Math.min(90, seasonalFactor + randomVariation));
      
      forecastData.push({
        date: date.toISOString().split('T')[0],
        premiseIndex: parseFloat(premiseIndex.toFixed(2))
      });
    }
    
    setForecast(forecastData);
  };

  const handleReportGenerated = async () => {
    try {
      await databaseService.clearAllPredictions();
      setPredictions([]);
      setDisplayPredictions([]);
      setCurrentPrediction(null);
      setAlerts(prev => ['Prediction history cleared.', ...prev.slice(0, 4)]);
    } catch (error) {
      console.error('Error clearing data after report generation:', error);
      setAlerts((prev: string[]) => ['Failed to clear prediction history.', ...prev.slice(0, 4)]);
    }
  };

  const dismissAlert = (index: number) => {
    setAlerts((prev: string[]) => prev.filter((_, i) => i !== index));
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center mb-4">
            <div className="bg-blue-600 p-3 rounded-full mr-4">
              <Activity className="w-8 h-8 text-white" />
            </div>
            <h1 className="text-4xl font-bold text-gray-800">
              Dengue Breeding Prediction System
            </h1>
          </div>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Advanced AI-powered system for predicting dengue breeding conditions using environmental sensor data
          </p>
          
          {/* Backend Status Indicator */}
          <div className="mt-4 flex items-center justify-center">
            <div className={`flex items-center px-3 py-1 rounded-full text-sm ${
              backendStatus === 'connected' ? 'bg-green-100 text-green-800' :
              backendStatus === 'disconnected' ? 'bg-red-100 text-red-800' :
              'bg-yellow-100 text-yellow-800'
            }`}>
              <div className={`w-2 h-2 rounded-full mr-2 ${
                backendStatus === 'connected' ? 'bg-green-600' :
                backendStatus === 'disconnected' ? 'bg-red-600' :
                'bg-yellow-600'
              }`}></div>
              {backendStatus === 'connected' ? 'ML Models Connected' :
               backendStatus === 'disconnected' ? 'Backend Disconnected' :
               'Checking Connection...'}
            </div>
            
            {/* Database Status */}
            {isSavingToDb && (
              <div className="ml-3 flex items-center px-3 py-1 rounded-full text-sm bg-blue-100 text-blue-800">
                <div className="w-2 h-2 rounded-full mr-2 bg-blue-600 animate-pulse"></div>
                Saving to Database...
              </div>
            )}
          </div>
          
          {/* Report Generation Button */}
          <div className="mt-6 flex justify-center">
            <ReportButton 
              predictions={predictions} 
              onReportGenerated={handleReportGenerated}
            />
          </div>
        </div>

        {/* Alert System */}
        <AlertSystem alerts={alerts} onDismiss={dismissAlert} />

        {/* Main Dashboard Grid */}
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-8 mb-8">
          {/* Sensor Input */}
          <div className="xl:col-span-1">
            <SensorInput onSubmit={handlePrediction} isLoading={isLoading} />
          </div>

          {/* Current Prediction */}
          <div className="xl:col-span-2">
            <PredictionDisplay 
              prediction={currentPrediction} 
              isLoading={isLoading} 
            />
          </div>
        </div>

        {/* Charts Section */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* History Chart */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <div className="flex items-center mb-6">
              <TrendingUp className="w-6 h-6 text-blue-600 mr-3" />
              <h2 className="text-2xl font-semibold text-gray-800">Prediction History (Last 30 Days)</h2>
            </div>
            <HistoryChart predictions={displayPredictions} />
          </div>

          {/* Forecast Chart */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <div className="flex items-center mb-6">
              <Calendar className="w-6 h-6 text-indigo-600 mr-3" />
              <h2 className="text-2xl font-semibold text-gray-800">Predicted Forecast</h2>
            </div>
            <ForecastChart forecast={forecast} />
          </div>
        </div>

        {/* Statistics Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <div className="bg-white rounded-xl shadow-lg p-6 text-center">
            <Database className="w-8 h-8 text-blue-600 mx-auto mb-3" />
            <h3 className="text-lg font-semibold text-gray-800">Total Predictions</h3>
            <p className="text-3xl font-bold text-blue-600">{displayPredictions.length}</p>
          </div>
          
          <div className="bg-white rounded-xl shadow-lg p-6 text-center">
            <AlertTriangle className="w-8 h-8 text-red-600 mx-auto mb-3" />
            <h3 className="text-lg font-semibold text-gray-800">High Risk Alerts</h3>
            <p className="text-3xl font-bold text-red-600">
              {displayPredictions.filter(p => p.riskLevel === 'high').length}
            </p>
          </div>
          
          <div className="bg-white rounded-xl shadow-lg p-6 text-center">
            <Thermometer className="w-8 h-8 text-orange-600 mx-auto mb-3" />
            <h3 className="text-lg font-semibold text-gray-800">Avg Temperature</h3>
            <p className="text-3xl font-bold text-orange-600">
              {displayPredictions.length > 0 
                ? `${(displayPredictions.reduce((acc, p) => acc + p.temperature, 0) / displayPredictions.length).toFixed(2)}°C`
                : '0.00°C'
              }
            </p>
          </div>
          
          <div className="bg-white rounded-xl shadow-lg p-6 text-center">
            <Droplets className="w-8 h-8 text-cyan-600 mx-auto mb-3" />
            <h3 className="text-lg font-semibold text-gray-800">Avg Rainfall</h3>
            <p className="text-3xl font-bold text-cyan-600">
              {displayPredictions.length > 0 
                ? `${(displayPredictions.reduce((acc, p) => acc + p.rainfall, 0) / displayPredictions.length).toFixed(2)}mm`
                : '0.00mm'
              }
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;