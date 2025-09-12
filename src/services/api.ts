// API service for communicating with the backend ML models
const API_BASE_URL = import.meta.env.VITE_REACT_APP_API_URL || 'http://localhost:5000';

export interface ModelPredictionRequest {
  rainfall: number;
  temperature: number;
  waterContent: number;
  Rainfall_7d_avg?: number;
  WaterContent_7d_avg?: number;
}

export interface ModelPredictionResponse {
  premiseIndex: number;
  riskLevel: 'low' | 'medium' | 'high';
  confidence: number;
  timestamp: string;
}

export interface TimeSeriesForecastResponse {
  dates: string[];
  predictions: number[];
  confidence_intervals?: {
    lower: number[];
    upper: number[];
  };
}

class APIService {
  protected async makeRequest<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    });

    if (!response.ok) {
      throw new Error(`API Error: ${response.status} - ${response.statusText}`);
    }

    return response.json();
  }

  async predictPremiseIndex(data: ModelPredictionRequest): Promise<ModelPredictionResponse> {
    return this.makeRequest<ModelPredictionResponse>('/api/predict', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async getTimeSeriesForecast(days: number = 90): Promise<TimeSeriesForecastResponse> {
    return this.makeRequest<TimeSeriesForecastResponse>(`/api/forecast?days=${days}`);
  }

  async getModelHealth(): Promise<{ status: string; models_loaded: boolean }> {
    return this.makeRequest<{ status: string; models_loaded: boolean }>('/api/health');
  }
}

export const apiService = new APIService();

// Additional methods for the APIService class
export class APIServiceExtended extends APIService {
  async getPrediction(data: ModelPredictionRequest): Promise<ModelPredictionResponse> {
    return this.predictPremiseIndex(data);
  }

  async checkHealth(): Promise<{ status: string; models_loaded: boolean }> {
    return this.getModelHealth();
  }

  async getTimeSeriesForecast(days: number = 90): Promise<TimeSeriesForecastResponse> {
    return this.makeRequest<TimeSeriesForecastResponse>(`/api/forecast?days=${days}`);
  }
}

// Export extended service instance
export const extendedApiService = new APIServiceExtended();