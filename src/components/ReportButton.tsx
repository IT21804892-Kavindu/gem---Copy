import React, { useState } from 'react';
import { FileText, Download, Loader2 } from 'lucide-react';
import type { Prediction } from '../App';
import { reportGenerator, ReportData } from '../services/reportGenerator';

interface ReportButtonProps {
  predictions: Prediction[];
  onReportGenerated: () => void;
}

const ReportButton: React.FC<ReportButtonProps> = ({ predictions, onReportGenerated }) => {
  const [isGenerating, setIsGenerating] = useState(false);

  const handleGenerateReport = async () => {
    if (predictions.length === 0) {
      alert('No prediction data available to generate report.');
      return;
    }

    setIsGenerating(true);
    
    try {
      const reportData: ReportData = {
        totalPredictions: predictions.length,
        highRiskAlerts: predictions.filter(p => p.riskLevel === 'high').length,
        avgTemperature: predictions.reduce((acc, p) => acc + p.temperature, 0) / predictions.length,
        avgRainfall: predictions.reduce((acc, p) => acc + p.rainfall, 0) / predictions.length,
        reportDate: new Date().toLocaleString(),
        predictions: predictions
      };

      await reportGenerator.generatePDFReport(reportData);
      
      // Show success message
      alert('Report generated successfully! Check your downloads folder.');

      // Prompt user to clear data
      if (window.confirm('Do you want to clear the data in prediction history?')) {
        onReportGenerated();
      }
      
    } catch (error) {
      console.error('Error generating report:', error);
      alert('Failed to generate report. Please try again.');
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <button
      onClick={handleGenerateReport}
      disabled={isGenerating || predictions.length === 0}
      className={`flex items-center px-6 py-3 rounded-lg font-semibold transition-all duration-200 ${
        predictions.length === 0
          ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
          : isGenerating
          ? 'bg-blue-400 text-white cursor-not-allowed'
          : 'bg-blue-600 hover:bg-blue-700 text-white shadow-lg hover:shadow-xl'
      }`}
    >
      {isGenerating ? (
        <>
          <Loader2 className="w-5 h-5 mr-2 animate-spin" />
          Generating...
        </>
      ) : (
        <>
          <FileText className="w-5 h-5 mr-2" />
          Generate PDF Report
        </>
      )}
    </button>
  );
};

export default ReportButton;