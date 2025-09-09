import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';
import type { Prediction } from '../App';

export interface ReportData {
  totalPredictions: number;
  highRiskAlerts: number;
  avgTemperature: number;
  avgRainfall: number;
  reportDate: string;
  predictions: Prediction[];
}

class ReportGenerator {
  async generatePDFReport(reportData: ReportData): Promise<void> {
    try {
      const pdf = new jsPDF('p', 'mm', 'a4');
      const pageWidth = pdf.internal.pageSize.getWidth();
      const pageHeight = pdf.internal.pageSize.getHeight();
      let yPosition = 20;

      // Header
      pdf.setFontSize(20);
      pdf.setFont('helvetica', 'bold');
      pdf.text('Dengue Breeding Prediction Report', pageWidth / 2, yPosition, { align: 'center' });
      
      yPosition += 15;
      pdf.setFontSize(12);
      pdf.setFont('helvetica', 'normal');
      pdf.text(`Generated on: ${reportData.reportDate}`, pageWidth / 2, yPosition, { align: 'center' });

      // Summary Statistics
      yPosition += 20;
      pdf.setFontSize(16);
      pdf.setFont('helvetica', 'bold');
      pdf.text('Summary Statistics', 20, yPosition);

      yPosition += 10;
      pdf.setFontSize(12);
      pdf.setFont('helvetica', 'normal');
      
      const summaryData = [
        ['Total Predictions:', reportData.totalPredictions.toString()],
        ['High Risk Alerts:', reportData.highRiskAlerts.toString()],
        ['Average Temperature:', `${reportData.avgTemperature.toFixed(1)}°C`],
        ['Average Rainfall:', `${reportData.avgRainfall.toFixed(1)}mm`]
      ];

      summaryData.forEach(([label, value]) => {
        yPosition += 8;
        pdf.text(label, 20, yPosition);
        pdf.text(value, 80, yPosition);
      });

      // Risk Distribution Chart
      yPosition += 20;
      pdf.setFontSize(16);
      pdf.setFont('helvetica', 'bold');
      pdf.text('Risk Level Distribution', 20, yPosition);

      const riskCounts = {
        low: reportData.predictions.filter(p => p.riskLevel === 'low').length,
        medium: reportData.predictions.filter(p => p.riskLevel === 'medium').length,
        high: reportData.predictions.filter(p => p.riskLevel === 'high').length
      };

      yPosition += 15;
      pdf.setFontSize(12);
      pdf.setFont('helvetica', 'normal');

      // Simple bar chart representation
      const chartWidth = 120;
      const chartHeight = 40;
      const maxCount = Math.max(riskCounts.low, riskCounts.medium, riskCounts.high);

      if (maxCount > 0) {
        const barWidth = chartWidth / 3 - 10;
        const colors = ['#059669', '#d97706', '#dc2626']; // green, yellow, red
        const labels = ['Low', 'Medium', 'High'];
        const counts = [riskCounts.low, riskCounts.medium, riskCounts.high];

        counts.forEach((count, index) => {
          const barHeight = (count / maxCount) * chartHeight;
          const x = 20 + index * (barWidth + 10);
          const y = yPosition + chartHeight - barHeight;

          // Draw bar
          pdf.setFillColor(colors[index]);
          pdf.rect(x, y, barWidth, barHeight, 'F');

          // Add label and count
          pdf.setFontSize(10);
          pdf.text(labels[index], x + barWidth / 2, yPosition + chartHeight + 8, { align: 'center' });
          pdf.text(count.toString(), x + barWidth / 2, yPosition + chartHeight + 15, { align: 'center' });
        });
      }

      // Recent Predictions Table
      yPosition += 70;
      if (yPosition > pageHeight - 60) {
        pdf.addPage();
        yPosition = 20;
      }

      pdf.setFontSize(16);
      pdf.setFont('helvetica', 'bold');
      pdf.text('Recent Predictions', 20, yPosition);

      yPosition += 15;
      pdf.setFontSize(10);
      pdf.setFont('helvetica', 'bold');

      // Table headers
      const headers = ['Date/Time', 'Premise Index', 'Risk Level', 'Temperature', 'Rainfall', 'Water Content'];
      const colWidths = [40, 25, 25, 25, 25, 25];
      let xPos = 20;

      headers.forEach((header, index) => {
        pdf.text(header, xPos, yPosition);
        xPos += colWidths[index];
      });

      // Table data
      pdf.setFont('helvetica', 'normal');
      const recentPredictions = reportData.predictions.slice(0, 15); // Show last 15 predictions

      recentPredictions.forEach((prediction, index) => {
        yPosition += 8;
        if (yPosition > pageHeight - 20) {
          pdf.addPage();
          yPosition = 30;
        }

        xPos = 20;
        const rowData = [
          prediction.timestamp,
          `${prediction.premiseIndex}%`,
          prediction.riskLevel.toUpperCase(),
          `${prediction.temperature}°C`,
          `${prediction.rainfall}mm`,
          `${prediction.waterContent}`
        ];

        rowData.forEach((data, colIndex) => {
          pdf.text(data, xPos, yPosition);
          xPos += colWidths[colIndex];
        });
      });

      // Footer
      const totalPages = pdf.getNumberOfPages();
      for (let i = 1; i <= totalPages; i++) {
        pdf.setPage(i);
        pdf.setFontSize(8);
        pdf.setFont('helvetica', 'normal');
        pdf.text(
          `Page ${i} of ${totalPages} - Dengue Breeding Prediction System`,
          pageWidth / 2,
          pageHeight - 10,
          { align: 'center' }
        );
      }

      // Save the PDF
      const fileName = `dengue-prediction-report-${new Date().toISOString().split('T')[0]}.pdf`;
      pdf.save(fileName);

    } catch (error) {
      console.error('Error generating PDF report:', error);
      throw error;
    }
  }
}

export const reportGenerator = new ReportGenerator();