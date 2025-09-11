import { collection, addDoc, getDocs, deleteDoc, doc, query, orderBy, where, Timestamp, limit } from 'firebase/firestore';
import { db } from '../config/firebase';
import type { Prediction } from '../App';

export interface PredictionRecord extends Omit<Prediction, 'id'> {
  createdAt: Timestamp;
}

class DatabaseService {
  private collectionName = 'predictions';

  async savePrediction(prediction: Prediction): Promise<void> {
    try {
      const predictionData: PredictionRecord = {
        timestamp: prediction.timestamp,
        premiseIndex: prediction.premiseIndex,
        rainfall: prediction.rainfall,
        temperature: prediction.temperature,
        waterContent: prediction.waterContent,
        riskLevel: prediction.riskLevel,
        confidence: prediction.confidence || 0,
        createdAt: Timestamp.now()
      };

      await addDoc(collection(db, this.collectionName), predictionData);
      console.log('Prediction saved to Firebase');
    } catch (error) {
      console.error('Error saving prediction:', error);
      throw error;
    }
  }

  async getAllPredictions(): Promise<Prediction[]> {
    try {
      const q = query(collection(db, this.collectionName), orderBy('createdAt', 'desc'));
      const querySnapshot = await getDocs(q);
      
      return querySnapshot.docs.map(doc => ({
        id: doc.id,
        timestamp: doc.data().timestamp,
        premiseIndex: doc.data().premiseIndex,
        rainfall: doc.data().rainfall,
        temperature: doc.data().temperature,
        waterContent: doc.data().waterContent,
        riskLevel: doc.data().riskLevel,
        confidence: doc.data().confidence,
        createdAt: doc.data().createdAt?.toDate()
      })) as Prediction[];
    } catch (error) {
      console.error('Error fetching predictions:', error);
      throw error;
    }
  }

  async getLast30DaysPredictions(): Promise<Prediction[]> {
    try {
      const thirtyDaysAgo = new Date();
      thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30);
      
      const q = query(
        collection(db, this.collectionName),
        where('createdAt', '>=', Timestamp.fromDate(thirtyDaysAgo)),
        orderBy('createdAt', 'desc'),
        limit(30)
      );
      
      const querySnapshot = await getDocs(q);
      
      return querySnapshot.docs.map(doc => ({
        id: doc.id,
        timestamp: doc.data().timestamp,
        premiseIndex: doc.data().premiseIndex,
        rainfall: doc.data().rainfall,
        temperature: doc.data().temperature,
        waterContent: doc.data().waterContent,
        riskLevel: doc.data().riskLevel,
        confidence: doc.data().confidence,
        createdAt: doc.data().createdAt?.toDate?.() || new Date()
      })) as Prediction[];
    } catch (error) {
      console.error('Error fetching 30-day predictions:', error);
      return []; // Return empty array instead of throwing error
    }
  }

  async clearAllPredictions(): Promise<void> {
    try {
      const querySnapshot = await getDocs(collection(db, this.collectionName));
      const deletePromises = querySnapshot.docs.map(document =>
        deleteDoc(doc(db, this.collectionName, document.id))
      );

      await Promise.all(deletePromises);
      console.log('All predictions cleared from Firebase');
    } catch (error) {
      console.error('Error clearing predictions:', error);
      throw error;
    }
  }

  async getLatestPrediction(): Promise<Prediction | null> {
    try {
      const q = query(
        collection(db, this.collectionName),
        orderBy('createdAt', 'desc'),
        limit(1)
      );

      const querySnapshot = await getDocs(q);

      if (querySnapshot.empty) {
        return null;
      }

      const doc = querySnapshot.docs[0];
      return {
        id: doc.id,
        timestamp: doc.data().timestamp,
        premiseIndex: doc.data().premiseIndex,
        rainfall: doc.data().rainfall,
        temperature: doc.data().temperature,
        waterContent: doc.data().waterContent,
        riskLevel: doc.data().riskLevel,
        confidence: doc.data().confidence,
        createdAt: doc.data().createdAt?.toDate?.() || new Date()
      } as Prediction;
    } catch (error) {
      console.error('Error fetching latest prediction:', error);
      return null;
    }
  }
}

export const databaseService = new DatabaseService();