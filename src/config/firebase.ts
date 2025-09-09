import { initializeApp } from 'firebase/app';
import { getFirestore } from 'firebase/firestore';
import { getAnalytics } from 'firebase/analytics';

const firebaseConfig = {
  apiKey: "AIzaSyCvpKEXaRl9uNuqPi2E2pVEp9xiGjNhfCk",
  authDomain: "prediction-20333.firebaseapp.com",
  projectId: "prediction-20333",
  storageBucket: "prediction-20333.firebasestorage.app",
  messagingSenderId: "856570120405",
  appId: "1:856570120405:web:64fe401b571343c4c17a5c",
  measurementId: "G-JNNDTX38QS"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

// Initialize Firestore
export const db = getFirestore(app);

// Initialize Analytics (optional)
export const analytics = getAnalytics(app);

export default app;