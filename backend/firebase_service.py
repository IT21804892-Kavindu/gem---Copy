# import firebase_admin
# from firebase_admin import credentials, firestore
# import json
# import os
# from datetime import datetime, timedelta

# class FirebaseService:
#     def __init__(self):
#         self.db = None
#         self.initialize_firebase()
    
#     def initialize_firebase(self):
#         """Initialize Firebase Admin SDK"""
#         try:
#             # Load Firebase credentials from JSON file
#             credentials_path = os.path.join(os.path.dirname(__file__), 'firebase-credentials.json')
            
#             if not os.path.exists(credentials_path):
#                 raise FileNotFoundError(f"Firebase credentials file not found at: {credentials_path}")
            
#             # Initialize Firebase Admin
#             if not firebase_admin._apps:
#                 cred = credentials.Certificate(credentials_path)
#                 firebase_admin.initialize_app(cred)
            
#             # Initialize Firestore
#             self.db = firestore.client()
#             print("Firebase Admin SDK initialized successfully")
            
#         except Exception as e:
#             print(f"Error initializing Firebase: {e}")
#             self.db = None
    
#     def save_prediction(self, prediction_data):
#         """Save prediction to Firestore"""
#         try:
#             if not self.db:
#                 raise Exception("Firebase not initialized")
            
#             # Add timestamp
#             prediction_data['createdAt'] = firestore.SERVER_TIMESTAMP
            
#             # Save to Firestore
#             doc_ref = self.db.collection('predictions').document()
#             doc_ref.set(prediction_data)
#             print(f"Prediction saved with ID: {doc_ref.id}")
#             return doc_ref.id
            
#         except Exception as e:
#             print(f"Error saving prediction: {e}")
#             raise e
    
#     def get_all_predictions(self):
#         """Get all predictions from Firestore"""
#         try:
#             if not self.db:
#                 raise Exception("Firebase not initialized")
            
#             predictions = []
#             docs = self.db.collection('predictions').order_by('createdAt', direction=firestore.Query.DESCENDING).stream()
            
#             for doc in docs:
#                 data = doc.to_dict()
#                 data['id'] = doc.id
#                 predictions.append(data)
            
#             return predictions
            
#         except Exception as e:
#             print(f"Error fetching predictions: {e}")
#             return []
    
#     def get_last_30_days_predictions(self):
#         """Get predictions from last 30 days"""
#         try:
#             if not self.db:
#                 raise Exception("Firebase not initialized")
            
#             # Calculate 30 days ago
#             thirty_days_ago = datetime.now() - timedelta(days=30)
            
#             predictions = []
#             docs = self.db.collection('predictions').where(
#                 'createdAt', '>=', thirty_days_ago
#             ).order_by('createdAt', direction=firestore.Query.DESCENDING).stream()
            
#             for doc in docs:
#                 data = doc.to_dict()
#                 data['id'] = doc.id
#                 predictions.append(data)
            
#             return predictions
            
#         except Exception as e:
#             print(f"Error fetching 30-day predictions: {e}")
#             return []

# # Global Firebase service instance
# firebase_service = FirebaseService()

import firebase_admin
from firebase_admin import credentials, firestore
import os
from datetime import datetime, timedelta
import logging

# Configure logging
logger = logging.getLogger(__name__)

# This will hold the Firestore client instance after initialization
db = None

def init_firebase():
    """
    Initialize the Firebase Admin SDK and the Firestore client.
    This function should be called once at the start of the application.
    """
    global db
    # Prevent re-initialization if called multiple times
    if db:
        return

    try:
        # Load Firebase credentials from JSON file
        credentials_path = os.path.join(os.path.dirname(__file__), 'firebase-credentials.json')
        
        if not os.path.exists(credentials_path):
            raise FileNotFoundError(f"Firebase credentials file not found at: {credentials_path}")
        
        # Initialize Firebase Admin only if it hasn't been done already
        if not firebase_admin._apps:
            cred = credentials.Certificate(credentials_path)
            firebase_admin.initialize_app(cred)
        
        # Get the Firestore client
        db = firestore.client()
        logger.info("Firebase Admin SDK initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing Firebase: {e}")
        db = None

def save_prediction(prediction_data):
    """Save prediction to Firestore"""
    if not db:
        logger.error("Error saving prediction: Firebase not initialized.")
        raise Exception("Firebase not initialized. Call init_firebase() first.")
    
    try:
        # Add a server-side timestamp
        prediction_data['createdAt'] = firestore.SERVER_TIMESTAMP
        
        # Save to Firestore
        doc_ref = db.collection('predictions').document()
        doc_ref.set(prediction_data)
        logger.info(f"Prediction saved with ID: {doc_ref.id}")
        return doc_ref.id
        
    except Exception as e:
        logger.error(f"Error saving prediction: {e}")
        raise e

def get_all_predictions():
    """Get all predictions from Firestore"""
    if not db:
        logger.error("Error fetching predictions: Firebase not initialized.")
        return []

    try:
        predictions = []
        docs = db.collection('predictions').order_by('createdAt', direction=firestore.Query.DESCENDING).stream()
        
        for doc in docs:
            data = doc.to_dict()
            data['id'] = doc.id
            predictions.append(data)
        
        return predictions
        
    except Exception as e:
        logger.error(f"Error fetching predictions: {e}")
        return []

def get_last_30_days_predictions():
    """Get predictions from last 30 days"""
    if not db:
        logger.error("Error fetching 30-day predictions: Firebase not initialized.")
        return []

    try:
        # Calculate 30 days ago
        thirty_days_ago = datetime.now() - timedelta(days=30)
        
        predictions = []
        docs = db.collection('predictions').where(
            'createdAt', '>=', thirty_days_ago
        ).order_by('createdAt', direction=firestore.Query.DESCENDING).stream()
        
        for doc in docs:
            data = doc.to_dict()
            data['id'] = doc.id
            predictions.append(data)
        
        return predictions
        
    except Exception as e:
        logger.error(f"Error fetching 30-day predictions: {e}")
        return []

def clear_all_predictions():
    """Deletes all documents from the 'predictions' collection."""
    if not db:
        logger.error("Error clearing predictions: Firebase not initialized.")
        raise Exception("Firebase not initialized.")

    try:
        collection_ref = db.collection('predictions')
        docs = collection_ref.stream()
        deleted_count = 0

        # Firestore batch can handle up to 500 operations
        batch = db.batch()
        for doc in docs:
            batch.delete(doc.reference)
            deleted_count += 1
            # Commit the batch every 500 deletes
            if deleted_count % 500 == 0:
                batch.commit()
                batch = db.batch() # Start a new batch

        # Commit any remaining deletes
        if deleted_count % 500 != 0:
            batch.commit()

        logger.info(f"Successfully deleted {deleted_count} predictions.")
        return deleted_count

    except Exception as e:
        logger.error(f"Error clearing predictions: {e}")
        raise e
