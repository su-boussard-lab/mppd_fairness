import pickle


class TestPredictions:
    """
    Container class for test predictions and data.
    
    Attributes:
        X_test: Test features
        y_test: True test labels
        y_pred: Binary predictions
        y_probs: Probability predictions
    """
    def __init__(self, model, X_test, y_test, y_pred, y_probs):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = y_pred
        self.y_probs = y_probs
    
    @classmethod
    def load(cls, filename):
        """Load test predictions from a pickle file"""
        if filename.endswith('.pkl'):
            filepath = filename
        else:
            filepath = f'{filename}.pkl'
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            return cls(
                data['model'],
                data['X_test'],
                data['y_test'],
                data['y_pred'],
                data['y_probs']
            )
    
    def save(self, filename):
        """Save test predictions to a pickle file"""
        to_save = {
            'model': self.model,
            'X_test': self.X_test,
            'y_test': self.y_test,
            'y_pred': self.y_pred,
            'y_probs': self.y_probs
        }
        with open(f'{filename}.pkl', 'wb') as f:
            pickle.dump(to_save, f)

