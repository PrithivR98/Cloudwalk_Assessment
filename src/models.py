from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from sklearn.metrics import accuracy_score

class ModelTrainer:
    def __init__(self, n_splits=5):
        self.n_splits = n_splits

    def train_with_stratified_kfold(self, X, y):
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        lr_scores, svm_scores, lgb_scores = [], [], []

        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            # LightGBM
            print("Training Initiated")
            lgb_train = lgb.Dataset(X_tr, label=y_tr)
            lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
            params = {
                'objective': 'multiclass',
                'num_class': 10,
                'metric': 'multi_logloss',
                'learning_rate': 0.1,
                'num_leaves': 31
            }
            lgb_model = lgb.train(params, lgb_train, num_boost_round=100,
                                  valid_sets=[lgb_val])
            lgb_scores.append(accuracy_score(y_val, lgb_model.predict(X_val).argmax(axis=1)))

        return {
            "LightGBM": sum(lgb_scores) / len(lgb_scores)
        }
