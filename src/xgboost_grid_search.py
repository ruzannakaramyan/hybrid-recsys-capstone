import os
import pandas as pd
import xgboost as xgb
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
    
    dataset_name = args.dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    
    input_file = os.path.join(data_dir, f'xgboost_train_{dataset_name}.parquet')
    
    if not os.path.exists(input_file):
        print(f"ERROR: Data file {input_file} not found. Did you run src/prepare_xgboost_data.py --dataset {dataset_name}?")
        return
    
    # 1. Load Data
    print(f"Loading data for {dataset_name}...")
    df = pd.read_parquet(input_file)
    
    # 2. Prepare Features
    features = ['history_len', 'average_rating', 'rating_number', 'price', 'main_cat_code', 'store_code', 'global_popularity']
    X = df[features]
    y = df['label']
    
    # 3. Train/Validation Split
    print("Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Grid Search (6 trials: 3 depths × 1 lr × 2 n_estimators)
    param_grid = {
        'max_depth': [4, 6, 8],
        'learning_rate': [0.1],
        'n_estimators': [100, 300]
    }
    
    results = []
    best_auc = 0
    best_model = None
    best_params = None

    print(f"\nStarting Grid Search for {dataset_name}...")
    for depth in param_grid['max_depth']:
        for lr in param_grid['learning_rate']:
            for n_est in param_grid['n_estimators']:
                print(f"\n--- Testing: depth={depth}, lr={lr}, n_estimators={n_est} ---")
                
                model = xgb.XGBClassifier(
                    max_depth=depth,
                    learning_rate=lr,
                    n_estimators=n_est,
                    tree_method='hist',
                    device='cpu',
                    random_state=42,
                    n_jobs=-1
                )
                
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                
                # Evaluate
                y_pred_prob = model.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, y_pred_prob)
                print(f"Validation AUC: {auc:.4f}")
                
                results.append({
                    'max_depth': depth,
                    'learning_rate': lr,
                    'n_estimators': n_est,
                    'auc': auc
                })
                
                if auc > best_auc:
                    best_auc = auc
                    best_model = model
                    best_params = (depth, lr, n_est)

    print(f"\n🏆 Best Params: {best_params} with AUC: {best_auc:.4f}")
    
    # 5. Save Results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(data_dir, f'xgboost_grid_results_{dataset_name}.csv'), index=False)
    
    # 6. Save Best Model
    model_path = os.path.join(script_dir, f'xgboost_pure_{dataset_name}_best.json')
    best_model.save_model(model_path)
    print(f"Best model saved to {model_path}")

if __name__ == "__main__":
    main()
