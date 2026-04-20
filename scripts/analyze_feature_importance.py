import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.data.loader import load_data, preprocess_data, clean_data
from src.data.splitter import time_series_split
from src.features.halftime_features import process_halftime_features
from src.training.trainer import Trainer
from src.config.config import MODEL_PARAMS, MODEL_DISPLAY_NAMES


def analyze_feature_importance():
    print("=" * 60)
    print("特征重要性分析")
    print("=" * 60)
    
    print("\n[1] 加载数据...")
    df = load_data()
    df = preprocess_data(df)
    df = clean_data(df)
    
    train_df, val_df, test_df = time_series_split(df)
    X_train, y_train = process_halftime_features(train_df)
    X_val, y_val = process_halftime_features(val_df)
    X_test, y_test = process_halftime_features(test_df)
    
    print(f"特征数量: {X_train.shape[1]}")
    print(f"训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")
    
    models_with_importance = ['random_forest', 'logistic_regression', 'xgboost', 'svm']
    all_importance = {}
    
    for model_type in models_with_importance:
        display_name = MODEL_DISPLAY_NAMES.get(model_type, model_type)
        print(f"\n{'='*60}")
        print(f"训练 {display_name}...")
        print('='*60)
        
        trainer = Trainer(
            model_type=model_type,
            params=MODEL_PARAMS.get(model_type, {}),
            task_name='feature_analysis'
        )
        
        trainer.train(X_train, y_train, X_val, y_val)
        
        importance = trainer.get_feature_importance()
        if importance:
            all_importance[model_type] = importance
            
            sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            
            print(f"\n{display_name} - Top 15 重要特征:")
            print("-" * 50)
            for i, (feature, score) in enumerate(sorted_importance[:15], 1):
                print(f"  {i:2d}. {feature:<40} {score:.4f}")
        else:
            print(f"  {display_name} 不支持特征重要性分析")
    
    if all_importance:
        print("\n" + "=" * 60)
        print("综合特征重要性分析")
        print("=" * 60)
        
        feature_scores = {}
        for model_type, importance in all_importance.items():
            for feature, score in importance.items():
                if feature not in feature_scores:
                    feature_scores[feature] = {}
                feature_scores[feature][model_type] = score
        
        avg_importance = {}
        for feature, scores in feature_scores.items():
            avg_importance[feature] = np.mean(list(scores.values()))
        
        sorted_avg = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
        
        print("\n平均重要性 Top 20 特征:")
        print("-" * 70)
        print(f"{'排名':<4} {'特征名称':<40} {'平均重要性':>10} {'RF':>8} {'LR':>8} {'XGB':>8} {'SVM':>8}")
        print("-" * 70)
        
        for i, (feature, avg_score) in enumerate(sorted_avg[:20], 1):
            scores = feature_scores[feature]
            rf = scores.get('random_forest', 0)
            lr = scores.get('logistic_regression', 0)
            xgb = scores.get('xgboost', 0)
            svm = scores.get('svm', 0)
            print(f"{i:<4} {feature:<40} {avg_score:>10.4f} {rf:>8.4f} {lr:>8.4f} {xgb:>8.4f} {svm:>8.4f}")
        
        print("\n" + "=" * 60)
        print("特征重要性可视化")
        print("=" * 60)
        
        top_features = [f[0] for f in sorted_avg[:15]]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Feature Importance Comparison Across Models', fontsize=14, fontweight='bold')
        
        for idx, model_type in enumerate(models_with_importance):
            ax = axes[idx // 2, idx % 2]
            
            if model_type in all_importance:
                importance = all_importance[model_type]
                features = []
                scores = []
                for f in top_features:
                    if f in importance:
                        features.append(f[:30])
                        scores.append(importance[f])
                
                y_pos = np.arange(len(features))
                ax.barh(y_pos, scores, align='center', color='steelblue')
                ax.set_yticks(y_pos)
                ax.set_yticklabels(features, fontsize=8)
                ax.invert_yaxis()
                ax.set_xlabel('Importance Score')
                ax.set_title(MODEL_DISPLAY_NAMES.get(model_type, model_type))
        
        plt.tight_layout()
        
        output_dir = Path(__file__).resolve().parent.parent / 'outputs' / 'reports'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = output_dir / 'feature_importance_analysis.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n可视化图表已保存至: {save_path}")
        
        summary_data = []
        for feature, avg_score in sorted_avg:
            row = {'feature': feature, 'avg_importance': avg_score}
            for model_type in models_with_importance:
                row[model_type] = feature_scores[feature].get(model_type, 0)
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        csv_path = output_dir / 'feature_importance_summary.csv'
        summary_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"特征重要性汇总已保存至: {csv_path}")
    
    print("\n" + "=" * 60)
    print("特征重要性分析完成")
    print("=" * 60)


if __name__ == "__main__":
    analyze_feature_importance()
