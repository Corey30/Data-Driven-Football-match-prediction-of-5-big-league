import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_data, preprocess_data, clean_data
from src.data.splitter import time_series_split
from src.features.halftime_features import (
    process_halftime_features
)
from src.training.trainer import Trainer
from src.evaluation.metrics import compute_class_distribution
from src.evaluation.visualization import generate_model_comparison_report
from src.config.config import MODEL_DISPLAY_NAMES


def train_and_evaluate_model(model_type, X_train, y_train, X_val, y_val, X_test, y_test):
    display_name = MODEL_DISPLAY_NAMES.get(model_type, model_type)
    print(f"\n{'='*60}")
    print(f"训练模型: {display_name} ({model_type})")
    print('='*60)
    
    trainer = Trainer(model_type=model_type, task_name='halftime')
    train_metrics, val_metrics = trainer.train(X_train, y_train, X_val, y_val)
    
    print("\n训练集指标:")
    for k, v in train_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    print("\n验证集指标:")
    for k, v in val_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    test_metrics, y_pred, y_proba = trainer.evaluate(X_test, y_test)
    
    print("\n测试集指标:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    model_path = trainer.save_model()
    print(f"\n模型保存至: {model_path}")
    
    importance = trainer.get_feature_importance()
    if importance:
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        print("\nTop 10 重要特征:")
        for feat, imp in sorted_importance[:10]:
            print(f"  {feat}: {imp:.4f}")
    
    return {
        'trainer': trainer,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'feature_importance': importance
    }


def print_comparison_summary(models_results):
    model_names = list(models_results.keys())
    
    print("\n" + "=" * 80)
    print("模型对比汇总")
    print("=" * 80)
    
    metrics_to_compare = ['accuracy', 'f1_macro', 'f1_weighted', 'log_loss', 'precision_macro', 'recall_macro']
    
    header = f"{'指标':<18}"
    for name in model_names:
        display_name = MODEL_DISPLAY_NAMES.get(name, name)
        header += f"{display_name[:12]:>14}"
    print(f"\n{header}")
    print("-" * (18 + 14 * len(model_names)))
    
    for metric in metrics_to_compare:
        line = f"{metric:<18}"
        for name in model_names:
            val = models_results[name]['test_metrics'].get(metric, 0)
            line += f"{val:>14.4f}"
        print(line)
    
    best_model = max(model_names, key=lambda x: models_results[x]['test_metrics'].get('f1_macro', 0))
    best_f1 = models_results[best_model]['test_metrics'].get('f1_macro', 0)
    best_display = MODEL_DISPLAY_NAMES.get(best_model, best_model)
    print(f"\n最佳模型 (F1-macro): {best_display} = {best_f1:.4f}")


def main():
    print("=" * 60)
    print("上半场预测模型训练 - 多模型对比")
    print("=" * 60)
    
    print("\n[1] 加载数据...")
    df = load_data()
    print(f"原始数据形状: {df.shape}")
    
    print("\n[2] 数据预处理...")
    df = preprocess_data(df)
    df = clean_data(df)
    print(f"处理后数据形状: {df.shape}")
    
    print("\n[3] 时间序列划分...")
    train_df, val_df, test_df = time_series_split(df)
    print(f"训练集: {len(train_df)} 条")
    print(f"验证集: {len(val_df)} 条")
    print(f"测试集: {len(test_df)} 条")
    
    print("\n[4] 构建上半场特征...")
    X_train, y_train = process_halftime_features(train_df)
    X_val, y_val = process_halftime_features(val_df)
    X_test, y_test = process_halftime_features(test_df)
    
    print(f"特征数量: {X_train.shape[1]}")
    
    print("\n[5] 类别分布...")
    print("训练集:", compute_class_distribution(y_train))
    print("验证集:", compute_class_distribution(y_val))
    print("测试集:", compute_class_distribution(y_test))
    
    models_results = {}
    model_types = ['random_forest', 'logistic_regression', 'xgboost', 
                   'naive_bayes', 'svm', 'fnn']
    
    for i, model_type in enumerate(model_types, 1):
        print(f"\n[{i}/{len(model_types)}] 训练 {MODEL_DISPLAY_NAMES.get(model_type, model_type)}...")
        try:
            models_results[model_type] = train_and_evaluate_model(
                model_type, X_train, y_train, X_val, y_val, X_test, y_test
            )
        except ImportError as e:
            print(f"跳过 {model_type}: {e}")
        except Exception as e:
            print(f"训练 {model_type} 时出错: {e}")
    
    print_comparison_summary(models_results)
    
    print(f"\n[{len(models_results)+1}] 生成模型对比可视化报告...")
    generate_model_comparison_report(
        task_name='halftime',
        models_results=models_results,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test
    )
    
    print("\n" + "=" * 60)
    print("上半场预测模型训练完成")
    print("=" * 60)


if __name__ == '__main__':
    main()
