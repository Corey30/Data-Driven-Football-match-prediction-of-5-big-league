import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_data, preprocess_data, clean_data
from src.data.splitter import time_series_split
from src.features.prematch_features import (
    build_prematch_features, handle_missing_values
)
from src.training.trainer import Trainer
from src.evaluation.metrics import compute_class_distribution
from src.evaluation.visualization import generate_training_report


def main():
    print("=" * 60)
    print("赛前预测模型训练")
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
    
    print("\n[4] 构建赛前特征...")
    X_train, y_train = build_prematch_features(train_df)
    X_val, y_val = build_prematch_features(val_df)
    X_test, y_test = build_prematch_features(test_df)
    
    X_train = handle_missing_values(X_train)
    X_val = handle_missing_values(X_val)
    X_test = handle_missing_values(X_test)
    
    print(f"特征数量: {X_train.shape[1]}")
    
    print("\n[5] 类别分布...")
    print("训练集:", compute_class_distribution(y_train))
    print("验证集:", compute_class_distribution(y_val))
    print("测试集:", compute_class_distribution(y_test))
    
    print("\n[6] 训练模型...")
    trainer = Trainer(model_type='random_forest', task_name='prematch')
    train_metrics, val_metrics = trainer.train(X_train, y_train, X_val, y_val)
    
    print("\n训练集指标:")
    for k, v in train_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    print("\n验证集指标:")
    for k, v in val_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    print("\n[7] 测试集评估...")
    test_metrics, y_pred, y_proba = trainer.evaluate(X_test, y_test)
    
    print("\n测试集指标:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    print("\n[8] 保存模型...")
    model_path = trainer.save_model()
    print(f"模型保存至: {model_path}")
    
    print("\n[9] 特征重要性...")
    importance = trainer.get_feature_importance()
    if importance:
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        print("Top 10 重要特征:")
        for feat, imp in sorted_importance[:10]:
            print(f"  {feat}: {imp:.4f}")
    
    print("\n[10] 生成可视化报告...")
    generate_training_report(
        task_name='prematch',
        model_type='random_forest',
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        y_pred_test=y_pred,
        feature_importance=importance
    )
    
    print("\n" + "=" * 60)
    print("赛前预测模型训练完成")
    print("=" * 60)


if __name__ == '__main__':
    main()
