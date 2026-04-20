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


def compute_feature_importance(X_train, y_train):
    models_with_importance = ['random_forest', 'logistic_regression', 'xgboost', 'svm']
    all_importance = {}

    for model_type in models_with_importance:
        try:
            trainer = Trainer(
                model_type=model_type,
                params=MODEL_PARAMS.get(model_type, {}),
                task_name='corr_analysis'
            )
            trainer.train(X_train, y_train, X_train[:1], y_train[:1])
            importance = trainer.get_feature_importance()
            if importance:
                all_importance[model_type] = importance
        except Exception:
            continue

    if not all_importance:
        return {}

    feature_scores = {}
    for model_type, importance in all_importance.items():
        for feature, score in importance.items():
            if feature not in feature_scores:
                feature_scores[feature] = {}
            feature_scores[feature][model_type] = score

    avg_importance = {}
    for feature, scores in feature_scores.items():
        avg_importance[feature] = np.mean(list(scores.values()))

    return avg_importance


def build_correlation_groups(high_corr_df):
    parent = {}

    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    all_features = set()
    for _, row in high_corr_df.iterrows():
        f1, f2 = row['feature_1'], row['feature_2']
        all_features.add(f1)
        all_features.add(f2)
        union(f1, f2)

    for f in all_features:
        if f not in parent:
            parent[f] = f

    groups = {}
    for f in all_features:
        root = find(f)
        if root not in groups:
            groups[root] = set()
        groups[root].add(f)

    sorted_groups = sorted(groups.values(), key=lambda g: len(g), reverse=True)
    return sorted_groups


def analyze_feature_correlation():
    print("=" * 80)
    print("特征相关性分析报告")
    print("=" * 80)

    print("\n[1] 加载数据...")
    df = load_data()
    df = preprocess_data(df)
    df = clean_data(df)

    train_df, val_df, test_df = time_series_split(df)
    X_train, y_train = process_halftime_features(train_df)

    print(f"特征数量: {X_train.shape[1]}")
    print(f"训练样本数: {len(X_train)}")
    print(f"样本/特征比: {len(X_train) / X_train.shape[1]:.1f}:1")

    print("\n" + "=" * 80)
    print("[2] 计算特征重要性 (用于后续决策)")
    print("=" * 80)
    importance_dict = compute_feature_importance(X_train, y_train)
    print(f"已计算 {len(importance_dict)} 个特征的重要性")

    print("\n" + "=" * 80)
    print("[3] 高相关性特征对分析 (|相关系数| > 0.8)")
    print("=" * 80)

    corr_matrix = X_train.corr()

    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.8:
                high_corr_pairs.append({
                    'feature_1': corr_matrix.columns[i],
                    'feature_2': corr_matrix.columns[j],
                    'correlation': corr_val,
                    'abs_correlation': abs(corr_val)
                })

    high_corr_df = pd.DataFrame(high_corr_pairs)
    high_corr_df = high_corr_df.sort_values('abs_correlation', ascending=False)

    print(f"\n发现 {len(high_corr_df)} 对高相关性特征:")
    print("-" * 90)
    print(f"{'特征1':<42} {'特征2':<42} {'相关系数':>8} {'|系数|':>8}")
    print("-" * 90)

    for _, row in high_corr_df.iterrows():
        f1 = row['feature_1'][:39] + '...' if len(row['feature_1']) > 42 else row['feature_1']
        f2 = row['feature_2'][:39] + '...' if len(row['feature_2']) > 42 else row['feature_2']
        print(f"{f1:<42} {f2:<42} {row['correlation']:>8.4f} {row['abs_correlation']:>8.4f}")

    print("\n" + "=" * 80)
    print("[4] 特征分组分析 (Union-Find 精确聚类)")
    print("=" * 80)

    groups = build_correlation_groups(high_corr_df)

    print(f"\n识别出 {len(groups)} 个特征组 (每组内特征互相关联):")
    for gi, group in enumerate(groups, 1):
        sorted_features = sorted(group, key=lambda f: importance_dict.get(f, 0), reverse=True)
        print(f"\n  Group {gi} ({len(group)} 个特征):")
        for rank, f in enumerate(sorted_features, 1):
            imp = importance_dict.get(f, 0)
            marker = " ★" if rank == 1 else ""
            print(f"    {rank:2d}. {f:<45} 重要性={imp:.4f}{marker}")

    print("\n" + "=" * 80)
    print("[5] 每组建议保留/移除 (保留重要性最高的1个)")
    print("=" * 80)

    suggested_to_drop = []
    group_details = []

    for gi, group in enumerate(groups, 1):
        sorted_features = sorted(group, key=lambda f: importance_dict.get(f, 0), reverse=True)
        keep = sorted_features[0]
        drops = sorted_features[1:]
        suggested_to_drop.extend(drops)

        group_details.append({
            'group': gi,
            'size': len(group),
            'keep': keep,
            'keep_importance': importance_dict.get(keep, 0),
            'drop': drops
        })

        print(f"\n  Group {gi} (保留 1/{len(group)}):")
        print(f"    ✅ 保留: {keep:<45} (重要性: {importance_dict.get(keep, 0):.4f})")
        print(f"    ❌ 移除:")
        for f in drops:
            imp = importance_dict.get(f, 0)
            print(f"         {f:<45} (重要性: {imp:.4f})")

    print("\n" + "=" * 80)
    print("[6] 汇总")
    print("=" * 80)

    final_feature_count = X_train.shape[1] - len(suggested_to_drop)
    print(f"\n  原始特征数:       {X_train.shape[1]}")
    print(f"  建议移除特征数:   {len(suggested_to_drop)}")
    print(f"  移除后特征数:     {final_feature_count}")
    print(f"  样本/特征比:      {len(X_train) / X_train.shape[1]:.1f}:1 -> {len(X_train) / final_feature_count:.1f}:1")

    print(f"\n  完整移除列表 (按重要性排序):")
    print("  " + "-" * 70)
    sorted_drops = sorted(suggested_to_drop, key=lambda f: importance_dict.get(f, 0))
    for f in sorted_drops:
        imp = importance_dict.get(f, 0)
        print(f"    {f:<50} 重要性={imp:.4f}")

    print("\n" + "=" * 80)
    print("[7] 保留特征列表")
    print("=" * 80)

    kept_features = [f for f in X_train.columns if f not in suggested_to_drop]
    print(f"\n  保留 {len(kept_features)} 个特征:")
    print("  " + "-" * 70)
    for f in sorted(kept_features, key=lambda x: importance_dict.get(x, 0), reverse=True):
        imp = importance_dict.get(f, 0)
        print(f"    {f:<50} 重要性={imp:.4f}")

    print("\n" + "=" * 80)
    print("[8] 生成可视化与文件")
    print("=" * 80)

    output_dir = Path(__file__).resolve().parent.parent / 'outputs' / 'reports'
    output_dir.mkdir(parents=True, exist_ok=True)

    top_corr_features = set()
    for _, row in high_corr_df.head(40).iterrows():
        top_corr_features.add(row['feature_1'])
        top_corr_features.add(row['feature_2'])

    top_corr_list = sorted(top_corr_features)[:30]
    if top_corr_list:
        sub_corr = corr_matrix.loc[top_corr_list, top_corr_list]

        fig, ax = plt.subplots(figsize=(18, 16))
        n = len(top_corr_list)
        mask = np.triu(np.ones((n, n), dtype=bool))

        plot_data = sub_corr.values.copy()
        plot_data[mask] = np.nan

        im = ax.imshow(plot_data, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)

        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        short_labels = [f[:22] for f in top_corr_list]
        ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(short_labels, fontsize=7)

        for i in range(n):
            for j in range(n):
                if not mask[i, j] and not np.isnan(plot_data[i, j]):
                    val = plot_data[i, j]
                    color = 'white' if abs(val) > 0.6 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=5, color=color)

        cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
        cbar.ax.set_ylabel('Pearson Correlation', rotation=-90, va='bottom', fontsize=10)

        plt.title('Feature Correlation Heatmap (High-Correlation Features)', fontsize=13)
        plt.tight_layout()

        heatmap_path = output_dir / 'feature_correlation_heatmap.png'
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  热力图: {heatmap_path}")

    high_corr_df.to_csv(output_dir / 'high_correlation_pairs.csv', index=False, encoding='utf-8-sig')
    print(f"  高相关对CSV: {output_dir / 'high_correlation_pairs.csv'}")

    with open(output_dir / 'feature_correlation_report.txt', 'w', encoding='utf-8') as f:
        f.write("特征相关性分析报告\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"原始特征数: {X_train.shape[1]}\n")
        f.write(f"建议移除特征数: {len(suggested_to_drop)}\n")
        f.write(f"移除后特征数: {final_feature_count}\n")
        f.write(f"样本/特征比: {len(X_train) / X_train.shape[1]:.1f}:1 -> {len(X_train) / final_feature_count:.1f}:1\n\n")

        f.write("特征分组详情:\n")
        f.write("-" * 70 + "\n")
        for detail in group_details:
            f.write(f"\nGroup {detail['group']} ({detail['size']} 个特征):\n")
            f.write(f"  保留: {detail['keep']} (重要性: {detail['keep_importance']:.4f})\n")
            f.write(f"  移除:\n")
            for feat in detail['drop']:
                imp = importance_dict.get(feat, 0)
                f.write(f"    - {feat} (重要性: {imp:.4f})\n")

        f.write(f"\n\n建议移除的特征列表:\n")
        f.write("-" * 70 + "\n")
        for feat in sorted_drops:
            f.write(f"{feat}\n")

        f.write(f"\n\n保留的特征列表:\n")
        f.write("-" * 70 + "\n")
        for feat in sorted(kept_features, key=lambda x: importance_dict.get(x, 0), reverse=True):
            imp = importance_dict.get(feat, 0)
            f.write(f"{feat} (重要性: {imp:.4f})\n")

    print(f"  完整报告: {output_dir / 'feature_correlation_report.txt'}")

    print("\n" + "=" * 80)
    print("分析完成")
    print("=" * 80)

    return high_corr_df, suggested_to_drop


if __name__ == "__main__":
    analyze_feature_correlation()
