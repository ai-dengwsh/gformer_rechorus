import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import seaborn as sns
from dataset_results import DATASET_RESULTS, DATASET_DESCRIPTIONS, DATASET_CHARACTERISTICS

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def create_dataset_comparison():
    """创建数据集特征对比图"""
    datasets = list(DATASET_RESULTS.keys())
    stats = ['users', 'items', 'interactions', 'sparsity']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('数据集特征对比', fontsize=16)
    
    for idx, stat in enumerate(stats):
        ax = axes[idx//2, idx%2]
        values = [DATASET_RESULTS[ds]['dataset_stats'][stat] for ds in datasets]
        
        if stat == 'sparsity':
            values = [v * 100 for v in values]  # 转换为百分比
            
        ax.bar(datasets, values)
        ax.set_title(f'数据集{stat.capitalize()}对比')
        ax.tick_params(axis='x', rotation=45)
        
        if stat == 'sparsity':
            ax.set_ylabel('稀疏度 (%)')
        else:
            ax.set_ylabel('数量')
            
    plt.tight_layout()
    plt.savefig('figures/dataset_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_metrics_heatmap():
    """创建性能指标热力图"""
    metrics = ['P@10', 'R@10', 'N@10', 'P@20', 'R@20', 'N@20', 'P@50', 'R@50', 'N@50']
    datasets = list(DATASET_RESULTS.keys())
    
    # 构建测试集结果矩阵
    matrix = np.zeros((len(datasets), len(metrics)))
    for i, dataset in enumerate(datasets):
        for j, metric in enumerate(metrics):
            matrix[i, j] = DATASET_RESULTS[dataset]['metrics']['test'][metric]
    
    plt.figure(figsize=(15, 8))
    sns.heatmap(matrix, annot=True, fmt='.4f', 
                xticklabels=metrics, 
                yticklabels=datasets,
                cmap='YlOrRd')
    plt.title('不同数据集的性能指标对比')
    plt.xlabel('评估指标')
    plt.ylabel('数据集')
    plt.tight_layout()
    plt.savefig('figures/metrics_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_training_loss_comparison():
    """创建训练损失对比图"""
    plt.figure(figsize=(12, 6))
    
    for dataset in DATASET_RESULTS.keys():
        loss = DATASET_RESULTS[dataset]['train_loss']
        epochs = range(len(loss))
        plt.plot(epochs, loss, marker='o', label=dataset)
    
    plt.title('不同数据集的训练损失对比', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('figures/training_loss_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_dataset_characteristics_radar():
    """创建数据集特征雷达图"""
    characteristics = ['avg_ratings_per_user', 'avg_ratings_per_item', 'sparsity']
    datasets = list(DATASET_RESULTS.keys())
    
    angles = np.linspace(0, 2*np.pi, len(characteristics), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # 闭合图形
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    for dataset in datasets:
        values = [
            DATASET_CHARACTERISTICS[dataset]['avg_ratings_per_user'],
            DATASET_CHARACTERISTICS[dataset]['avg_ratings_per_item'],
            DATASET_RESULTS[dataset]['dataset_stats']['sparsity'] * 100
        ]
        values = np.concatenate((values, [values[0]]))  # 闭合图形
        ax.plot(angles, values, 'o-', linewidth=2, label=dataset)
        ax.fill(angles, values, alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(['平均用户评分数', '平均物品评分数', '稀疏度(%)'])
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('数据集特征雷达图对比')
    plt.tight_layout()
    plt.savefig('figures/dataset_characteristics_radar.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_performance_comparison_by_k():
    """创建不同K值下的性能对比图"""
    k_values = [10, 20, 50]
    metrics = ['P', 'R', 'N']
    datasets = list(DATASET_RESULTS.keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('不同K值下的性能指标对比', fontsize=16)
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        x = np.arange(len(k_values))
        width = 0.25
        
        for i, dataset in enumerate(datasets):
            values = [DATASET_RESULTS[dataset]['metrics']['test'][f'{metric}@{k}'] 
                     for k in k_values]
            ax.bar(x + i*width, values, width, label=dataset)
        
        ax.set_title(f'{metric}@K')
        ax.set_xticks(x + width)
        ax.set_xticklabels(k_values)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)
        
    plt.tight_layout()
    plt.savefig('figures/performance_comparison_by_k.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # 创建figures目录
    import os
    os.makedirs('figures', exist_ok=True)
    
    # 生成所有可视化
    create_dataset_comparison()
    create_metrics_heatmap()
    create_training_loss_comparison()
    create_dataset_characteristics_radar()
    create_performance_comparison_by_k()
    
    print("所有可视化图表已生成在'figures'目录中。") 