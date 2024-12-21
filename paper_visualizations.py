import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 设置全局样式
plt.style.use('default')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

def create_ablation_study():
    """创建消融实验对比图（类似论文Figure 3）"""
    models = ['−Ead', 'MM', 'RM', '-TE', 'Ours']
    
    # Yelp数据集结果
    yelp_recall = [0.075, 0.080, 0.078, 0.082, 0.088]
    yelp_ndcg = [0.035, 0.040, 0.038, 0.042, 0.044]
    
    # Ifashion数据集结果
    ifashion_recall = [0.060, 0.075, 0.070, 0.080, 0.090]
    ifashion_ndcg = [0.045, 0.050, 0.048, 0.055, 0.060]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Yelp数据集图表
    ax1.bar(np.arange(len(models)), yelp_recall, color=['lightgray', 'wheat', 'lightcoral', 'lightblue', 'royalblue'])
    ax1.set_title('(a) Yelp Data')
    ax1.set_ylabel('Recall@20')
    ax1.set_xticks(np.arange(len(models)))
    ax1.set_xticklabels(models)
    
    ax2.bar(np.arange(len(models)), yelp_ndcg, color=['lightgray', 'wheat', 'lightcoral', 'lightblue', 'royalblue'])
    ax2.set_ylabel('NDCG@20')
    ax2.set_xticks(np.arange(len(models)))
    ax2.set_xticklabels(models)
    
    # Ifashion数据集图表
    ax3.bar(np.arange(len(models)), ifashion_recall, color=['lightgray', 'wheat', 'lightcoral', 'lightblue', 'royalblue'])
    ax3.set_title('(b) Ifashion Data')
    ax3.set_ylabel('Recall@20')
    ax3.set_xticks(np.arange(len(models)))
    ax3.set_xticklabels(models)
    
    ax4.bar(np.arange(len(models)), ifashion_ndcg, color=['lightgray', 'wheat', 'lightcoral', 'lightblue', 'royalblue'])
    ax4.set_ylabel('NDCG@20')
    ax4.set_xticks(np.arange(len(models)))
    ax4.set_xticklabels(models)
    
    plt.tight_layout()
    plt.savefig('figures/ablation_study.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_noise_robustness():
    """创建噪声鲁棒性实验图（类似论文Figure 4）"""
    noise_ratio = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    # Yelp数据集结果
    yelp_gformer = [1.00, 0.98, 0.96, 0.94, 0.92, 0.90]
    yelp_sgl = [0.99, 0.96, 0.93, 0.90, 0.87, 0.85]
    yelp_lightgcn = [0.98, 0.94, 0.90, 0.86, 0.83, 0.80]
    
    # LastFM数据集结果
    lastfm_gformer = [1.00, 0.97, 0.95, 0.93, 0.91, 0.89]
    lastfm_sgl = [0.98, 0.95, 0.92, 0.89, 0.86, 0.84]
    lastfm_lightgcn = [0.97, 0.93, 0.89, 0.85, 0.82, 0.79]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Yelp数据集的Recall@20
    ax1.plot(noise_ratio, yelp_gformer, 'o-', label='GFormer', color='royalblue')
    ax1.plot(noise_ratio, yelp_sgl, 's-', label='SGL', color='orange')
    ax1.plot(noise_ratio, yelp_lightgcn, '^-', label='LightGCN', color='green')
    ax1.set_title('(a) Yelp Dataset')
    ax1.set_xlabel('Noise Ratio')
    ax1.set_ylabel('Normalized Recall@20')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Yelp数据集的NDCG@20
    ax2.plot(noise_ratio, yelp_gformer, 'o-', label='GFormer', color='royalblue')
    ax2.plot(noise_ratio, yelp_sgl, 's-', label='SGL', color='orange')
    ax2.plot(noise_ratio, yelp_lightgcn, '^-', label='LightGCN', color='green')
    ax2.set_xlabel('Noise Ratio')
    ax2.set_ylabel('Normalized NDCG@20')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # LastFM数据集的Recall@20
    ax3.plot(noise_ratio, lastfm_gformer, 'o-', label='GFormer', color='royalblue')
    ax3.plot(noise_ratio, lastfm_sgl, 's-', label='SGL', color='orange')
    ax3.plot(noise_ratio, lastfm_lightgcn, '^-', label='LightGCN', color='green')
    ax3.set_title('(b) LastFM Dataset')
    ax3.set_xlabel('Noise Ratio')
    ax3.set_ylabel('Normalized Recall@20')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # LastFM数据集的NDCG@20
    ax4.plot(noise_ratio, lastfm_gformer, 'o-', label='GFormer', color='royalblue')
    ax4.plot(noise_ratio, lastfm_sgl, 's-', label='SGL', color='orange')
    ax4.plot(noise_ratio, lastfm_lightgcn, '^-', label='LightGCN', color='green')
    ax4.set_xlabel('Noise Ratio')
    ax4.set_ylabel('Normalized NDCG@20')
    ax4.legend()
    ax4.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('figures/noise_robustness.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_training_convergence():
    """创建训练收敛性分析图（类似论文Figure 6）"""
    epochs = np.arange(0, 100, 1)
    
    # 模拟训练数据
    gformer_recall = 0.09 - 0.05 * np.exp(-epochs/20)
    gformer_ndcg = 0.045 - 0.025 * np.exp(-epochs/20)
    
    hccf_recall = 0.08 - 0.04 * np.exp(-epochs/25)
    hccf_ndcg = 0.04 - 0.02 * np.exp(-epochs/25)
    
    lightgcn_recall = 0.07 - 0.03 * np.exp(-epochs/30)
    lightgcn_ndcg = 0.035 - 0.015 * np.exp(-epochs/30)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Recall@20
    ax1.plot(epochs, gformer_recall, '-', label='GFormer', color='royalblue')
    ax1.plot(epochs, hccf_recall, '--', label='HCCF', color='orange')
    ax1.plot(epochs, lightgcn_recall, ':', label='LightGCN', color='green')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Recall@20')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 标记收敛点
    ax1.plot(30, gformer_recall[30], '*', color='red', markersize=10)
    ax1.plot(40, hccf_recall[40], '*', color='red', markersize=10)
    ax1.plot(50, lightgcn_recall[50], '*', color='red', markersize=10)
    
    # NDCG@20
    ax2.plot(epochs, gformer_ndcg, '-', label='GFormer', color='royalblue')
    ax2.plot(epochs, hccf_ndcg, '--', label='HCCF', color='orange')
    ax2.plot(epochs, lightgcn_ndcg, ':', label='LightGCN', color='green')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('NDCG@20')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 标记收敛点
    ax2.plot(30, gformer_ndcg[30], '*', color='red', markersize=10)
    ax2.plot(40, hccf_ndcg[40], '*', color='red', markersize=10)
    ax2.plot(50, lightgcn_ndcg[50], '*', color='red', markersize=10)
    
    plt.suptitle('Test results in terms of Recall@20 and NDCG@20\nw.r.t training epochs on Yelp dataset')
    plt.tight_layout()
    plt.savefig('figures/training_convergence.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_embedding_dimension():
    """创建嵌入维度实验图（类似论文Figure 7）"""
    dimensions = np.array([50, 100, 150, 200, 250, 300, 350, 400, 450, 500])
    
    # 模拟数据
    recall = 0.24 - 0.1 * np.exp(-dimensions/100)
    ndcg = 0.22 - 0.08 * np.exp(-dimensions/100)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Recall@20
    ax1.plot(dimensions, recall, 'o-', color='royalblue')
    ax1.set_xlabel('Dimension of Embeddings')
    ax1.set_ylabel('Recall@20')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # NDCG@20
    ax2.plot(dimensions, ndcg, 'o-', color='royalblue')
    ax2.set_xlabel('Dimension of Embeddings')
    ax2.set_ylabel('NDCG@20')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.suptitle('Performance in terms of Recall@20 and NDCG@20\nunder different embedding dimensionality')
    plt.tight_layout()
    plt.savefig('figures/embedding_dimension.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # 创建figures目录
    import os
    os.makedirs('figures', exist_ok=True)
    
    # 生成所有可视化
    create_ablation_study()
    create_noise_robustness()
    create_training_convergence()
    create_embedding_dimension()
    
    print("所有论文风格的可视化图表已生成在'figures'目录中。") 