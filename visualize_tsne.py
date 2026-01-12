import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, f1_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import label_binarize
from itertools import cycle
import matplotlib
import math
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.special import softmax
import scipy.io as sio
from joblib import load

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
matplotlib.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# 1. 损失曲线
def plot_loss_curve(train_loss, output_dir,val_loss=None):
    plt.figure()
    plt.plot(train_loss, label='Train Loss', color='blue')
    if val_loss:
        plt.plot(val_loss, label='Validation Loss', color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{output_dir}损失曲线.png'), dpi=600)
    # plt.show()
    plt.close()

# 2. 准确率曲线
def plot_accuracy_curve(train_acc,output_dir,val_acc=None):
    plt.figure()
    plt.plot(train_acc, label='Train Accuracy', color='green')
    if val_acc:
        plt.plot(val_acc, label='Validation Accuracy', color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{output_dir}_准确率曲线.png'), dpi=600)
    # plt.show(
    plt.close()

# 3. 混淆矩阵
def plot_confusion_matrix_10class(y_true, y_pred, class_names, output_dir,normalize=False):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{output_dir}_混淆矩阵.png'), dpi=600)
    # plt.show()
    plt.close()

# 4. ROC 曲线
def plot_roc_curve(y_true, y_score, class_names,output_dir):
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    plt.figure(figsize=(8, 6))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple', 'brown', 'pink', 'gray', 'olive'])
    for i, color in zip(range(n_classes), colors):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{output_dir}_ROC 曲线.png'), dpi=600)
    # plt.show()
    plt.close()

# 5. PR 曲线
def plot_precision_recall_curve(y_true, y_score, class_names,output_dir):
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    plt.figure(figsize=(8, 6))
    for i in range(len(class_names)):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
        plt.plot(recall, precision, label=f'{class_names[i]}')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{output_dir}_PR 曲线.png'), dpi=600)
    # plt.show()
    plt.close()

# 6. F1-score 条形图
def plot_f1_scores(y_true, y_pred, class_names,output_dir):
    scores = f1_score(y_true, y_pred, average=None)
    plt.figure(figsize=(8, 4))
    plt.bar(class_names, scores, color='purple')
    plt.ylabel("F1 Score")
    plt.title("F1 Score per Class")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{output_dir}_F1-score 条形图.png'), dpi=600)
    # plt.show()
    plt.close()

# 7. t-SNE 可视化
def plot_tsne(features, labels, class_names,output_dir):
    tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42)
    reduced = tsne.fit_transform(features)
    plt.figure(figsize=(8, 6))
    for i, name in enumerate(class_names):
        idx = labels == i
        plt.scatter(reduced[idx, 0], reduced[idx, 1], label=name, s=10)
    plt.title("t-SNE Feature Visualization")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{output_dir}_t-SNE 可视化.png'), dpi=600)
    # plt.show()
    plt.close()

# 8. PCA 可视化
def plot_pca(features, labels, class_names,output_dir):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(features)
    plt.figure(figsize=(8, 6))
    for i, name in enumerate(class_names):
        idx = labels == i
        plt.scatter(reduced[idx, 0], reduced[idx, 1], label=name, s=10)
    plt.title("PCA Feature Visualization")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{output_dir}_PCA 可视化.png'), dpi=600)
    # plt.show()
    plt.close()

# 9. 类别样本数量柱状图
def plot_class_distribution(labels, class_names,output_dir):
    counts = [np.sum(labels == i) for i in range(len(class_names))]
    plt.figure(figsize=(8, 4))
    plt.bar(class_names, counts, color='skyblue')
    plt.ylabel("Sample Count")
    plt.title("Class Distribution")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{output_dir}_类别样本数量柱状图.png'), dpi=600)
    # plt.show()
    plt.close()

# 10. 学习率变化曲线
def plot_learning_rate_schedule(lrs,output_dir):
    plt.figure()
    plt.plot(lrs, label='Learning Rate', color='magenta')
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{output_dir}_学习率变化曲线.png'), dpi=600)
    # plt.show()
    plt.close()

# 11. 梯度变化图
def plot_gradient_flow(named_parameters, output_dir):
    """
    展示模型各层梯度的最大值与平均值，用于检测梯度消失或爆炸问题。
    参数: named_parameters: model.named_parameters() 得到的迭代器
    """
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if p.requires_grad and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().item() if p.grad is not None else 0)
            max_grads.append(p.grad.abs().max().item() if p.grad is not None else 0)
    plt.figure(figsize=(12, 6))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, color="c", label="Max Gradients")
    plt.bar(np.arange(len(ave_grads)), ave_grads, alpha=0.5, color="b", label="Mean Gradients")
    plt.hlines(0, 0, len(ave_grads), lw=2, color="k")
    plt.xticks(range(len(ave_grads)), layers, rotation="vertical")
    plt.xlabel("Layers")
    plt.ylabel("Gradient")
    plt.title("Gradient Flow")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{output_dir}_梯度变化图.png'), dpi=600)
    plt.close()

# 12. 特征图可视化
def plot_feature_map(feature_map, output_dir):
    """
    可视化单个特征图（例如卷积层的输出）。
    参数:feature_map: numpy 数组，形状 [C, H, W] 或 [H, W]
    """
    plt.figure()
    if feature_map.ndim == 3:
        num_channels = feature_map.shape[0]
        num_plots = min(num_channels, 8)
        for i in range(num_plots):
            plt.subplot(1, num_plots, i+1)
            plt.imshow(feature_map[i], cmap='viridis')
            plt.axis('off')
        plt.suptitle("Feature Map Visualization")
    elif feature_map.ndim == 2:
        plt.imshow(feature_map, cmap='viridis')
        plt.title("Feature Map Visualization")
        plt.axis('off')
    else:
        print("feature_map 的维度不符合要求")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{output_dir}_特征图可视化.png'), dpi=600)
    plt.close()

# 13. t-SNE 降维可视化（投影图）
def plot_tsne_projection(features, labels, class_names, output_dir):
    """利用 t-SNE 降维展示高维特征在二维空间的分布。"""
    tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42)
    reduced = tsne.fit_transform(features)
    plt.figure(figsize=(8, 6))
    for i, name in enumerate(class_names):
        idx = labels == i
        plt.scatter(reduced[idx, 0], reduced[idx, 1], label=name, s=10)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title("t-SNE Projection")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{output_dir}_t-SNE投影图projection.png'), dpi=600)
    plt.close()

# 14. 热力图绘制
def plot_heatmap_data(y_true, y_pred, class_names, output_dir):
    """
    根据输入的二维数值矩阵绘制热力图。
    参数:
        data_matrix: 2D numpy 数组
    """
    # 生成热力图数据（10 x 10）生成混淆矩阵（返回的是二维 numpy 数组）
    heatmap_data = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, cmap="YlGnBu", annot=True, fmt=".2f")
    plt.title("Heatmap")
    plt.xlabel("X 轴")
    plt.ylabel("Y 轴")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{output_dir}_热力图.png'), dpi=600)
    plt.close()

# 15. 模型结构摘要（利用 torchsummary）
def plot_model_summary(model, input_size, output_dir):
    """
    利用 torchsummary 展示模型结构及参数量。
    参数:
        model: 待评估模型
        input_size: 输入尺寸，例如 (1, 224, 224)
    """
    try:
        from torchsummary import summary
        summary_str = summary(model, input_size=input_size, verbose=0)
        # 将输出保存到文本文件中
        summary_path = os.path.join(output_dir, f'{output_dir}_模型结构摘要.txt')
        with open(summary_path, 'w') as f:
            f.write(str(summary_str))
        print(f"模型结构摘要已保存至 {summary_path}")
    except ImportError:
        print("请安装 torchsummary: pip install torchsummary")

# 16. 模型结构可视化图（利用 torchviz）
def plot_model_architecture(model, input_tensor, output_dir):
    """
    利用 torchviz 绘制模型计算图，并保存为图片。
    参数:
        model: 待评估模型
        input_tensor: 示例输入（torch.Tensor）
    """
    try:
        from torchviz import make_dot
        model.eval()
        output = model(input_tensor)
        dot = make_dot(output, params=dict(model.named_parameters()))
        architecture_path = os.path.join(output_dir, f'{output_dir}_模型结构图')
        dot.format = 'png'
        dot.render(architecture_path)
        print(f"模型结构图已保存为 {architecture_path}.png")
    except ImportError:
        print("请安装 torchviz: pip install torchviz")

# 17. 训练时间对比图
def plot_training_time_comparison(models, training_times, output_dir):
    """
    绘制不同模型训练时间的柱状图。
    参数:
        models: 模型名称列表
        training_times: 对应的训练时间列表（秒）
    """
    plt.figure(figsize=(8, 6))
    plt.bar(models, training_times, color='teal')
    plt.xlabel("Model")
    plt.ylabel("Training Time (s)")
    plt.title("Training Time Comparison")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{output_dir}_训练时间对比图.png'), dpi=600)
    plt.close()

# 18. F1/Precision/Recall 曲线（随 epoch 变化）
def plot_f1_precision_recall_curve(epoch_list, y_true_list, y_pred_list, output_dir):
    """
    绘制 F1、Precision 和 Recall 随 epoch 变化的曲线。
    参数:
        epoch_list: 每个 epoch 的编号列表
        y_true_list: 每个 epoch 的真实标签数组或列表的列表
        y_pred_list: 每个 epoch 的预测标签数组或列表的列表
        output_dir: 输出目录
    """
    # 如果传入的 y_true_list 和 y_pred_list 是单个 ndarray 或单个列表，则包装为列表
    if isinstance(y_true_list, np.ndarray) and isinstance(y_pred_list, np.ndarray):
        y_true_list = [y_true_list]
        y_pred_list = [y_pred_list]
    elif isinstance(y_true_list, list) and len(y_true_list) > 0 and not isinstance(y_true_list[0], (list, np.ndarray)):
        y_true_list = [y_true_list]
        y_pred_list = [y_pred_list]

    f1_scores = []
    precisions = []
    recalls = []
    for yt, yp in zip(y_true_list, y_pred_list):
        yt = np.asarray(yt)
        yp = np.asarray(yp)
        f1_scores.append(f1_score(yt, yp, average='macro'))
        precisions.append(precision_score(yt, yp, average='macro'))
        recalls.append(recall_score(yt, yp, average='macro'))

    # 如果 epoch_list 长度与指标数量不符，则重构 x 轴
    if not hasattr(epoch_list, '__len__') or len(epoch_list) != len(f1_scores):
        epoch_axis = list(range(1, len(f1_scores) + 1))
    else:
        epoch_axis = epoch_list

    plt.figure(figsize=(8, 6))
    plt.plot(epoch_axis, f1_scores, marker='o', label="F1 Score")
    plt.plot(epoch_axis, precisions, marker='s', label="Precision")
    plt.plot(epoch_axis, recalls, marker='^', label="Recall")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("F1/Precision/Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{output_dir}_F1_Precision_Recall曲线.png'), dpi=600)
    plt.close()

# 19. FLOPs vs Accuracy 图
def plot_flops_vs_accuracy(flops, accuracies, models, output_dir):
    """
    绘制 FLOPs（或参数量）与准确率之间关系的散点图。
    参数:
        flops: FLOPs 列表
        accuracies: 准确率列表
        models: 模型名称列表（可选，显示在散点旁边）
    """
    plt.figure(figsize=(8,6))
    plt.scatter(flops, accuracies, color='dodgerblue')
    for i, model in enumerate(models):
        plt.text(flops[i], accuracies[i], str(model), fontsize=9)
    plt.xlabel("FLOPs/Parameters")
    plt.ylabel("Accuracy")
    plt.title("FLOPs vs Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{output_dir}_FLOPs_vs_Accuracy.png'), dpi=600)
    plt.close()

# 20. 损失分布直方图
def plot_loss_distribution(val_loss, output_dir, bins=10):
    """
    绘制所有样本损失分布的直方图，并标注均值和中位数。
    参数:
        val_loss   : list 或 numpy 数组，所有样本的损失值
        output_dir : 字符串，图像保存目录
        bins       : 整数，直方图的 bin 数量，默认 10
    """
    # 转成 numpy 数组，方便计算
    losses = np.array(val_loss)

    # 计算统计量
    mean_val   = np.mean(losses)
    median_val = np.median(losses)

    # 创建图形
    plt.figure(figsize=(8, 5))
    plt.hist(losses, bins=bins, color='coral', alpha=0.75, edgecolor='black')

    # 标注均值、中位数
    plt.axvline(mean_val,   color='blue', linestyle='--', linewidth=1.5,
                label=f'均值 = {mean_val:.3f}')
    plt.axvline(median_val, color='green', linestyle='-.', linewidth=1.5,
                label=f'中位数 = {median_val:.3f}')

    # 坐标和标题
    plt.xlabel("val_Loss Value")
    plt.ylabel("Frequency")
    plt.title("Loss Distribution with Mean & Median")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    # 确保目录存在
    os.makedirs(output_dir, exist_ok=True)
    fname = os.path.join(output_dir,
                         f'{os.path.basename(output_dir)}_损失分布直方图.png')
    plt.savefig(fname, dpi=600)
    plt.close()

# 21. 平滑训练损失曲线
def plot_smoothed_loss(train_loss, output_dir, alpha):
    """
    绘制经过指数移动平均平滑处理的训练损失曲线。
    参数:
        train_loss: 每个 epoch 的训练损失列表
        alpha: 平滑因子（0<alpha<=1）
    """
    df = pd.DataFrame({'train_loss': train_loss})
    smoothed = df['train_loss'].ewm(alpha=alpha).mean()
    plt.figure()
    plt.plot(train_loss, label="Original Loss", alpha=0.3)
    plt.plot(smoothed, label=f"Smoothed Loss (α={alpha})", color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Smoothed Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{output_dir}_平滑训练损失曲线.png'), dpi=600)
    plt.close()

# 22. 预测概率分布图
def plot_prediction_probability_distribution(y_true, y_pred, y_score, class_names, output_dir):
    """
    对比正确和错误预测时的概率分布（密度图）。
    参数:
        y_true: 真实标签数组
        y_pred: 预测标签数组
        y_prob: 预测概率数组（通常为某一类别的概率或置信度）
    """
    y_score = softmax(y_score, axis=1)
    if len(class_names) == 2:
        y_prob = y_score[:, 1]
    else:
        idx = np.arange(len(y_pred))
        y_prob = y_score[idx, y_pred]
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "y_prob": y_prob})
    df["correct"] = df["y_true"] == df["y_pred"]
    plt.figure(figsize=(8,5))
    sns.kdeplot(data=df, x="y_prob", hue="correct", fill=True)
    plt.xlabel("Predicted Probability")
    plt.title("Prediction Probability Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{output_dir}_预测概率分布图.png'), dpi=600)
    plt.close()

# 23. 损失与置信度关系图
def plot_loss_vs_confidence(loss_values, y_score, y_pred, class_names, output_dir):
    """
    展示样本损失与预测置信度之间的关系。
    参数:
        loss_values: 每个样本的损失值
        y_prob: 每个样本的预测概率（置信度），通常差值 0.5*2 作为置信度指标
    """
    y_score = softmax(y_score, axis=1)
    if len(class_names) == 2:
        y_prob = y_score[:, 1]
    else:
        idx = np.arange(len(y_pred))
        y_prob = y_score[idx, y_pred]
    loss_values = np.asarray(loss_values)
    y_prob = np.asarray(y_prob)
    if loss_values.shape[0] != y_prob.shape[0]:
        min_size = min(loss_values.shape[0], y_prob.shape[0])
        loss_values = loss_values[:min_size]
        y_prob = y_prob[:min_size]
    confidence = np.abs(y_prob - 0.5) * 2
    plt.figure(figsize=(8,6))
    sc = plt.scatter(confidence, loss_values, c=loss_values, cmap='viridis', alpha=0.6)
    plt.xlabel("Confidence")
    plt.ylabel("Loss")
    plt.title("Loss vs Confidence")
    plt.colorbar(sc)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{output_dir}_损失与置信度关系图.png'), dpi=600)
    plt.close()

# 24. 动态特征演化图
def plot_dynamic_feature_evolution(epochs, feature_over_time, output_dir):
    """
    展示训练过程中，不同 epoch 下样本在特征空间中的分布变化。
    参数:
        epoch_list: 每个数据点对应的 epoch 标签（数组）
        features_over_time: 每个样本的二维特征（已经降维），形状 (N, 2)
    """
    N = feature_over_time.shape[0] // len(epochs)  # 每个 epoch 的样本数
    epoch_list = []
    for epoch in epochs:
        epoch_list.extend([epoch] * N)
    epoch_list = np.array(epoch_list)
    plt.figure(figsize=(10,8))
    unique_epochs = np.unique(epoch_list)
    for epoch in unique_epochs:
        idx = np.array(epoch_list) == epoch
        plt.scatter(feature_over_time[idx, 0], feature_over_time[idx, 1], label=f"Epoch {epoch}", s=10, alpha=0.6)
    plt.xlabel("Feature Dimension 1")
    plt.ylabel("Feature Dimension 2")
    plt.title("Dynamic Feature Evolution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{output_dir}_动态特征演化图.png'), dpi=600)
    plt.close()

# 25. 错分样本可视化
def plot_misclassified_samples(y_true, y_pred, class_names, output_dir, samples_data, num_samples=25):
    """
    可视化模型错分的样本。
    参数:
        y_true: ndarray[int]，真实标签数组（长度为 N）
        y_pred: ndarray[int]，预测标签数组（长度为 N）
        samples: numpy 数组或列表，样本数据（例如图像或波形）；要求其索引与 y_true,y_pred 对应
        class_names: list[str]，类别名称列表
        output_dir: 字符串，保存图表的输出文件夹
        num_samples: int，最多展示的错分样本个数（默认25个）
    图表表达的内容：
        展示模型错分样本的分布情况，并标注每个样本的真实和预测类别。
    """
    samples = samples_data.iloc[:, :-1].values
    mis_idx = [i for i in range(len(y_true)) if y_true[i] != y_pred[i] and i < len(samples)]
    # mis_idx = [i for i in range(len(y_true)) if y_true[i] != y_pred[i]]
    if len(mis_idx) == 0:
        print("没有错分样本！")
        return
    num_show = min(num_samples, len(mis_idx))
    cols = int(math.sqrt(num_show))
    rows = math.ceil(num_show / cols)
    plt.figure(figsize=(cols * 3, rows * 3))
    for idx, sample_idx in enumerate(mis_idx[:num_show]):
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(samples[sample_idx], cmap='gray' if samples[sample_idx].ndim == 2 else None)
        plt.title(f"真: {class_names[y_true[sample_idx]]}\n预测: {class_names[y_pred[sample_idx]]}", fontsize=8)
        plt.axis("off")
    plt.suptitle("错分样本可视化", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, f'{output_dir}_错分样本可视化.png'), dpi=600)
    plt.close()

# 26. 置信度柱状图
def plot_top_n_confidence(y_scores, y_pred, class_names, output_dir, top_n=5):
    """
    展示预测置信度最高的 Top-N 类别的平均置信度。
    参数:
        y_scores: ndarray[float]，模型预测的概率数组，形状 [N, num_classes]
        y_pred: ndarray[int]，预测标签数组（长度为 N）
        class_names: list[str]，类别名称列表
        output_dir: 字符串，保存图表的文件夹
        top_n: int，选取排名前 N 的类别（默认5个）
    图表表达的内容：
        通过柱状图展示各类别中，模型对预测类别的平均置信度。
    """
    num_classes = len(class_names)
    avg_confidence = []
    for c in range(num_classes):
        idx = (y_pred == c)
        if np.sum(idx) > 0:
            conf = np.max(y_scores[idx], axis=1)
            avg_confidence.append(np.mean(conf))
        else:
            avg_confidence.append(0)
    avg_confidence = np.array(avg_confidence)
    top_idx = np.argsort(avg_confidence)[-top_n:][::-1]
    plt.figure(figsize=(8, 4))
    plt.bar([class_names[i] for i in top_idx], avg_confidence[top_idx], color='orange')
    plt.xlabel("类别")
    plt.ylabel("平均预测置信度")
    plt.title(f"Top-{top_n} 预测置信度柱状图")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{output_dir}_置信度柱状图.png'), dpi=600)
    plt.close()

# 27. Top-k Accuracy 曲线
def plot_topk_accuracy(topk_dict, output_dir):
    """
    绘制 Top-k Accuracy 随 epoch 变化的曲线，其中 k 可为1到5（或其他）。
    参数:
        topk_dict: dict, 键为 k 值（例如 1,2,...），值为每个 epoch 对应的准确率列表
        output_dir: 字符串，保存图表的输出文件夹
    图表表达的内容：
        展示不同 k 值对应的 Top-k Accuracy 随训练过程的变化情况。
    """
    plt.figure(figsize=(8, 6))
    epochs = list(range(1, len(next(iter(topk_dict.values()))) + 1))
    for k, acc_list in topk_dict.items():
        plt.plot(epochs, acc_list, marker='o', label=f"Top-{k} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Top-k Accuracy 曲线")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{output_dir}_Top-k_Accuracy曲线.png'), dpi=600)
    plt.close()

# 28. 类别间相似性热力图
def plot_class_similarity(y_true, y_pred, class_names, output_dir):
    """
    基于混淆矩阵构造类别间的相似性热力图，展示不同类别之间的互相混淆情况。
    参数:
        y_true: ndarray[int]，真实标签数组
        y_pred: ndarray[int]，预测标签数组
        class_names: list[str]，类别名称列表
        output_dir: 字符串，保存图表的输出文件夹
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_sym = (cm_norm + cm_norm.T) / 2.0
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_sym, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("类别")
    plt.ylabel("类别")
    plt.title("类别间相似性热力图")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{output_dir}_类别间相似性热力图.png'), dpi=600)
    plt.close()

# 29. 特征通道统计
def plot_feature_channel_statistics(feature_map, output_dir):
    """
    对单个特征图的各个通道进行统计分析，并绘制每个通道的平均激活值柱状图。
    参数:
        feature_map: numpy 数组，形状 [C, H, W]
        output_dir: 字符串，保存图表的输出文件夹
    """
    if feature_map.ndim != 3:
        print("feature_map 的维度不符合要求，需为 [C, H, W]")
        return
    num_channels = feature_map.shape[0]
    channel_means = [np.mean(feature_map[i]) for i in range(num_channels)]
    plt.figure(figsize=(10, 4))
    plt.bar(range(num_channels), channel_means, color='teal')
    plt.xlabel("Channel")
    plt.ylabel("Average Activation")
    plt.title("特征图通道激活统计")
    plt.xticks(range(num_channels))
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{output_dir}_特征通道统计.png'), dpi=600)
    plt.close()

# 30. 错误样本分析报告
def save_misclassified_report(y_true, y_pred, class_names, output_dir, samples_data, num_samples=25):
    """
    保存错误样本的分析报告，同时生成错分样本的可视化图。
    参数:
        y_true: ndarray[int]，真实标签数组
        y_pred: ndarray[int]，预测标签数组
        samples: 样本数据（图像或波形），要求与 y_true,y_pred 对应
        class_names: list[str]，类别名称列表
        output_dir: 字符串，保存报告及图表的文件夹
        num_samples: int，展示的错分样本个数（默认 25 个）
    """
    samples = samples_data.iloc[:, :-1].values
    mis_indices = [i for i in range(len(y_true)) if y_true[i] != y_pred[i]]
    report_lines = ["Index,True Label,Predicted Label"]
    for i in mis_indices:
        report_lines.append(f"{i},{class_names[y_true[i]]},{class_names[y_pred[i]]}")
    report_text = "\n".join(report_lines)
    report_path = os.path.join(output_dir, f'{output_dir}_错误样本报告.txt')
    with open(report_path, "w") as f:
        f.write(report_text)
    # print(f"错误样本报告已保存至 {report_path}")
    # plot_misclassified_samples(y_true, y_pred, samples, class_names, output_dir, num_samples=num_samples)

# 31. 加载多个.mat文件中的样本数据
def load_samples_from_mat(file_names, data_dir):
    """
    从多个 .mat 文件中提取样本数据。
    返回：samples: ndarray，所有样本拼接后的数据（N, T）或 (N, T, C)
    """
    all_samples = []
    for fname in file_names:
        file_path = f"{data_dir}/{fname}"
        mat_data = sio.loadmat(file_path)

        # 假设数据存在键 'X'（形状如 (num_samples, 16) 或 (num_samples, 16, 2)）
        # 你可以用 print(mat_data.keys()) 查看实际键名
        data = mat_data['X']  # 根据实际情况替换为正确的键名

        all_samples.append(data)

    return np.vstack(all_samples)  # 合并成一个 (N, ...) 的 ndarray

# 32. 损失准确率曲线（统一横纵轴 0~1，刻度间距一致）
def loss_acc_curve(train_loss, train_acc, validate_loss, validate_acc, output_dir, filename):
    """
    train_loss:     list or array, 训练损失
    train_acc:      list or array, 训练准确率（0~1）
    validate_loss:  list or array, 验证损失
    validate_acc:   list or array, 验证准确率（0~1）
    output_dir:     str, 保存图像的目录，文件名前缀也用此参数
    """
    epochs = len(train_loss)
    # 横坐标：1,2,...,epochs
    x_epochs = np.arange(0, epochs)

    plt.figure()
    plt.plot(x_epochs, train_loss,      label='Train Loss', color='blue')
    plt.plot(x_epochs, train_acc,       label='Train Accuracy', color='green')
    plt.plot(x_epochs, validate_loss,   label='Validation Loss', color='orange')
    plt.plot(x_epochs, validate_acc,    label='Validation Accuracy', color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(f"{filename} 的损失与准确率曲线")

    plt.xlim(0, epochs-1)  # 横轴范围 1 到 epochs
    if epochs <= 10:  # 自动选取刻度间隔：最多显示 10 个刻度
        xticks = x_epochs
    else:
        step = max(1, epochs // 10)
        xticks = np.arange(0, epochs + 1, step)
    plt.xticks(xticks)
    plt.yticks(np.linspace(0, 1, 11))

    plt.legend(loc='best', framealpha=0.5)
    plt.grid(True)
    plt.tight_layout()

    # 保存
    if output_dir is not None:
        filename = f"{output_dir}_损失准确率曲线.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=600)
    plt.close()


def visualize_all_results(
    file_name,                   # str：文件名
    train_loss,                  # List[float]：训练损失，每个 epoch 一个数
    val_loss,                    # List[float]：验证损失
    train_acc,                   # List[float]：训练准确率
    val_acc,                     # List[float]：验证准确率
    y_true,                      # ndarray[int]：真实标签（长度为 N）
    y_pred,                      # ndarray[int]：预测标签（长度为 N）
    y_scores,                    # ndarray[float]：每个样本的预测概率（shape: [N, num_classes]）
    features_model,              # ndarray[float]：分类后每个样本的特征向量（shape: [N, dim]）
    features,                    # ndarray[float]：每个样本的特征向量（shape: [N, dim]）
    class_names,                 # List[str]：类别名称（长度为 num_classes）
    learning_rates,              # List[float]：每个 epoch 的学习率
    epochs,                      # int：训练轮数
    bins,                        # int：loss 值分布直方图的 bin 数 bin 数量越多，直方图越“细腻”但更分散；bin 数量越少，直方图越“粗糙”但更集中
    exp_name                     # str：实验名称

):
    """
    可视化分类模型的全套训练结果。
    包含损失曲线、准确率曲线、混淆矩阵、ROC/PR曲线、F1-score、t-SNE、PCA、分布柱状图、学习率曲线等。

    示例调用：
        visualize_all_results(train_loss, val_loss, train_acc, val_acc,
                              y_true, y_pred, y_scores, features,
                              class_names, learning_rates)
    """

    #创建文件夹
    output_dir = f'{file_name}结果可视化示意图'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    samples_data = load(f'{exp_name}_samples_data.joblib')
    # print(samples_data.shape)
    feature_map = np.random.rand(8, 28, 28)  # 模拟特征图（8 x 28 x 28）注意：梯度变化图需要真实梯度数据，此处省略传参

    # 导入图表函数（假设你已经 from result_visualization import *）
    plot_loss_curve(train_loss,val_loss= val_loss,output_dir = output_dir)
    plot_accuracy_curve(train_acc,val_acc = val_acc,output_dir=output_dir)
    plot_confusion_matrix_10class(y_true, y_pred, class_names,output_dir=output_dir)
    plot_roc_curve(y_true, y_scores, class_names,output_dir)
    plot_precision_recall_curve(y_true, y_scores, class_names,output_dir)
    plot_f1_scores(y_true, y_pred, class_names,output_dir)
    plot_tsne(features_model, y_true, class_names,output_dir)
    plot_pca(features_model, y_true, class_names,output_dir)
    plot_class_distribution(y_true, class_names,output_dir)
    plot_learning_rate_schedule(learning_rates,output_dir)
    plot_feature_map(feature_map, output_dir)
    plot_tsne_projection(features, y_true, class_names, output_dir)
    plot_heatmap_data(y_true, y_pred, class_names, output_dir)
    plot_f1_precision_recall_curve(epochs, y_true, y_pred, output_dir)
    plot_loss_distribution(val_loss, output_dir, bins)
    plot_smoothed_loss(train_loss, output_dir, alpha=0.8)
    plot_prediction_probability_distribution(y_true, y_pred, y_scores, class_names, output_dir)
    plot_loss_vs_confidence(val_loss, y_scores, y_pred, class_names, output_dir)
    plot_top_n_confidence(y_scores, y_pred, class_names, output_dir, top_n=5)
    plot_class_similarity(y_true, y_pred, class_names, output_dir)
    plot_feature_channel_statistics(feature_map, output_dir)
    save_misclassified_report(y_true, y_pred, class_names, output_dir, samples_data, num_samples=25)
    loss_acc_curve(train_loss, train_acc, val_loss, val_acc, output_dir, file_name)
    # plot_misclassified_samples(y_true, y_pred, class_names, output_dir, samples_data, num_samples=25)
    # epoch_list = np.random.choice(epochs, size=100)
    # features_over_time = np.random.randn(100, 2)
    # plot_dynamic_feature_evolution(epochs, features, output_dir)  # 此处需要修改训练过程

    # 模型性能指标比较
    models = ["Model A", "Model B", "Model C"]  # 每个模型的名称字符串
    training_times = [120, 150, 100]  # 每个模型的训练时间（单位：秒）
    plot_training_time_comparison(models, training_times, output_dir)
    flops = [1e9, 1.5e9, 0.8e9]  # 每个模型的 FLOPs 或参数量（建议统一单位，如 GFLOPs）
    accuracies = [0.80, 0.82, 0.78]  # 每个模型 val_acc[-1] 或测试集 acc
    plot_flops_vs_accuracy(flops, accuracies, models, output_dir)

    # 模型结构摘要与可视化（需实际模型）
    # from your_model_file import model
    # plot_model_summary(model, input_size=(3, 224, 224), output_dir=output_dir)
    # input_tensor = torch.randn(1, 3, 224, 224)
    # plot_model_architecture(model, input_tensor, output_dir)


if __name__ == '__main__':
    # 模拟：训练过程记录
    train_loss = [1.2, 0.9, 0.7, 0.5, 0.4]
    val_loss = [1.1, 0.95, 0.8, 0.6, 0.5]
    train_acc = [0.55, 0.65, 0.75, 0.82, 0.88]
    val_acc = [0.52, 0.62, 0.7, 0.77, 0.83]
    learning_rates = [0.01, 0.005, 0.001, 0.0005, 0.0001]

    # 模拟：真实和预测标签（100样本，10类）
    y_true = np.random.randint(0, 10, 100)
    y_pred = y_true.copy()
    y_pred[np.random.choice(100, 20, replace=False)] = np.random.randint(0, 10, 20)  # 模拟一些错误预测

    # 模拟：模型输出的概率 (100 样本 × 10 类)
    y_scores = np.random.rand(100, 10)
    y_scores /= y_scores.sum(axis=1, keepdims=True)

    # 模拟：每个样本的特征向量（用于 tsne/pca）
    features = np.random.randn(100, 64)

    # 类别名
    class_names = [f"Class {i}" for i in range(10)]

    exp_name = "model_name"
    visualize_all_results(
        file_name='分类模型训练结果',
        train_loss=train_loss,
        val_loss=val_loss,
        train_acc=train_acc,
        val_acc=val_acc,
        y_true=y_true,
        y_pred=y_pred,
        y_scores=y_scores,
        features=features,
        class_names=class_names,
        learning_rates=learning_rates,
        epochs=5,
        bins=10,
        exp_name=exp_name
    )
