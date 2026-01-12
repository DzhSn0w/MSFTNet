import torch
from joblib import dump, load
import torch.utils.data as Data
import numpy as np
import torch
import torch.nn as nn
from MSFTNet import MSFTNet
import time
import torch.nn.functional as F
import numpy as np
from visualize_tsne import visualize_all_results
import pandas as pd

#实验次数
num_exp = 5
# 参数与配置
snr_db = 0 # 信噪比，单位 dB
torch.manual_seed(1000)  # 设置随机种子，以使实验结果具有可重复性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 有GPU先用GPU训练
#目前工况
exp_name = "MSFTNet_add_noisy_0db_1x2048_990"


for zhe in range(num_exp):
    zhe = zhe + 1
    print(f'第{zhe}次实验.')
    # 加载数据集
    #切分数据集
    import pandas as pd
    import scipy.io
    import os

    def add_noise_by_snr(signal, snr_db):
        """
        按指定 SNR（单位：dB）向信号添加高斯白噪声
        参数：
            signal: numpy 数组或 torch 张量
            snr_db: 信噪比，单位 dB
        返回：
            添加了噪声的 signal（类型与输入一致）
        """
        if isinstance(signal, torch.Tensor):
            signal_np = signal.numpy()
        else:
            signal_np = signal

        # 计算信号功率
        signal_power = np.mean(signal_np ** 2)
        # 计算噪声功率
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        # 生成高斯白噪声
        noise = np.random.normal(0, np.sqrt(noise_power), signal_np.shape)
        # 添加噪声
        noisy_signal = signal_np + noise

        # 返回与输入相同类型
        if isinstance(signal, torch.Tensor):
            return torch.from_numpy(noisy_signal).float()
        else:
            return noisy_signal

    # 10类标签名
    class_names = [
        'IF_0.2', 'IF_0.4', 'IF_0.6',
        'NC', 'OF_0.2', 'OF_0.4', 'OF_0.6',
        'RF_0.2', 'RF_0.4', 'RF_0.6'
    ]

    # 10个 mat 文件名
    file_names = [
        'IF0.2 800~1500 0.mat',
        'IF0.4 800~1500 0.mat',
        'IF0.6 800~1500 0.mat',
        'NC 800~1500 0.mat',
        'OF0.2 800~1500 0.mat',
        'OF0.4 800~1500 0.mat',
        'OF0.6 800~1500 0.mat',
        'RF0.2 800~1500 0.mat',
        'RF0.4 800~1500 0.mat',
        'RF0.6 800~1500 0.mat'
    ]

    # 文件夹路径
    data_folder = ''  # ❗❗你的.mat文件所在文件夹

    # 每列最大长度，比如限制到50
    max_length = 50 * 2048

    data = {}
    for cls, fname in zip(class_names, file_names):
        file_path = os.path.join(data_folder, fname)

        # 加载mat文件
        mat = scipy.io.loadmat(file_path)

        # 取出 Signal.y_values.values
        values = mat['Signal']['y_values'][0, 0]['values'][0, 0]

        # 只取第一列
        series = pd.Series(values[:, 0])

        # 截断最大长度
        series = series.iloc[:max_length].reset_index(drop=True)

        # 存到字典
        data[cls] = series

    # 合并成大表
    df_all = pd.DataFrame(data)

    # 保存
    output_path = 'merged_10class_SDUST.csv'
    df_all.to_csv(output_path, index=False)

    print(f'合并完成，保存到：{output_path}')

    import numpy as np
    import pandas as pd
    import sklearn
    from joblib import dump, load


    # 时间步长 1024 和 重叠率 -0.5
    # window = 1024  step = 512

    def split_data_with_overlap(data, time_steps, lable, overlap_ratio=0.5):
        """
            data:要切分的时间序列数据,可以是一个一维数组或列表。
            time_steps:切分的时间步长,表示每个样本包含的连续时间步数。
            lable: 表示切分数据对应 类别标签
            overlap_ratio:前后帧切分时的重叠率,取值范围为 0 到 1,表示重叠的比例。
        """
        stride = int(time_steps * (1 - overlap_ratio))  # 计算步幅
        samples = (len(data) - time_steps) // stride + 1  # 计算样本数
        # 用于存储生成的数据
        Clasiffy_dataFrame = pd.DataFrame(columns=[x for x in range(time_steps + 1)])
        data_list = []
        for i in range(samples):
            start_idx = i * stride
            end_idx = start_idx + time_steps
            temp_data = data[start_idx:end_idx].tolist()
            temp_data.append(lable)  # 对应哪一类
            data_list.append(temp_data)
        Clasiffy_dataFrame = pd.DataFrame(data_list, columns=Clasiffy_dataFrame.columns)
        return Clasiffy_dataFrame


    # 归一化数据
    def normalize(data):
        ''' (0,1)归一化
            参数:一维时间序列数据
        '''
        s = (data - min(data)) / (max(data) - min(data))
        return s


    # 数据集的制作
    def make_datasets(data_file_csv, split_rate=[0.5, 0.2, 0.3]):
        '''
            参数:
            data_file_csv: 故障分类的数据集,csv格式,数据形状: 119808行  10列
            label_list: 故障分类标签
            split_rate: 训练集、验证集、测试集划分比例

            返回:
            train_set: 训练集数据
            val_set: 验证集数据
            test_set: 测试集数据
        '''
        # 1.读取数据
        origin_data = pd.read_csv(data_file_csv)
        # 2.分割样本点
        time_steps = 2048  # 时间步长
        overlap_ratio = 0.5  # 重叠率
        # 用于存储生成的数据# 10个样本集合
        samples_data = pd.DataFrame(columns=[x for x in range(time_steps + 1)])
        # 记录类别标签
        label = 0
        # 使用iteritems()方法遍历每一列
        for column_name, column_data in origin_data.items():
            # 对数据集的每一维进行归一化
            # column_data = normalize(column_data)
            # 划分样本点  window = 512  overlap_ratio = 0.5  samples = 467 每个类有467个样本
            split_data = split_data_with_overlap(column_data, time_steps, label, overlap_ratio)
            label += 1  # 类别标签递增
            samples_data = pd.concat([samples_data, split_data])
            # 随机打乱样本点顺序
            samples_data = sklearn.utils.shuffle(samples_data)  # 设置随机种子 保证每次实验数据一致

        # 3.分割训练集-、验证集、测试集
        sample_len = len(samples_data)  # 每一类样本数量
        train_len = int(sample_len * split_rate[0])  # 向下取整
        val_len = int(sample_len * split_rate[1])
        train_set = samples_data.iloc[0:train_len, :]
        val_set = samples_data.iloc[train_len:train_len + val_len, :]
        test_set = samples_data.iloc[train_len + val_len:sample_len, :]
        return train_set, val_set, test_set, samples_data


    # 生成数据集
    train_set, val_set, test_set, samples_data = make_datasets('merged_10class_SDUST.csv')

    dump(samples_data, f'{exp_name}_samples_data.joblib')

    # 保存数据
    dump(train_set, 'train_set')
    dump(val_set, 'val_set')
    dump(test_set, 'test_set')

    # 制作数据集和标签
    import torch


    # 这些转换是为了将数据和标签从Pandas数据结构转换为PyTorch可以处理的张量，
    # 以便在神经网络中进行训练和预测。

    def make_data_labels(dataframe,snr_db=None):
        '''
            参数 dataframe: 数据框
            返回 x_data: 数据集     torch.tensor
                y_label: 对应标签值  torch.tensor
        '''
        # 信号值
        x_data = dataframe.iloc[:, 0:-1]
        # 标签值
        y_label = dataframe.iloc[:, -1]
        x_data = torch.tensor(x_data.values).float()
        y_label = torch.tensor(y_label.values.astype('int64'))
        if snr_db is not None:
            x_data = add_noise_by_snr(x_data, snr_db)
        return x_data, y_label


    # 加载数据
    train_set = load('train_set')
    val_set = load('val_set')
    test_set = load('test_set')

    # 制作标签
    train_xdata, train_ylabel = make_data_labels(train_set,snr_db=snr_db)
    val_xdata, val_ylabel = make_data_labels(val_set,snr_db=snr_db)
    test_xdata, test_ylabel = make_data_labels(test_set,snr_db=snr_db)
    # 保存数据
    dump(train_xdata, 'trainX_2048_10c_SDUST')
    dump(val_xdata, 'valX_2048_10c_SDUST')
    dump(test_xdata, 'testX_2048_10c_SDUST')
    dump(train_ylabel, 'trainY_2048_10c_SDUST')
    dump(val_ylabel, 'valY_2048_10c_SDUST')
    dump(test_ylabel, 'testY_2048_10c_SDUST')

    print('数据 形状：')
    print(train_xdata.size(), train_ylabel.shape)
    print(val_xdata.size(), val_ylabel.shape)
    print(test_xdata.size(), test_ylabel.shape)

    # 加载数据集
    def dataloader(batch_size, workers=0):
        # 训练集
        train_xdata = load('trainX_2048_10c_SDUST')
        train_ylabel = load('trainY_2048_10c_SDUST')
        # 验证集
        val_xdata = load('valX_2048_10c_SDUST')
        val_ylabel = load('valY_2048_10c_SDUST')
        # 测试集
        test_xdata = load('testX_2048_10c_SDUST')
        test_ylabel = load('testY_2048_10c_SDUST')

        # 加载数据
        train_loader = Data.DataLoader(dataset=Data.TensorDataset(train_xdata, train_ylabel),
                                       batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
        val_loader = Data.DataLoader(dataset=Data.TensorDataset(val_xdata, val_ylabel),
                                     batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
        test_loader = Data.DataLoader(dataset=Data.TensorDataset(test_xdata, test_ylabel),
                                      batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
        return train_loader, val_loader, test_loader

    batch_size = 32
    epochs = 40
    # 加载数据
    train_loader, val_loader, test_loader = dataloader(batch_size)
    print('数据集长度')
    print(len(train_loader), len(val_loader), len(test_loader))
    # 创建模型
    model = MSFTNet(
        class_num=10,
    ).to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    # 看下这个网络结构总共有多少个参数
    def count_parameters(model):
        params = [p.numel() for p in model.parameters() if p.requires_grad]
        return sum(params)

    model_params = count_parameters(model)

    # 计算FLOPs
    from thop import profile
    input = torch.randn(1, 2048).to(device)  # 假设输入为图像
    flops, params = profile(model, inputs=(input,))
    print(f"FLOPs: {flops / 1e6:.2f} MFLOPs")
    print(f"Params: {params / 1e6:.2f} M")

    # 训练模型
    def model_train(batch_size, epochs, train_loader, val_loader, model, loss_function, optimizer,model_params,flops):
        model = model.to(device)
        train_size = len(train_loader) * batch_size
        val_size = len(val_loader) * batch_size

        best_loss = 0.0
        best_accuracy = 0.0
        best_model = model
        val_y_true_all, val_y_pred_all, val_y_scores_all, val_features_all = [], [], [], []
        train_loss = []
        train_acc = []
        validate_acc = []
        validate_loss = []
        lr_record = []

        start_time = time.time()
        for epoch in range(epochs):
            model.train()
            loss_epoch = 0.
            correct_epoch = 0

            for seq, labels in train_loader:
                seq, labels = seq.to(device), labels.to(device)
                seq = seq.reshape(batch_size,2048)
                optimizer.zero_grad()
                y_pred = model(seq)
                probabilities = F.softmax(y_pred, dim=1)
                predicted_labels = torch.argmax(probabilities, dim=1)
                correct_epoch += (predicted_labels == labels).sum().item()
                loss = loss_function(y_pred, labels)
                loss_epoch += loss.item()
                loss.backward()
                optimizer.step()

            train_accuracy = correct_epoch / train_size
            train_loss.append(loss_epoch / train_size)
            train_acc.append(train_accuracy)
            lr_record.append(optimizer.param_groups[0]['lr'])


            print(f'Epoch: {epoch+1:2} train_Loss: {loss_epoch/train_size:.8f} train_Accuracy:{train_accuracy:.4f}')

            # 验证
            with torch.no_grad():
                loss_validate = 0.
                correct_validate = 0
                for data, label in val_loader:
                    data, label = data.to(device), label.to(device)
                    data = data.reshape(batch_size,2048)
                    pre = model(data)
                    probabilities = F.softmax(pre, dim=1)
                    predicted_labels = torch.argmax(probabilities, dim=1)
                    correct_validate += (predicted_labels == label).sum().item()
                    loss = loss_function(pre, label)
                    loss_validate += loss.item()

                val_accuracy = correct_validate / val_size
                validate_loss.append(loss_validate / val_size)
                validate_acc.append(val_accuracy)

                #计算recall
                val_y_true_all.append(label.cpu().numpy())
                val_y_pred_all.append(predicted_labels.cpu().numpy())
                val_y_scores_all.append(probabilities.cpu().numpy())
                val_features_all.append(data.view(batch_size, -1).cpu().numpy())  # 保持原来展平特征



                print(f'Epoch: {epoch+1:2} val_Loss: {loss_validate/val_size:.8f},  validate_Acc: {val_accuracy:.4f}')
                if val_accuracy > best_accuracy:
                    best_loss = loss_validate/val_size
                    best_accuracy = val_accuracy
                    best_model = model

        # 保存最优模型
        torch.save(best_model.state_dict(), f'{exp_name}best_model_{zhe}.pt')

        print(f'\n训练耗时: {time.time() - start_time:.0f} ')
        print("Best Model:", best_model)
        print(f"Best Validate Accuracy:{best_accuracy:.8f}")
        print(f"Best Validate Loss:{best_loss:.8f}")
        print(f"Best Train Accuracy:{train_acc[-1]:.8f}")
        print(f"Best Train Loss:{train_loss[-1]:.8f}")
        print(f"Model Parameters: {model_params}")
        print(f"FLOPs: {flops / 1e6:.2f} MFLOPs")

        # ======测试集可视化部分======
        best_model.eval()
        y_true_all, y_pred_all, y_scores_all, features_all = [], [], [], []
        model_features_all = []  # 新增一个列表存模型提取的特征
        batch_losses = []
        batch_accuracies = []

        with torch.no_grad():
            for data, label in test_loader:
                data, label = data.to(device), label.to(device)
                data = data.reshape(batch_size, 2048)
                output = best_model(data)
                probs = F.softmax(output, dim=1)
                pred_labels = torch.argmax(probs, dim=1)

                loss = F.cross_entropy(output, label)
                acc = (pred_labels == label).sum().item() / label.size(0)
                batch_losses.append(loss.item())
                batch_accuracies.append(acc)

                y_true_all.append(label.cpu().numpy())
                y_pred_all.append(pred_labels.cpu().numpy())
                y_scores_all.append(probs.cpu().numpy())

                features_all.append(data.view(batch_size, -1).cpu().numpy())  # 保持原来展平特征

                feat = best_model.extract_features(data)  # [batch_size, feature_dim]
                model_features_all.append(feat.cpu().numpy())

        # concat所有
        y_true_all = np.concatenate(y_true_all)
        y_pred_all = np.concatenate(y_pred_all)
        y_scores_all = np.concatenate(y_scores_all)
        features_all = np.concatenate(features_all)
        model_features_all = np.concatenate(model_features_all)  # 新增

        class_names = [
            'IF_0.2',
            'IF_0.4',
            'IF_0.6',
            'NC',
            'OF_0.2',
            'OF_0.4',
            'OF_0.6',
            'RF_0.2',
            'RF_0.4',
            'RF_0.6'
        ]  # 10类名

        avg_loss = np.mean(batch_losses)
        avg_acc = np.mean(batch_accuracies)

        from sklearn.metrics import f1_score, recall_score

        # 计算宏平均的 F1 分数和召回率
        f1 = f1_score(y_true_all, y_pred_all, average='macro')
        recall = recall_score(y_true_all, y_pred_all, average='macro')

        print(f"F1 Score (macro): {f1:.8f}")
        print(f"Recall (macro): {recall:.8f}")
        print(f"Best testdata Accuracy:{avg_acc:.8f}")
        print(f"Best testdate Loss:{avg_loss:.8f}")

        test_loss_list = [avg_loss] * epochs
        test_acc_list = [avg_acc] * epochs
        batch_loss_padded = batch_losses + [np.nan] * (epochs - len(batch_losses))
        batch_acc_padded = batch_accuracies + [np.nan] * (epochs - len(batch_accuracies))
        f1_list = [f1] * epochs
        recall_list = [recall] * epochs
        #打印长度
        # print(len(train_loss), len(validate_loss), len(test_loss_list), len(test_acc_list), len(batch_loss_padded), len(batch_acc_padded), len(f1_list), len(recall_list))
        df = pd.DataFrame(
            {
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': validate_loss,
                'val_acc': validate_acc,
                'lr': lr_record,
                "model_params": model_params,
                "MFLOPs": flops / 1e6,
                "test_loss": test_loss_list,
                "test_acc": test_acc_list,
                "f1_score": f1_list,
                "recall": recall_list
            }
        )

        filename = f"{exp_name}_SDUST_{zhe}"
        df.to_csv(f'{filename}_output.csv', index=False, float_format="%.10f")

        visualize_all_results(
            file_name=filename,
            train_loss=train_loss,
            val_loss=validate_loss,
            train_acc=train_acc,
            val_acc=validate_acc,
            y_true=y_true_all,
            y_pred=y_pred_all,
            y_scores=y_scores_all,
            features_model= model_features_all,
            features=features_all,
            class_names=class_names,
            learning_rates=lr_record,
            epochs=epochs,
            bins=10,
            exp_name=exp_name,
            # t_y_true=val_y_true_all,
            # t_y_pred=val_y_pred_all,
            # t_y_scores=val_y_scores_all,
            # t_features = val_features_all,
        )
    model_train(batch_size, epochs, train_loader, val_loader, model, loss_function, optimizer,model_params, flops)

    if zhe == num_exp:
        #创建csv表
        target_path = f'{exp_name}_summary_{file_names[3]}.csv'
        if os.path.exists(target_path):
            os.remove(target_path)
        for P in range(1,num_exp+1):
            #读取每轮次的csv，取test_loss最低的行，打到大csv里面
            df = pd.read_csv(f'{exp_name}_SDUST_{P}_output.csv')
            # 找到 val_acc 最大的那一行（返回的是 DataFrame）
            min_row = df.loc[df['val_acc'].idxmax()].to_frame().T  # 转置成一行
            # 第一次写入用 header=True，其余用 header=False
            header_flag = (P == 1)
            # 写入目标文件（追加）
            min_row.to_csv(target_path, mode='a', index=False, header=header_flag)

        #保存csv表
        df = pd.read_csv(target_path)
        avg_row = df.mean(numeric_only=True).to_frame().T
        avg_row.index = ['avg']  # 将这一行的 index 设置为 'avg'
        df_with_avg = pd.concat([df, avg_row])
        df_with_avg.to_csv(target_path, index=False, float_format="%.10f")
        print(f"保存到{target_path}成功！")