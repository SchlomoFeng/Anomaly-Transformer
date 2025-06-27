import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from model.AnomalyTransformer import AnomalyTransformer
from data_factory.data_loader import get_loader_segment


def my_kl_loss(p, q):
    """KL散度损失函数"""
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


class AnomalyTester:
    def __init__(self, config):
        self.dataset = config.dataset
        self.data_path = config.data_path
        self.model_path = config.model_path
        self.batch_size = config.batch_size
        self.win_size = config.win_size
        self.input_c = config.input_c
        self.output_c = config.output_c
        self.anormly_ratio = config.anormly_ratio
        
        # 设备配置
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # 构建模型
        self.build_model()
        
        # 加载数据
        self.load_data()
    
    def build_model(self):
        """构建并加载模型"""
        self.model = AnomalyTransformer(
            win_size=self.win_size, 
            enc_in=self.input_c, 
            c_out=self.output_c, 
            e_layers=3
        )
        
        if torch.cuda.is_available():
            self.model.cuda()
        
        # 加载预训练模型
        if os.path.exists(self.model_path):
            print(f"Loading model from: {self.model_path}")
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            print("Model loaded successfully!")
        else:
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
    
    def load_data(self):
        """加载测试数据"""
        # 训练集用于统计正常数据的能量分布
        self.train_loader = get_loader_segment(
            self.data_path, 
            batch_size=self.batch_size, 
            win_size=self.win_size,
            mode='train',
            dataset=self.dataset
        )
        
        # 测试集用于阈值计算
        self.thre_loader = get_loader_segment(
            self.data_path, 
            batch_size=self.batch_size, 
            win_size=self.win_size,
            mode='thre',
            dataset=self.dataset
        )
        
        print("Data loaders created successfully!")
    
    def calculate_anomaly_scores(self, data_loader, mode="train"):
        """计算异常分数"""
        criterion = nn.MSELoss(reduce=False)
        temperature = 50
        attens_energy = []
        labels_list = []
        
        print(f"Calculating anomaly scores for {mode} data...")
        
        with torch.no_grad():
            for i, (input_data, labels) in enumerate(data_loader):
                input_tensor = input_data.float().to(self.device)
                output, series, prior, _ = self.model(input_tensor)
                
                # 重构损失
                loss = torch.mean(criterion(input_tensor, output), dim=-1)
                
                # 关联差异损失
                series_loss = 0.0
                prior_loss = 0.0
                
                for u in range(len(prior)):
                    normalized_prior = prior[u] / torch.unsqueeze(
                        torch.sum(prior[u], dim=-1), dim=-1
                    ).repeat(1, 1, 1, self.win_size)
                    
                    if u == 0:
                        series_loss = my_kl_loss(
                            series[u], 
                            normalized_prior.detach()
                        ) * temperature
                        prior_loss = my_kl_loss(
                            normalized_prior, 
                            series[u].detach()
                        ) * temperature
                    else:
                        series_loss += my_kl_loss(
                            series[u], 
                            normalized_prior.detach()
                        ) * temperature
                        prior_loss += my_kl_loss(
                            normalized_prior, 
                            series[u].detach()
                        ) * temperature
                
                # 计算最终的异常分数
                metric = torch.softmax((-series_loss - prior_loss), dim=-1)
                anomaly_score = metric * loss
                
                attens_energy.append(anomaly_score.detach().cpu().numpy())
                labels_list.append(labels.numpy())
                
                if (i + 1) % 50 == 0:
                    print(f"Processed {i + 1} batches...")
        
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        labels_array = np.concatenate(labels_list, axis=0).reshape(-1)
        
        return attens_energy, labels_array
    
    def test(self):
        """执行测试并计算评估指标"""
        print("======================TEST MODE======================")
        
        # 1. 计算训练集的异常分数（用于建立正常基线）
        train_energy, _ = self.calculate_anomaly_scores(self.train_loader, "train")
        
        # 2. 计算测试集的异常分数和标签
        test_energy, test_labels = self.calculate_anomaly_scores(self.thre_loader, "test")
        
        # 3. 确定异常检测阈值
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        threshold = np.percentile(combined_energy, 100 - self.anormly_ratio)
        print(f"Anomaly detection threshold: {threshold:.6f}")
        
        # 4. 进行异常检测
        predictions = (test_energy > threshold).astype(int)
        ground_truth = test_labels.astype(int)
        
        print(f"Predictions shape: {predictions.shape}")
        print(f"Ground truth shape: {ground_truth.shape}")
        
        # 5. 检测调整（根据论文的调整策略）
        predictions = self.adjust_predictions(predictions, ground_truth)
        
        # 6. 计算评估指标
        accuracy = accuracy_score(ground_truth, predictions)
        precision, recall, f_score, _ = precision_recall_fscore_support(
            ground_truth, predictions, average='binary'
        )
        
        # 7. 输出结果
        print("\n======================RESULTS======================")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f_score:.4f}")
        print("===================================================")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f_score,
            'threshold': threshold
        }
    
    def adjust_predictions(self, pred, gt):
        """根据论文策略调整预测结果"""
        pred = pred.copy()
        anomaly_state = False
        
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                # 向前调整
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                # 向后调整
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1
        
        return pred


def main():
    parser = argparse.ArgumentParser(description='Anomaly Transformer Testing')
    
    # 模型和数据参数
    parser.add_argument('--model_path', type=str,
                        default='checkpoints/credit_checkpoint.pth',
                        help='Path to the .pth model file')
    parser.add_argument('--dataset', type=str,
                        choices=['SMD', 'MSL', 'SMAP', 'PSM','credit'],
                        default='credit',
                        help='Dataset name')
    parser.add_argument('--data_path', type=str,
                        default='data/YT.11PI_04019.PV.csv',
                        help='Path to the dataset directory')
    
    # 模型配置参数
    parser.add_argument('--win_size', type=int, default=100,
                        help='Window size for time series')
    parser.add_argument('--input_c', type=int, default=1,
                        help='Number of input channels')
    parser.add_argument('--output_c', type=int, default=1,
                        help='Number of output channels')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for testing')
    parser.add_argument('--anormly_ratio', type=float, default=0.5,
                        help='Anomaly ratio for threshold calculation')
    
    args = parser.parse_args()
    
    # 创建测试器并运行测试
    tester = AnomalyTester(args)
    results = tester.test()
    
    # 保存结果
    results_file = f"test_results_{args.dataset}.txt"
    with open(results_file, 'w') as f:
        f.write("Anomaly Transformer Test Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Accuracy:  {results['accuracy']:.4f}\n")
        f.write(f"Precision: {results['precision']:.4f}\n")
        f.write(f"Recall:    {results['recall']:.4f}\n")
        f.write(f"F1-Score:  {results['f1_score']:.4f}\n")
        f.write(f"Threshold: {results['threshold']:.6f}\n")
    
    print(f"Results saved to: {results_file}")


if __name__ == '__main__':
    main()
