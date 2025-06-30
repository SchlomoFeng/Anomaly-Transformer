import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import os
import time
import argparse
import logging
from datetime import datetime
from collections import deque
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 导入项目模块
from model.AnomalyTransformer import AnomalyTransformer
from data_factory.data_loader import CustomCSVSegLoader


class OnlineAnomalyDetector:
    """在线异常检测器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 设置日志
        self.setup_logging()
        
        # 初始化模型
        self.model = None
        self.scaler = None
        self.threshold = None
        
        # 滑动窗口
        self.window_size = config.win_size
        self.history_window = deque(maxlen=self.window_size)
        
        # 统计信息
        self.detection_count = 0
        self.anomaly_count = 0
        self.start_time = time.time()
        
        # 结果存储
        self.results_buffer = deque(maxlen=1000)  # 存储最近1000个检测结果
        
        # 在线检测状态
        self.data_index = 0  # 当前处理的数据索引
        self.data_cache = None  # 缓存的数据
        
        self.logger.info(f"初始化在线异常检测器完成 - 设备: {self.device}")
    
    def setup_logging(self):
        """设置日志系统"""
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        log_filename = os.path.join(log_dir, f"anomaly_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_model_and_scaler(self):
        """加载训练好的模型和预处理器"""
        try:
            # 1. 加载模型
            self.model = AnomalyTransformer(
                win_size=self.config.win_size,
                enc_in=self.config.input_c,
                c_out=self.config.output_c,
                e_layers=3
            )
            
            # 检查模型文件是否存在
            if not os.path.exists(self.config.model_path):
                raise FileNotFoundError(f"模型文件不存在: {self.config.model_path}")
            
            # 加载模型权重
            checkpoint = torch.load(self.config.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            self.model.to(self.device)
            self.model.eval()
            
            self.logger.info(f"成功加载模型: {self.config.model_path}")
            
            # 2. 加载或创建预处理器
            scaler_path = self.config.model_path.replace('.pth', '_scaler.pkl')
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                self.logger.info(f"成功加载预处理器: {scaler_path}")
            else:
                # 使用训练数据创建预处理器
                self.logger.info("未找到预处理器文件，基于训练数据创建...")
                self._create_scaler_from_training_data()
            
            # 3. 计算阈值
            self._calculate_threshold()
            
            return True
            
        except Exception as e:
            self.logger.error(f"加载模型和预处理器失败: {str(e)}")
            return False
    
    def _create_scaler_from_training_data(self):
        """基于训练数据创建预处理器"""
        try:
            # 使用CustomCSVSegLoader来获取训练数据的scaler
            train_loader = CustomCSVSegLoader(
                self.config.training_data_path, 
                self.config.win_size, 
                step=1, 
                mode='train'
            )
            self.scaler = train_loader.scaler
            
            # 保存scaler
            scaler_path = self.config.model_path.replace('.pth', '_scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            self.logger.info(f"成功创建并保存预处理器: {scaler_path}")
            
        except Exception as e:
            self.logger.error(f"创建预处理器失败: {str(e)}")
            raise
    
    def _calculate_threshold(self):
        """计算异常检测阈值"""
        try:
            self.logger.info("开始计算异常检测阈值...")
            
            # 使用训练数据计算阈值
            train_loader = CustomCSVSegLoader(
                self.config.training_data_path,
                self.config.win_size,
                step=self.config.win_size,
                mode='train'
            )
            
            anomaly_scores = []
            temperature = 50
            criterion = nn.MSELoss(reduce=False)
            
            with torch.no_grad():
                for i in range(min(len(train_loader), 100)):  # 使用前100个样本计算阈值
                    try:
                        data, _ = train_loader[i]
                        input_tensor = torch.FloatTensor(data).unsqueeze(0).to(self.device)
                        
                        output, series, prior, _ = self.model(input_tensor)
                        loss = torch.mean(criterion(input_tensor, output), dim=-1)
                        
                        # 计算异常分数
                        series_loss = 0.0
                        prior_loss = 0.0
                        for u in range(len(prior)):
                            if u == 0:
                                series_loss = self._my_kl_loss(series[u], (
                                    prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.config.win_size)
                                ).detach()) * temperature
                                prior_loss = self._my_kl_loss((
                                    prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.config.win_size)
                                ), series[u].detach()) * temperature
                            else:
                                series_loss += self._my_kl_loss(series[u], (
                                    prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.config.win_size)
                                ).detach()) * temperature
                                prior_loss += self._my_kl_loss((
                                    prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.config.win_size)
                                ), series[u].detach()) * temperature
                        
                        metric = torch.softmax((-series_loss - prior_loss), dim=-1)
                        anomaly_score = (metric * loss).detach().cpu().numpy()
                        anomaly_scores.extend(anomaly_score.flatten())
                        
                    except Exception as e:
                        self.logger.warning(f"计算阈值时跳过第{i}个样本: {str(e)}")
                        continue
            
            if anomaly_scores:
                # 使用指定的异常比例计算阈值
                self.threshold = np.percentile(anomaly_scores, 100 - self.config.anormly_ratio)
                self.logger.info(f"计算得到异常检测阈值: {self.threshold:.6f}")
            else:
                # 如果无法计算阈值，使用默认值
                self.threshold = 0.1
                self.logger.warning(f"无法计算阈值，使用默认值: {self.threshold}")
                
        except Exception as e:
            self.logger.error(f"计算阈值失败: {str(e)}")
            self.threshold = 0.1
            self.logger.warning(f"使用默认阈值: {self.threshold}")
    
    def _my_kl_loss(self, p, q):
        """KL散度损失函数"""
        res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
        return torch.mean(torch.sum(res, dim=-1), dim=1)
    
    def preprocess_data(self, raw_data):
        """预处理数据"""
        try:
            # 确保数据是numpy数组
            if isinstance(raw_data, list):
                raw_data = np.array(raw_data)
            
            # 重塑为二维数组
            if raw_data.ndim == 1:
                raw_data = raw_data.reshape(-1, 1)
            
            # 标准化
            processed_data = self.scaler.transform(raw_data)
            return processed_data.flatten()
            
        except Exception as e:
            self.logger.error(f"数据预处理失败: {str(e)}")
            raise
    
    def detect_anomaly(self, new_data_point, timestamp=None):
        """检测单个数据点的异常"""
        try:
            # 预处理新数据点
            processed_point = self.preprocess_data([new_data_point])
            
            # 更新滑动窗口
            self.history_window.append(processed_point[0])
            
            # 只有当窗口填满时才进行检测
            if len(self.history_window) < self.window_size:
                return {
                    'timestamp': timestamp or datetime.now().isoformat(),
                    'raw_value': new_data_point,
                    'is_anomaly': False,
                    'anomaly_score': 0.0,
                    'status': 'warming_up',
                    'window_fill_rate': len(self.history_window) / self.window_size
                }
            
            # 转换为模型输入格式
            window_data = np.array(list(self.history_window)).reshape(1, self.window_size, 1)
            input_tensor = torch.FloatTensor(window_data).to(self.device)
            
            # 模型推理
            with torch.no_grad():
                output, series, prior, _ = self.model(input_tensor)
                
                # 计算异常分数
                anomaly_score = self._calculate_anomaly_score(input_tensor, output, series, prior)
                
                # 判断是否异常
                is_anomaly = anomaly_score > self.threshold
                
                # 更新统计信息
                self.detection_count += 1
                if is_anomaly:
                    self.anomaly_count += 1
                
                # 构建结果
                result = {
                    'timestamp': timestamp or datetime.now().isoformat(),
                    'raw_value': new_data_point,
                    'processed_value': processed_point[0],
                    'is_anomaly': bool(is_anomaly),
                    'anomaly_score': float(anomaly_score),
                    'threshold': float(self.threshold),
                    'confidence': float(anomaly_score / self.threshold) if self.threshold > 0 else 0.0,
                    'status': 'normal'
                }
                
                # 存储结果
                self.results_buffer.append(result)
                
                # 记录异常
                if is_anomaly:
                    self.logger.warning(f"检测到异常! 分数: {anomaly_score:.6f}, 阈值: {self.threshold:.6f}, 原始值: {new_data_point}")
                
                return result
                
        except Exception as e:
            self.logger.error(f"异常检测失败: {str(e)}")
            return {
                'timestamp': timestamp or datetime.now().isoformat(),
                'raw_value': new_data_point,
                'is_anomaly': False,
                'anomaly_score': 0.0,
                'status': 'error',
                'error_message': str(e)
            }
    
    def _calculate_anomaly_score(self, input_tensor, output, series, prior):
        """计算异常分数"""
        temperature = 50
        criterion = nn.MSELoss(reduction='none')
        
        loss = torch.mean(criterion(input_tensor, output), dim=-1)
        
        series_loss = 0.0
        prior_loss = 0.0
        for u in range(len(prior)):
            if u == 0:
                series_loss = self._my_kl_loss(series[u], (
                    prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.window_size)
                ).detach()) * temperature
                prior_loss = self._my_kl_loss((
                    prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.window_size)
                ), series[u].detach()) * temperature
            else:
                series_loss += self._my_kl_loss(series[u], (
                    prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.window_size)
                ).detach()) * temperature
                prior_loss += self._my_kl_loss((
                    prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.window_size)
                ), series[u].detach()) * temperature
        
        metric = torch.softmax((-series_loss - prior_loss), dim=-1)
        anomaly_score = (metric * loss).detach().cpu().numpy()
        
        return float(np.mean(anomaly_score))
    
    def load_data_for_online_detection(self, csv_path):
        """为在线检测加载数据"""
        try:
            df = pd.read_csv(csv_path)
            
            # 验证CSV格式
            required_columns = ['timestamp', 'value']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"CSV文件必须包含列: {required_columns}")
            
            # 按时间戳排序
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            self.data_cache = df
            self.data_index = 0
            
            self.logger.info(f"成功加载数据文件: {csv_path}, 共 {len(df)} 条记录")
            return True
            
        except Exception as e:
            self.logger.error(f"加载数据文件失败: {str(e)}")
            return False
    
    def batch_detect_from_csv(self, csv_path):
        """从CSV文件批量检测异常"""
        try:
            self.logger.info(f"开始批量检测: {csv_path}")
            
            # 读取CSV文件
            df = pd.read_csv(csv_path)
            
            # 验证CSV格式
            required_columns = ['timestamp', 'value']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"CSV文件必须包含列: {required_columns}")
            
            # 按时间戳排序
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            results = []
            total_rows = len(df)
            
            self.logger.info(f"开始处理 {total_rows} 条数据...")
            
            for idx, row in df.iterrows():
                result = self.detect_anomaly(row['value'], row['timestamp'].isoformat())
                
                # 如果有标签列，添加真实标签
                if 'label' in df.columns:
                    result['true_label'] = int(row['label'])
                
                results.append(result)
                
                # 进度报告
                if (idx + 1) % 100 == 0:
                    progress = (idx + 1) / total_rows * 100
                    self.logger.info(f"处理进度: {progress:.1f}% ({idx + 1}/{total_rows})")
            
            self.logger.info(f"批量检测完成，总计处理 {len(results)} 条数据")
            return results
            
        except Exception as e:
            self.logger.error(f"批量检测失败: {str(e)}")
            raise
    
    def evaluate_performance(self, results):
        """评估检测性能"""
        try:
            # 筛选有真实标签的结果
            labeled_results = [r for r in results if 'true_label' in r and r['status'] != 'error']
            
            if not labeled_results:
                self.logger.warning("没有标签数据，无法计算性能指标")
                return None
            
            # 提取预测和真实标签
            y_true = [r['true_label'] for r in labeled_results]
            y_pred = [int(r['is_anomaly']) for r in labeled_results]
            
            # 计算性能指标
            accuracy = accuracy_score(y_true, y_pred)
            precision, recall, f1_score, _ = precision_recall_fscore_support(
                y_true, y_pred, average='binary', zero_division=0
            )
            
            performance = {
                'total_samples': len(labeled_results),
                'true_positives': sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 1),
                'false_positives': sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 1),
                'true_negatives': sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 0),
                'false_negatives': sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 0),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score
            }
            
            self.logger.info("=" * 50)
            self.logger.info("性能评估结果:")
            self.logger.info(f"准确率 (Accuracy):  {accuracy:.4f}")
            self.logger.info(f"精确率 (Precision): {precision:.4f}")
            self.logger.info(f"召回率 (Recall):    {recall:.4f}")
            self.logger.info(f"F1分数 (F1-Score):  {f1_score:.4f}")
            self.logger.info("=" * 50)
            
            return performance
            
        except Exception as e:
            self.logger.error(f"性能评估失败: {str(e)}")
            return None
    
    def save_results(self, results, output_path):
        """保存检测结果"""
        try:
            df = pd.DataFrame(results)
            df.to_csv(output_path, index=False)
            self.logger.info(f"结果已保存到: {output_path}")
            
        except Exception as e:
            self.logger.error(f"保存结果失败: {str(e)}")
    
    def get_statistics(self):
        """获取统计信息"""
        runtime = time.time() - self.start_time
        anomaly_rate = (self.anomaly_count / self.detection_count * 100) if self.detection_count > 0 else 0
        detection_rate = self.detection_count / runtime if runtime > 0 else 0
        
        return {
            'total_detections': self.detection_count,
            'total_anomalies': self.anomaly_count,
            'anomaly_rate': anomaly_rate,
            'runtime_seconds': runtime,
            'detection_rate_per_second': detection_rate,
            'threshold': self.threshold,
            'device': str(self.device)
        }
    
    def run_online_detection(self, data_source_path, detection_interval=1):
        """运行在线异常检测 - 修正版本"""
        self.logger.info("开始在线异常检测模式...")
        
        # 加载数据文件
        if not self.load_data_for_online_detection(data_source_path):
            self.logger.error("无法加载数据文件，退出在线检测")
            return
        
        try:
            while self.data_index < len(self.data_cache):
                # 获取当前数据点
                current_row = self.data_cache.iloc[self.data_index]
                timestamp = current_row['timestamp'].isoformat()
                value = current_row['value']
                
                # 检测异常
                result = self.detect_anomaly(value, timestamp)
                
                # 输出结果
                status_icon = "⚠️" if result['is_anomaly'] else "✅"
                status_text = "异常检测" if result['is_anomaly'] else "正常数据"
                
                print(f"{status_icon} {status_text} - 时间: {timestamp}, 值: {value:.6f}, 分数: {result['anomaly_score']:.6f}")
                
                # 如果是异常，额外记录详细信息
                if result['is_anomaly']:
                    print(f"    置信度: {result['confidence']:.2f}, 阈值: {result['threshold']:.6f}")
                
                # 移动到下一个数据点
                self.data_index += 1
                
                # 等待下次检测
                time.sleep(detection_interval)
                
        except KeyboardInterrupt:
            self.logger.info("在线检测已停止")
        finally:
            # 输出统计信息
            stats = self.get_statistics()
            self.logger.info(f"检测统计: {stats}")
            print(f"\n 检测完成统计:")
            print(f"   总检测数: {stats['total_detections']}")
            print(f"   异常数量: {stats['total_anomalies']}")
            print(f"   异常率: {stats['anomaly_rate']:.2f}%")
            print(f"   处理数据索引: {self.data_index}/{len(self.data_cache) if self.data_cache is not None else 0}")


def main():
    parser = argparse.ArgumentParser(description='在线异常检测系统')
    
    # 模型和数据参数
    parser.add_argument('--model_path', type=str, 
                        default='checkpoints/credit_checkpoint.pth',
                        help='训练好的模型文件路径')
    parser.add_argument('--training_data_path', type=str,
                        default='data/YT.11PI_04019.PV.csv',
                        help='训练数据文件路径（用于创建预处理器和计算阈值）')
    parser.add_argument('--test_data_path', type=str,
                        default='data/YT.11PI_45201.PV.csv',
                        help='测试数据文件路径')
    
    # 模型配置参数
    parser.add_argument('--win_size', type=int, default=100,
                        help='时间窗口大小')
    parser.add_argument('--input_c', type=int, default=1,
                        help='输入通道数')
    parser.add_argument('--output_c', type=int, default=1,
                        help='输出通道数')
    parser.add_argument('--anormly_ratio', type=float, default=0.5,
                        help='异常比例（用于计算阈值）')
    
    # 运行模式
    parser.add_argument('--mode', type=str, 
                        choices=['batch', 'online', 'test'],
                        default='online',
                        help='运行模式: batch-批量检测, online-在线检测, test-测试模式')
    parser.add_argument('--output_path', type=str,
                        default='detection_results.csv',
                        help='结果输出路径')
    parser.add_argument('--detection_interval', type=int, default=1,
                        help='在线检测间隔（秒）')
    
    args = parser.parse_args()
    
    # 创建检测器
    detector = OnlineAnomalyDetector(args)
    
    # 加载模型和预处理器
    if not detector.load_model_and_scaler():
        print("❌ 模型加载失败，程序退出")
        return
    
    print("✅ 模型加载成功")
    
    try:
        if args.mode == 'batch':
            # 批量检测模式
            print(f"🔍 开始批量检测: {args.test_data_path}")
            results = detector.batch_detect_from_csv(args.test_data_path)
            
            # 保存结果
            detector.save_results(results, args.output_path)
            
            # 评估性能
            performance = detector.evaluate_performance(results)
            
            # 显示统计信息
            stats = detector.get_statistics()
            print("\n 检测统计:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            
        elif args.mode == 'online':
            # 在线检测模式
            print(f"启动在线检测模式，数据文件: {args.test_data_path}")
            print(f"   检测间隔: {args.detection_interval}秒")
            print("   按 Ctrl+C 停止检测")
            detector.run_online_detection(args.test_data_path, args.detection_interval)
            
        elif args.mode == 'test':
            # 测试模式 - 检测几个样本
            print("测试模式 - 检测前10个样本")
            df = pd.read_csv(args.test_data_path)
            
            for i in range(min(10, len(df))):
                timestamp = df['timestamp'].iloc[i] if 'timestamp' in df.columns else None
                value = df['value'].iloc[i]
                result = detector.detect_anomaly(value, timestamp)
                
                status_icon = "⚠️" if result['is_anomaly'] else "✅"
                print(f"{status_icon} 样本 {i+1}: 时间={timestamp}, 值={value:.6f}, 分数={result['anomaly_score']:.6f}, 异常={result['is_anomaly']}")
    
    except Exception as e:
        print(f"❌ 执行失败: {str(e)}")
        detector.logger.error(f"执行失败: {str(e)}")
    
    finally:
        # 输出最终统计信息
        stats = detector.get_statistics()
        print(f"\n 最终统计: 总检测={stats['total_detections']}, 异常={stats['total_anomalies']}, 异常率={stats['anomaly_rate']:.2f}%")


if __name__ == "__main__":
    main()
