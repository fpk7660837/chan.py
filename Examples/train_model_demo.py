"""
训练模型示例

演示完整的模型训练流程：
1. 准备训练数据
2. 配置参数
3. 训练模型
4. 保存模型
5. 评估模型
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Chan import CChan
from Common.CEnum import KL_TYPE, DATA_SRC
from Config.MLConfig import MLConfig
from ML.Training.Trainer import Trainer
from ML.Utils.ModelIO import ModelIO
from ML.Prediction.Predictor import Predictor
from ML.FeatureEngine.BSPFeatureExtractor import BSPFeatureExtractor
from ML.Evaluation.Metrics import Metrics
import numpy as np


def prepare_training_data():
    """准备训练数据"""
    print("=" * 80)
    print("Step 1: Preparing training data...")
    print("=" * 80)

    # 训练数据：多只股票，多个时间段
    train_codes = ['600000', '600016', '600036', '600519', '000001']
    train_chan_list = []

    for code in train_codes:
        try:
            print(f"\nLoading data for {code}...")
            chan = CChan(
                code=code,
                begin_time='2020-01-01',
                end_time='2022-12-31',
                data_src=DATA_SRC.AKSHARE,
                lv_list=[KL_TYPE.K_DAY]  # 使用日线级别
            )
            train_chan_list.append(chan)
            print(f"  Loaded {code} successfully")
        except Exception as e:
            print(f"  Failed to load {code}: {e}")
            continue

    print(f"\nTotal training data: {len(train_chan_list)} stocks")
    return train_chan_list


def prepare_test_data():
    """准备测试数据"""
    print("\n" + "=" * 80)
    print("Step 2: Preparing test data...")
    print("=" * 80)

    # 测试数据：相同股票，不同时间段
    test_codes = ['600000', '600016', '600036', '600519', '000001']
    test_chan_list = []

    for code in test_codes:
        try:
            print(f"\nLoading test data for {code}...")
            chan = CChan(
                code=code,
                begin_time='2023-01-01',
                end_time='2023-12-31',
                data_src=DATA_SRC.AKSHARE,
                lv_list=[KL_TYPE.K_DAY]
            )
            test_chan_list.append(chan)
            print(f"  Loaded {code} successfully")
        except Exception as e:
            print(f"  Failed to load {code}: {e}")
            continue

    print(f"\nTotal test data: {len(test_chan_list)} stocks")
    return test_chan_list


def train_model(train_chan_list, model_type='lightgbm'):
    """训练模型"""
    print("\n" + "=" * 80)
    print(f"Step 3: Training {model_type} model...")
    print("=" * 80)

    # 创建配置
    config = MLConfig()

    # 可以自定义配置
    config.update_config('label', {
        'lookforward_bars': 20,  # 未来20根K线
        'threshold_pct': 0.05,   # 5%收益率阈值
    })

    config.update_config('training', {
        'test_size': 0.2,
        'early_stopping_rounds': 30,
        'verbose_eval': 5,
    })

    # 创建训练器
    trainer = Trainer(config.to_dict())

    # 训练模型
    model = trainer.train(train_chan_list, model_type=model_type)

    return model, config


def save_model(model, config, version='v1.0'):
    """保存模型"""
    print("\n" + "=" * 80)
    print("Step 4: Saving model...")
    print("=" * 80)

    # 创建ModelIO
    model_io = ModelIO('./models')

    # 准备元数据
    metadata = {
        'description': 'Chan theory buy/sell point quality prediction model',
        'config': config.to_dict(),
    }

    # 保存模型
    model_path = model_io.save(model, version=version, metadata=metadata)

    return model_path


def evaluate_model(model, test_chan_list, config):
    """评估模型"""
    print("\n" + "=" * 80)
    print("Step 5: Evaluating model on test set...")
    print("=" * 80)

    # 创建预测器
    feature_extractor = BSPFeatureExtractor(config.feature_config)
    predictor = Predictor(model, feature_extractor)

    # 从测试集提取买卖点
    from ML.Training.Trainer import Trainer
    trainer = Trainer(config.to_dict())
    test_bsp_list = trainer._extract_bsp_from_chan_list(test_chan_list)

    print(f"Total test buy/sell points: {len(test_bsp_list)}")

    # 提取特征和标签
    X_test, feature_names = trainer._extract_features(test_bsp_list)
    y_test, returns_test = trainer.label_builder.build_labels(test_bsp_list)

    # 预测
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # 计算指标
    metrics = Metrics.calculate_all_metrics(
        y_test, y_pred, y_proba,
        returns=returns_test
    )

    # 打印指标
    Metrics.print_metrics(metrics, title="Model Evaluation on Test Set")

    return metrics


def main():
    """主流程"""
    print("\n" + "=" * 80)
    print("Chan Theory ML Training Demo")
    print("=" * 80)

    # 1. 准备数据
    train_chan_list = prepare_training_data()

    if len(train_chan_list) == 0:
        print("\nNo training data available. Exiting...")
        return

    test_chan_list = prepare_test_data()

    # 2. 训练模型
    model, config = train_model(train_chan_list, model_type='lightgbm')

    # 3. 保存模型
    model_path = save_model(model, config, version='v1.0')

    # 4. 评估模型
    if len(test_chan_list) > 0:
        metrics = evaluate_model(model, test_chan_list, config)

    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print("=" * 80)
    print(f"\nModel saved to: {model_path}")
    print("You can now use this model for prediction (see predict_demo.py)")
    print("Or run backtest (see backtest_demo.py)")


if __name__ == '__main__':
    main()
