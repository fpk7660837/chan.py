"""
回测示例

演示如何使用训练好的模型进行回测：
1. 加载模型
2. 准备回测数据
3. 运行回测
4. 分析回测结果
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Chan import CChan
from Common.CEnum import KL_TYPE, DATA_SRC
from ML.Utils.ModelIO import ModelIO
from ML.Prediction.Predictor import Predictor
from ML.FeatureEngine.BSPFeatureExtractor import BSPFeatureExtractor
from ML.Backtest.MLBacktest import MLBacktest


def load_model(version=None):
    """加载模型"""
    print("=" * 80)
    print("Step 1: Loading model...")
    print("=" * 80)

    model_io = ModelIO('./models')

    # 列出可用版本
    versions = model_io.list_versions()
    print(f"\nAvailable model versions: {versions}")

    # 加载模型
    model = model_io.load(version=version)
    print(f"Model loaded successfully!")

    # 加载元数据
    metadata = model_io.load_metadata(version=version)
    if metadata:
        print(f"\nModel info:")
        print(f"  Type: {metadata.get('model_type', 'Unknown')}")
        print(f"  Version: {metadata.get('version', 'Unknown')}")

    return model, metadata


def prepare_backtest_data():
    """准备回测数据"""
    print("\n" + "=" * 80)
    print("Step 2: Preparing backtest data...")
    print("=" * 80)

    # 回测数据：多只股票
    test_codes = ['600000', '600016', '600036', '600519']
    test_chan_list = []

    for code in test_codes:
        try:
            print(f"\nLoading backtest data for {code}...")
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

    print(f"\nTotal backtest data: {len(test_chan_list)} stocks")
    return test_chan_list


def run_backtest(model, test_chan_list, metadata, config_override=None):
    """运行回测"""
    print("\n" + "=" * 80)
    print("Step 3: Running backtest...")
    print("=" * 80)

    # 创建特征提取器
    feature_config = metadata.get('config', {}).get('feature_config', {})
    feature_extractor = BSPFeatureExtractor(feature_config)

    # 创建预测器
    predictor = Predictor(model, feature_extractor)

    # 回测配置
    backtest_config = metadata.get('config', {}).get('backtest_config', {})

    # 可以覆盖配置
    if config_override:
        backtest_config.update(config_override)

    print(f"\nBacktest configuration:")
    print(f"  Score threshold: {backtest_config.get('score_threshold', 0.7)}")
    print(f"  Holding period: {backtest_config.get('holding_period', 20)} bars")
    print(f"  Initial capital: {backtest_config.get('initial_capital', 100000):,.0f}")
    print(f"  Commission rate: {backtest_config.get('commission_rate', 0.0003):.4f}")
    print(f"  Slippage: {backtest_config.get('slippage', 0.001):.4f}")

    # 创建回测引擎
    backtest = MLBacktest(predictor, backtest_config)

    # 运行回测
    metrics = backtest.run(test_chan_list)

    return metrics


def compare_thresholds(model, test_chan_list, metadata):
    """比较不同阈值的回测结果"""
    print("\n" + "=" * 80)
    print("Step 4: Comparing different score thresholds...")
    print("=" * 80)

    # 创建特征提取器
    feature_config = metadata.get('config', {}).get('feature_config', {})
    feature_extractor = BSPFeatureExtractor(feature_config)

    # 创建预测器
    predictor = Predictor(model, feature_extractor)

    # 测试不同阈值
    thresholds = [0.5, 0.6, 0.7, 0.8]
    results = {}

    for threshold in thresholds:
        print(f"\n{'='*80}")
        print(f"Testing threshold: {threshold}")
        print(f"{'='*80}")

        # 配置
        backtest_config = metadata.get('config', {}).get('backtest_config', {})
        backtest_config['score_threshold'] = threshold

        # 回测
        backtest = MLBacktest(predictor, backtest_config)
        metrics = backtest.run(test_chan_list)

        results[threshold] = metrics

    # 对比结果
    print("\n" + "=" * 80)
    print("Threshold Comparison Summary")
    print("=" * 80)
    print(f"\n{'Threshold':<12} {'Signals':<10} {'Win Rate':<12} {'Sharpe':<12} {'Calmar':<12} {'Total Return':<15}")
    print("-" * 80)

    for threshold in thresholds:
        metrics = results[threshold]
        print(f"{threshold:<12.2f} "
              f"{metrics.get('total_signals', 0):<10} "
              f"{metrics.get('win_rate', 0):<12.4f} "
              f"{metrics.get('sharpe_ratio', 0):<12.4f} "
              f"{metrics.get('calmar_ratio', 0):<12.4f} "
              f"{metrics.get('total_return', 0):<15.2%}")

    return results


def main():
    """主流程"""
    print("\n" + "=" * 80)
    print("Chan Theory ML Backtest Demo")
    print("=" * 80)

    # 1. 加载模型
    try:
        model, metadata = load_model(version=None)  # 加载最新版本
    except Exception as e:
        print(f"\nError loading model: {e}")
        print("Please run train_model_demo.py first to train a model.")
        return

    # 2. 准备回测数据
    test_chan_list = prepare_backtest_data()

    if len(test_chan_list) == 0:
        print("\nNo backtest data available. Exiting...")
        return

    # 3. 运行回测（使用默认阈值）
    metrics = run_backtest(model, test_chan_list, metadata)

    # 4. 比较不同阈值
    threshold_results = compare_thresholds(model, test_chan_list, metadata)

    print("\n" + "=" * 80)
    print("Backtest completed successfully!")
    print("=" * 80)

    # 找出最佳阈值
    best_threshold = max(threshold_results.keys(),
                        key=lambda t: threshold_results[t].get('sharpe_ratio', 0))

    print(f"\nBest threshold (by Sharpe ratio): {best_threshold}")
    print(f"  Sharpe ratio: {threshold_results[best_threshold].get('sharpe_ratio', 0):.4f}")
    print(f"  Win rate: {threshold_results[best_threshold].get('win_rate', 0):.4f}")
    print(f"  Total return: {threshold_results[best_threshold].get('total_return', 0):.2%}")


if __name__ == '__main__':
    main()
