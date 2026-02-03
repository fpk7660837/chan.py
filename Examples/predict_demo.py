"""
预测示例

演示如何使用训练好的模型进行预测：
1. 加载模型
2. 准备新数据
3. 预测买卖点质量
4. 排序和筛选高质量买卖点
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
        print(f"  Save time: {metadata.get('save_time', 'Unknown')}")

    return model, metadata


def prepare_prediction_data(code='600519', begin_time='2024-01-01', end_time='2024-12-31'):
    """准备预测数据"""
    print("\n" + "=" * 80)
    print("Step 2: Preparing prediction data...")
    print("=" * 80)

    print(f"\nLoading data for {code}...")
    chan = CChan(
        code=code,
        begin_time=begin_time,
        end_time=end_time,
        data_src=DATA_SRC.AKSHARE,
        lv_list=[KL_TYPE.K_DAY]
    )

    print(f"Data loaded successfully!")
    return chan


def predict_bsp_quality(model, chan, metadata):
    """预测买卖点质量"""
    print("\n" + "=" * 80)
    print("Step 3: Predicting buy/sell point quality...")
    print("=" * 80)

    # 创建特征提取器
    feature_config = metadata.get('config', {}).get('feature_config', {})
    feature_extractor = BSPFeatureExtractor(feature_config)

    # 创建预测器
    predictor = Predictor(model, feature_extractor)

    # 获取所有买卖点数量
    total_bsp = 0
    if hasattr(chan, 'bs_point_lst'):
        for bsp_dict in chan.bs_point_lst.values():
            total_bsp += len(bsp_dict)

    print(f"\nTotal buy/sell points: {total_bsp}")

    # 预测并排序（买点）
    print("\n" + "-" * 80)
    print("Top 5 BUY points (highest quality):")
    print("-" * 80)

    top_buy_points = predictor.rank_bsp(chan, top_k=5, direction='buy')

    for i, (bsp, score) in enumerate(top_buy_points, 1):
        print(f"\n{i}. Buy Point:")
        print(f"   Time: {bsp.klu.time if hasattr(bsp.klu, 'time') else 'N/A'}")
        print(f"   Type: {bsp.type.value}")
        print(f"   Price: {bsp.klu.close:.2f}")
        print(f"   Quality Score: {score:.4f} ({score*100:.2f}%)")

    # 预测并排序（卖点）
    print("\n" + "-" * 80)
    print("Top 5 SELL points (highest quality):")
    print("-" * 80)

    top_sell_points = predictor.rank_bsp(chan, top_k=5, direction='sell')

    for i, (bsp, score) in enumerate(top_sell_points, 1):
        print(f"\n{i}. Sell Point:")
        print(f"   Time: {bsp.klu.time if hasattr(bsp.klu, 'time') else 'N/A'}")
        print(f"   Type: {bsp.type.value}")
        print(f"   Price: {bsp.klu.close:.2f}")
        print(f"   Quality Score: {score:.4f} ({score*100:.2f}%)")

    return top_buy_points, top_sell_points


def filter_by_threshold(model, chan, metadata, threshold=0.7):
    """根据阈值筛选买卖点"""
    print("\n" + "=" * 80)
    print(f"Step 4: Filtering buy/sell points (threshold={threshold})...")
    print("=" * 80)

    # 创建特征提取器
    feature_config = metadata.get('config', {}).get('feature_config', {})
    feature_extractor = BSPFeatureExtractor(feature_config)

    # 创建预测器
    predictor = Predictor(model, feature_extractor)

    # 筛选买点
    filtered_buy = predictor.filter_bsp_by_threshold(chan, threshold=threshold, direction='buy')
    print(f"\nFiltered BUY points (score >= {threshold}): {len(filtered_buy)}")

    for i, (bsp, score) in enumerate(filtered_buy[:3], 1):  # 只显示前3个
        print(f"  {i}. Time: {bsp.klu.time if hasattr(bsp.klu, 'time') else 'N/A'}, "
              f"Price: {bsp.klu.close:.2f}, Score: {score:.4f}")

    # 筛选卖点
    filtered_sell = predictor.filter_bsp_by_threshold(chan, threshold=threshold, direction='sell')
    print(f"\nFiltered SELL points (score >= {threshold}): {len(filtered_sell)}")

    for i, (bsp, score) in enumerate(filtered_sell[:3], 1):
        print(f"  {i}. Time: {bsp.klu.time if hasattr(bsp.klu, 'time') else 'N/A'}, "
              f"Price: {bsp.klu.close:.2f}, Score: {score:.4f}")

    return filtered_buy, filtered_sell


def main():
    """主流程"""
    print("\n" + "=" * 80)
    print("Chan Theory ML Prediction Demo")
    print("=" * 80)

    # 1. 加载模型
    try:
        model, metadata = load_model(version=None)  # 加载最新版本
    except Exception as e:
        print(f"\nError loading model: {e}")
        print("Please run train_model_demo.py first to train a model.")
        return

    # 2. 准备预测数据
    try:
        chan = prepare_prediction_data(code='600519', begin_time='2024-01-01', end_time='2024-12-31')
    except Exception as e:
        print(f"\nError loading prediction data: {e}")
        return

    # 3. 预测买卖点质量
    top_buy, top_sell = predict_bsp_quality(model, chan, metadata)

    # 4. 根据阈值筛选
    filtered_buy, filtered_sell = filter_by_threshold(model, chan, metadata, threshold=0.7)

    print("\n" + "=" * 80)
    print("Prediction completed successfully!")
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  Top buy points: {len(top_buy)}")
    print(f"  Top sell points: {len(top_sell)}")
    print(f"  Filtered buy points (score >= 0.7): {len(filtered_buy)}")
    print(f"  Filtered sell points (score >= 0.7): {len(filtered_sell)}")


if __name__ == '__main__':
    main()
