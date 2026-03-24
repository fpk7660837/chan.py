"""
股票池排序与横截面组合回测示例

演示流程：
1. 加载模型
2. 准备股票池数据
3. 输出当前股票池推荐列表
4. 运行周期调仓组合回测
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Chan import CChan
from Common.CEnum import DATA_SRC, KL_TYPE
from Config.MLConfig import MLConfig
from ML.Backtest.CrossSectionBacktest import CrossSectionBacktest
from ML.FeatureEngine.BSPFeatureExtractor import BSPFeatureExtractor
from ML.Prediction.Predictor import Predictor
from ML.Utils.ModelIO import ModelIO


def load_model(version=None):
    model_io = ModelIO('./models')
    model = model_io.load(version=version)
    metadata = model_io.load_metadata(version=version)
    return model, metadata


def prepare_stock_pool():
    codes = ['600000', '600016', '600036', '600519', '000001', '000333']
    chan_list = []

    for code in codes:
        try:
            print(f"Loading {code}...")
            chan = CChan(
                code=code,
                begin_time='2022-01-01',
                end_time='2024-12-31',
                data_src=DATA_SRC.AKSHARE,
                lv_list=[KL_TYPE.K_DAY],
            )
            chan_list.append(chan)
        except Exception as exc:
            print(f"  Skip {code}: {exc}")

    return chan_list


def show_latest_recommendations(predictor, chan_list):
    print("\n" + "=" * 80)
    print("Latest Stock Pool Ranking")
    print("=" * 80)

    ranked = predictor.rank_stock_pool(
        chan_list,
        top_k=5,
        direction='buy',
        score_threshold=0.6,
        signal_lookback_bars=20,
    )
    for idx, item in enumerate(ranked, 1):
        bsp = item['bsp']
        print(
            f"{idx}. {item['code']} | score={item['score']:.4f} | "
            f"time={bsp.klu.time} | price={bsp.klu.close:.2f} | type={bsp.type2str()}"
        )


def run_portfolio_backtest(predictor, metadata):
    print("\n" + "=" * 80)
    print("Cross-Section Portfolio Backtest")
    print("=" * 80)

    config = MLConfig()
    if metadata:
        saved_config = metadata.get('config', {})
        if 'portfolio_backtest_config' in saved_config:
            config.portfolio_backtest_config.update(saved_config['portfolio_backtest_config'])

    config.portfolio_backtest_config.update({
        'top_k': 3,
        'score_threshold': 0.55,
        'rebalance_bars': 5,
        'signal_lookback_bars': 20,
    })

    chan_list = prepare_stock_pool()
    if not chan_list:
        print("No stock data available")
        return

    backtest = CrossSectionBacktest(predictor, config.portfolio_backtest_config)
    metrics = backtest.run(chan_list)

    print(f"Periods: {int(metrics.get('periods', 0))}")
    print(f"Average positions: {metrics.get('avg_positions', 0.0):.2f}")


def main():
    try:
        model, metadata = load_model(version=None)
    except Exception as exc:
        print(f"Load model failed: {exc}")
        print("Please run Examples/train_model_demo.py first.")
        return

    feature_config = metadata.get('config', {}).get('feature_config', {}) if metadata else {}
    predictor = Predictor(model, BSPFeatureExtractor(feature_config))

    chan_list = prepare_stock_pool()
    if not chan_list:
        print("No stock data available")
        return

    show_latest_recommendations(predictor, chan_list)
    run_portfolio_backtest(predictor, metadata)


if __name__ == '__main__':
    main()
