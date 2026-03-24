"""
买卖点特征提取器

从买卖点对象提取结构、技术指标和市场环境特征。
"""

from typing import Any, Dict, Optional

import numpy as np

from BuySellPoint.BS_Point import CBS_Point
from Common.CEnum import BSP_TYPE, DATA_FIELD


class BSPFeatureExtractor:
    """买卖点特征提取器"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.use_bi_features = self.config.get('use_bi_features', True)
        self.use_seg_features = self.config.get('use_seg_features', True)
        self.use_zs_features = self.config.get('use_zs_features', True)
        self.use_klu_features = self.config.get('use_klu_features', True)

    def extract(self, bsp: CBS_Point) -> Dict[str, float]:
        """提取买卖点的所有特征。"""
        features: Dict[str, float] = {}

        features.update(self._extract_bsp_basic_features(bsp))

        if self.use_bi_features and bsp.bi is not None:
            features.update(self._extract_bi_features(bsp.bi))

        seg = getattr(bsp.bi, 'parent_seg', None) if bsp.bi is not None else None
        if self.use_seg_features and seg is not None:
            features.update(self._extract_seg_features(seg))

        if self.use_zs_features and bsp.bi is not None:
            features.update(self._extract_zs_features(bsp, seg))

        if self.use_klu_features and bsp.klu is not None:
            features.update(self._extract_klu_features(bsp.klu))

        features.update(self._extract_market_env_features(bsp))

        # 保证输出字段稳定，避免首个样本字段不全导致训练集列错位。
        return {name: float(features.get(name, 0.0)) for name in self.get_feature_names()}

    def _extract_bsp_basic_features(self, bsp: CBS_Point) -> Dict[str, float]:
        type_mapping = {
            BSP_TYPE.T1: 1,
            BSP_TYPE.T1P: 2,
            BSP_TYPE.T2: 3,
            BSP_TYPE.T2S: 4,
            BSP_TYPE.T3A: 5,
            BSP_TYPE.T3B: 6,
        }
        primary_type = bsp.type[0] if isinstance(bsp.type, list) and bsp.type else bsp.type
        divergence_rate = 0.0
        if getattr(bsp, 'features', None) is not None:
            for key, value in bsp.features.items():
                if key == 'divergence_rate' and value is not None:
                    divergence_rate = float(value)
                    break

        return {
            'bsp_type': float(type_mapping.get(primary_type, 0)),
            'bsp_direction': 1.0 if bsp.is_buy else -1.0,
            'is_seg_bsp': 1.0 if bsp.is_segbsp else 0.0,
            'bsp_type_count': float(len(bsp.type) if isinstance(bsp.type, list) else 1),
            'bsp_divergence_rate': divergence_rate,
        }

    def _extract_bi_features(self, bi) -> Dict[str, float]:
        begin_val = bi.get_begin_val()
        amplitude = bi.amp()
        return {
            'bi_amp': amplitude / abs(begin_val) if begin_val else 0.0,
            'bi_macd_area': float(bi.Cal_MACD_area()),
            'bi_macd_peak': float(bi.Cal_MACD_peak()),
            'bi_macd_slope': float(bi.Cal_MACD_slope()),
            'bi_volume': float(bi.Cal_MACD_trade_metric(DATA_FIELD.FIELD_VOLUME)),
            'bi_klu_cnt': float(bi.get_klu_cnt()),
            'bi_dir': 1.0 if bi.is_up() else -1.0,
            'bi_rsi': float(bi.Cal_Rsi()),
        }

    def _extract_seg_features(self, seg) -> Dict[str, float]:
        return {
            'seg_amp': float(seg.cal_amp()),
            'seg_dir': 1.0 if seg.is_up() else -1.0,
            'seg_bi_cnt': float(seg.cal_bi_cnt()),
            'seg_slope': float(seg.cal_klu_slope()),
        }

    def _extract_zs_features(self, bsp: CBS_Point, seg) -> Dict[str, float]:
        if seg is None or not getattr(seg, 'zs_lst', None):
            return {
                'zs_high': 0.0,
                'zs_low': 0.0,
                'zs_amp': 0.0,
                'zs_cnt': 0.0,
                'price_to_zs': 0.5,
            }

        zs = seg.zs_lst[-1]
        zs_high = float(getattr(zs, 'high', 0.0) or 0.0)
        zs_low = float(getattr(zs, 'low', 0.0) or 0.0)
        current_price = float(bsp.klu.close) if bsp.klu is not None else 0.0
        if zs_high != zs_low:
            price_to_zs = (current_price - zs_low) / (zs_high - zs_low)
        else:
            price_to_zs = 0.5
        return {
            'zs_high': zs_high,
            'zs_low': zs_low,
            'zs_amp': (zs_high - zs_low) / zs_low if zs_low else 0.0,
            'zs_cnt': float(len(seg.zs_lst)),
            'price_to_zs': float(price_to_zs),
        }

    def _extract_klu_features(self, klu) -> Dict[str, float]:
        macd = getattr(klu, 'macd', None)
        rsi = getattr(klu, 'rsi', None)
        kdj = getattr(klu, 'kdj', None)
        trade_info = getattr(klu, 'trade_info', None)
        volume = 0.0
        if trade_info is not None:
            volume = float(trade_info.metric.get(DATA_FIELD.FIELD_VOLUME) or 0.0)

        return {
            'macd_dif': float(getattr(macd, 'DIF', 0.0)),
            'macd_dea': float(getattr(macd, 'DEA', 0.0)),
            'macd_macd': float(getattr(macd, 'macd', 0.0)),
            'kdj_k': float(getattr(kdj, 'k', 50.0)),
            'kdj_d': float(getattr(kdj, 'd', 50.0)),
            'kdj_j': float(getattr(kdj, 'j', 50.0)),
            'rsi': float(rsi if rsi is not None else 50.0),
            'volume': volume,
        }

    def _extract_market_env_features(self, bsp: CBS_Point) -> Dict[str, float]:
        prices = []
        current_klu = bsp.klu

        while current_klu is not None and len(prices) < 20:
            prices.append(float(current_klu.close))
            current_klu = getattr(current_klu, 'pre', None)

        if len(prices) < 2:
            return {
                'volatility': 0.0,
                'trend_strength': 0.0,
                'recent_return_5': 0.0,
                'recent_return_20': 0.0,
            }

        prices = list(reversed(prices))
        price_arr = np.array(prices, dtype=float)
        returns = np.diff(price_arr) / price_arr[:-1]

        x = np.arange(len(price_arr), dtype=float)
        slope = np.polyfit(x, price_arr, 1)[0] if len(price_arr) > 1 else 0.0

        recent_return_5 = 0.0
        if len(price_arr) >= 6 and price_arr[-6] != 0:
            recent_return_5 = (price_arr[-1] - price_arr[-6]) / price_arr[-6]

        recent_return_20 = 0.0
        if len(price_arr) >= 20 and price_arr[0] != 0:
            recent_return_20 = (price_arr[-1] - price_arr[0]) / price_arr[0]

        return {
            'volatility': float(np.std(returns)) if len(returns) > 0 else 0.0,
            'trend_strength': float(slope / np.mean(price_arr)) if np.mean(price_arr) else 0.0,
            'recent_return_5': float(recent_return_5),
            'recent_return_20': float(recent_return_20),
        }

    def get_feature_names(self) -> list:
        feature_names = [
            'bsp_type',
            'bsp_direction',
            'is_seg_bsp',
            'bsp_type_count',
            'bsp_divergence_rate',
        ]

        if self.use_bi_features:
            feature_names.extend([
                'bi_amp',
                'bi_macd_area',
                'bi_macd_peak',
                'bi_macd_slope',
                'bi_volume',
                'bi_klu_cnt',
                'bi_dir',
                'bi_rsi',
            ])

        if self.use_seg_features:
            feature_names.extend([
                'seg_amp',
                'seg_dir',
                'seg_bi_cnt',
                'seg_slope',
            ])

        if self.use_zs_features:
            feature_names.extend([
                'zs_high',
                'zs_low',
                'zs_amp',
                'zs_cnt',
                'price_to_zs',
            ])

        if self.use_klu_features:
            feature_names.extend([
                'macd_dif',
                'macd_dea',
                'macd_macd',
                'kdj_k',
                'kdj_d',
                'kdj_j',
                'rsi',
                'volume',
            ])

        feature_names.extend([
            'volatility',
            'trend_strength',
            'recent_return_5',
            'recent_return_20',
        ])
        return feature_names

    def get_feature_vector(self, bsp: CBS_Point) -> np.ndarray:
        """按固定顺序返回特征向量。"""
        features = self.extract(bsp)
        return np.array([features[name] for name in self.get_feature_names()], dtype=float)

    def get_feature_dict(self, bsp: CBS_Point) -> Dict[str, float]:
        """兼容旧接口。"""
        return self.extract(bsp)
