"""
买卖点特征提取器

从买卖点对象提取多层次特征，包括：
1. 买卖点基础特征
2. 笔级别特征
3. 线段特征
4. 中枢特征
5. K线技术指标特征
6. 市场环境特征
"""

import numpy as np
from typing import Dict, Any, Optional
from BuySellPoint.BS_Point import CBS_Point
from Common.CEnum import BSP_TYPE, TREND_TYPE
from Common.ChanException import CChanException


class BSPFeatureExtractor:
    """买卖点特征提取器"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化特征提取器

        Args:
            config: 特征配置字典，包含：
                - use_bi_features: 是否使用笔特征
                - use_seg_features: 是否使用线段特征
                - use_zs_features: 是否使用中枢特征
                - use_klu_features: 是否使用K线技术指标特征
        """
        self.config = config or {}
        self.use_bi_features = self.config.get('use_bi_features', True)
        self.use_seg_features = self.config.get('use_seg_features', True)
        self.use_zs_features = self.config.get('use_zs_features', True)
        self.use_klu_features = self.config.get('use_klu_features', True)

    def extract(self, bsp: CBS_Point) -> Dict[str, float]:
        """
        提取买卖点的所有特征

        Args:
            bsp: 买卖点对象

        Returns:
            特征字典，key为特征名，value为特征值
        """
        features = {}

        # 1. 买卖点基础特征
        features.update(self._extract_bsp_basic_features(bsp))

        # 2. 笔级别特征
        if self.use_bi_features and bsp.bi is not None:
            features.update(self._extract_bi_features(bsp.bi))

        # 3. 线段特征
        if self.use_seg_features and bsp.bi is not None and bsp.bi.get_parent_seg() is not None:
            features.update(self._extract_seg_features(bsp.bi.get_parent_seg()))

        # 4. 中枢特征
        if self.use_zs_features and bsp.bi is not None:
            features.update(self._extract_zs_features(bsp))

        # 5. K线技术指标特征
        if self.use_klu_features and bsp.klu is not None:
            features.update(self._extract_klu_features(bsp.klu))

        # 6. 市场环境特征
        features.update(self._extract_market_env_features(bsp))

        return features

    def _extract_bsp_basic_features(self, bsp: CBS_Point) -> Dict[str, float]:
        """提取买卖点基础特征"""
        features = {}

        # 买卖点类型编码（1类买=1, 1p=2, 2=3, 2s=4, 3a=5, 3b=6）
        type_mapping = {
            BSP_TYPE.T1: 1,
            BSP_TYPE.T1P: 2,
            BSP_TYPE.T2: 3,
            BSP_TYPE.T2S: 4,
            BSP_TYPE.T3A: 5,
            BSP_TYPE.T3B: 6,
        }
        features['bsp_type'] = type_mapping.get(bsp.type, 0)

        # 方向（买点=1，卖点=-1）
        features['bsp_direction'] = 1 if bsp.is_buy else -1

        # 是否为线段买卖点
        features['is_seg_bsp'] = 1 if bsp.is_segbsp else 0

        return features

    def _extract_bi_features(self, bi) -> Dict[str, float]:
        """提取笔级别特征"""
        features = {}

        try:
            # 振幅
            features['bi_amp'] = bi.amp()

            # MACD相关特征
            features['bi_macd_area'] = bi.Cal_MACD_area()
            features['bi_macd_peak'] = bi.Cal_MACD_peak()
            features['bi_macd_slope'] = bi.Cal_MACD_slope()

            # 成交量统计
            try:
                volume_metric = bi.Cal_MACD_trade_metric('volume')
                features['bi_volume'] = volume_metric if volume_metric is not None else 0.0
            except:
                features['bi_volume'] = 0.0

            # K线数量
            features['bi_klu_cnt'] = bi.get_klu_cnt()

            # 方向（向上=1，向下=-1）
            features['bi_dir'] = 1 if bi.dir == TREND_TYPE.UP else -1

            # RSI
            try:
                features['bi_rsi'] = bi.get_klu_list()[-1].close.metric_model_lst[0].rsi if hasattr(bi.get_klu_list()[-1].close, 'metric_model_lst') and len(bi.get_klu_list()[-1].close.metric_model_lst) > 0 else 50.0
            except:
                features['bi_rsi'] = 50.0

        except Exception as e:
            # 如果提取失败，使用默认值
            for key in ['bi_amp', 'bi_macd_area', 'bi_macd_peak', 'bi_macd_slope', 'bi_volume', 'bi_klu_cnt', 'bi_dir', 'bi_rsi']:
                if key not in features:
                    features[key] = 0.0

        return features

    def _extract_seg_features(self, seg) -> Dict[str, float]:
        """提取线段特征"""
        features = {}

        try:
            # 线段振幅
            begin_val = seg.start_bi.get_begin_val()
            end_val = seg.end_bi.get_end_val()
            features['seg_amp'] = abs(end_val - begin_val) / begin_val if begin_val != 0 else 0.0

            # 线段方向
            features['seg_dir'] = 1 if seg.dir == TREND_TYPE.UP else -1

            # 包含笔的数量
            features['seg_bi_cnt'] = len(seg.bi_list) if hasattr(seg, 'bi_list') else 0

            # 线段斜率（振幅/时间）
            if seg.end_bi.get_end_klu().idx > seg.start_bi.get_begin_klu().idx:
                time_span = seg.end_bi.get_end_klu().idx - seg.start_bi.get_begin_klu().idx
                features['seg_slope'] = features['seg_amp'] / time_span if time_span > 0 else 0.0
            else:
                features['seg_slope'] = 0.0

        except Exception as e:
            for key in ['seg_amp', 'seg_dir', 'seg_bi_cnt', 'seg_slope']:
                if key not in features:
                    features[key] = 0.0

        return features

    def _extract_zs_features(self, bsp: CBS_Point) -> Dict[str, float]:
        """提取中枢特征"""
        features = {}

        try:
            # 获取相关线段
            seg = bsp.bi.get_parent_seg() if bsp.bi is not None else None

            if seg is not None and hasattr(seg, 'zs_lst') and len(seg.zs_lst) > 0:
                # 获取最近的中枢
                zs = seg.zs_lst[-1]

                # 中枢区间
                features['zs_high'] = zs.high
                features['zs_low'] = zs.low
                features['zs_amp'] = (zs.high - zs.low) / zs.low if zs.low != 0 else 0.0

                # 中枢数量
                features['zs_cnt'] = len(seg.zs_lst)

                # 当前价格与中枢的相对位置
                current_price = bsp.klu.close
                if zs.high != zs.low:
                    features['price_to_zs'] = (current_price - zs.low) / (zs.high - zs.low)
                else:
                    features['price_to_zs'] = 0.5
            else:
                # 没有中枢，使用默认值
                features['zs_high'] = 0.0
                features['zs_low'] = 0.0
                features['zs_amp'] = 0.0
                features['zs_cnt'] = 0
                features['price_to_zs'] = 0.5

        except Exception as e:
            for key in ['zs_high', 'zs_low', 'zs_amp', 'zs_cnt', 'price_to_zs']:
                if key not in features:
                    features[key] = 0.0

        return features

    def _extract_klu_features(self, klu) -> Dict[str, float]:
        """提取K线技术指标特征"""
        features = {}

        try:
            # MACD指标
            if hasattr(klu.close, 'metric_model_lst') and len(klu.close.metric_model_lst) > 0:
                metric = klu.close.metric_model_lst[0]
                features['macd_dif'] = metric.dif if hasattr(metric, 'dif') else 0.0
                features['macd_dea'] = metric.dea if hasattr(metric, 'dea') else 0.0
                features['macd_macd'] = metric.macd if hasattr(metric, 'macd') else 0.0

                # KDJ指标
                features['kdj_k'] = metric.k if hasattr(metric, 'k') else 50.0
                features['kdj_d'] = metric.d if hasattr(metric, 'd') else 50.0
                features['kdj_j'] = metric.j if hasattr(metric, 'j') else 50.0

                # RSI指标
                features['rsi'] = metric.rsi if hasattr(metric, 'rsi') else 50.0
            else:
                # 默认值
                features['macd_dif'] = 0.0
                features['macd_dea'] = 0.0
                features['macd_macd'] = 0.0
                features['kdj_k'] = 50.0
                features['kdj_d'] = 50.0
                features['kdj_j'] = 50.0
                features['rsi'] = 50.0

            # 成交量
            features['volume'] = klu.trade_info.metric[0] if hasattr(klu, 'trade_info') and klu.trade_info is not None else 0.0

        except Exception as e:
            for key in ['macd_dif', 'macd_dea', 'macd_macd', 'kdj_k', 'kdj_d', 'kdj_j', 'rsi', 'volume']:
                if key not in features:
                    features[key] = 0.0 if 'macd' in key or key == 'volume' else 50.0

        return features

    def _extract_market_env_features(self, bsp: CBS_Point) -> Dict[str, float]:
        """提取市场环境特征"""
        features = {}

        try:
            # 前N根K线波动率
            n = 20
            klu_list = []

            # 获取买卖点所在K线及之前的K线
            if bsp.klu is not None:
                current_klu = bsp.klu
                klu_list.append(current_klu.close)

                # 尝试获取前面的K线
                if hasattr(current_klu, 'pre') and current_klu.pre is not None:
                    temp_klu = current_klu.pre
                    for _ in range(n - 1):
                        if temp_klu is not None:
                            klu_list.append(temp_klu.close)
                            temp_klu = temp_klu.pre if hasattr(temp_klu, 'pre') else None
                        else:
                            break

            if len(klu_list) > 1:
                prices = np.array(klu_list)
                returns = np.diff(prices) / prices[:-1]
                features['volatility'] = np.std(returns) if len(returns) > 0 else 0.0

                # 趋势强度（线性回归斜率）
                x = np.arange(len(prices))
                if len(x) > 1:
                    slope = np.polyfit(x, prices, 1)[0]
                    features['trend_strength'] = slope / np.mean(prices) if np.mean(prices) != 0 else 0.0
                else:
                    features['trend_strength'] = 0.0
            else:
                features['volatility'] = 0.0
                features['trend_strength'] = 0.0

        except Exception as e:
            features['volatility'] = 0.0
            features['trend_strength'] = 0.0

        return features

    def get_feature_names(self) -> list:
        """获取所有特征名称列表"""
        feature_names = ['bsp_type', 'bsp_direction', 'is_seg_bsp']

        if self.use_bi_features:
            feature_names.extend(['bi_amp', 'bi_macd_area', 'bi_macd_peak', 'bi_macd_slope',
                                'bi_volume', 'bi_klu_cnt', 'bi_dir', 'bi_rsi'])

        if self.use_seg_features:
            feature_names.extend(['seg_amp', 'seg_dir', 'seg_bi_cnt', 'seg_slope'])

        if self.use_zs_features:
            feature_names.extend(['zs_high', 'zs_low', 'zs_amp', 'zs_cnt', 'price_to_zs'])

        if self.use_klu_features:
            feature_names.extend(['macd_dif', 'macd_dea', 'macd_macd', 'kdj_k', 'kdj_d',
                                'kdj_j', 'rsi', 'volume'])

        feature_names.extend(['volatility', 'trend_strength'])

        return feature_names
