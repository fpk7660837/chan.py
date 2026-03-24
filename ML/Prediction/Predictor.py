"""
预测器

为买卖点打分、排序，并提供股票池横截面排序接口。
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from Chan import CChan
from BuySellPoint.BS_Point import CBS_Point
from Common.CTime import CTime
from ..FeatureEngine.BSPFeatureExtractor import BSPFeatureExtractor
from ..Models.BaseModel import BaseModel


class Predictor:
    """预测器"""

    def __init__(self, model: BaseModel, feature_extractor: BSPFeatureExtractor = None):
        self.model = model
        self.feature_extractor = feature_extractor or BSPFeatureExtractor()

    def predict_single(self, bsp: CBS_Point) -> Dict[str, Any]:
        X = np.array([self.feature_extractor.get_feature_vector(bsp)], dtype=float)
        proba = self.model.predict_proba(X)
        score = float(proba[0, 1])
        label = int(self.model.predict(X)[0])
        return {'score': score, 'label': label}

    def predict_batch(self, bsp_list: List[CBS_Point]) -> List[Dict[str, Any]]:
        if len(bsp_list) == 0:
            return []

        X = np.array(
            [self.feature_extractor.get_feature_vector(bsp) for bsp in bsp_list],
            dtype=float,
        )
        proba = self.model.predict_proba(X)
        scores = proba[:, 1]
        labels = self.model.predict(X)
        return [
            {'score': float(score), 'label': int(label)}
            for score, label in zip(scores, labels)
        ]

    def collect_bsp(self, chan: CChan, direction: str = 'all') -> List[CBS_Point]:
        bsp_list = chan[0].bs_point_lst.getSortedBspList()
        if direction == 'all':
            return bsp_list
        if direction == 'buy':
            return [bsp for bsp in bsp_list if bsp.is_buy]
        if direction == 'sell':
            return [bsp for bsp in bsp_list if not bsp.is_buy]
        raise ValueError(f"Unsupported direction: {direction}")

    def score_bsp(self, chan: CChan, direction: str = 'all') -> List[Tuple[CBS_Point, float]]:
        bsp_list = self.collect_bsp(chan, direction=direction)
        results = self.predict_batch(bsp_list)
        return [(bsp, result['score']) for bsp, result in zip(bsp_list, results)]

    def rank_bsp(self, chan: CChan, top_k: int = 5, direction: str = 'all') -> List[Tuple[CBS_Point, float]]:
        bsp_scores = self.score_bsp(chan, direction=direction)
        bsp_scores.sort(key=lambda x: x[1], reverse=True)
        return bsp_scores[:top_k]

    def filter_bsp_by_threshold(
        self,
        chan: CChan,
        threshold: float = 0.7,
        direction: str = 'all',
    ) -> List[Tuple[CBS_Point, float]]:
        bsp_scores = self.score_bsp(chan, direction=direction)
        filtered = [(bsp, score) for bsp, score in bsp_scores if score >= threshold]
        filtered.sort(key=lambda x: x[1], reverse=True)
        return filtered

    def rank_stock_pool(
        self,
        chan_list: List[CChan],
        top_k: int = 10,
        direction: str = 'buy',
        score_threshold: Optional[float] = None,
        signal_lookback_bars: Optional[int] = None,
        as_of: Optional[CTime] = None,
    ) -> List[Dict[str, Any]]:
        stock_candidates = []

        for chan in chan_list:
            candidate = self._get_stock_candidate(
                chan=chan,
                direction=direction,
                signal_lookback_bars=signal_lookback_bars,
                as_of=as_of,
            )
            if candidate is None:
                continue
            if score_threshold is not None and candidate['score'] < score_threshold:
                continue
            stock_candidates.append(candidate)

        stock_candidates.sort(key=lambda item: item['score'], reverse=True)
        return stock_candidates[:top_k]

    def _get_stock_candidate(
        self,
        chan: CChan,
        direction: str,
        signal_lookback_bars: Optional[int],
        as_of: Optional[CTime],
    ) -> Optional[Dict[str, Any]]:
        bsp_scores = self.score_bsp(chan, direction=direction)
        if not bsp_scores:
            return None

        reference_klu = self._get_reference_klu(chan, as_of)
        if reference_klu is None:
            return None

        candidates = []
        for bsp, score in bsp_scores:
            if bsp.klu is None:
                continue
            if as_of is not None and bsp.klu.time.ts > as_of.ts:
                continue
            if signal_lookback_bars is not None and reference_klu.idx - bsp.klu.idx > signal_lookback_bars:
                continue
            candidates.append((bsp, score))

        if not candidates:
            return None

        # 同一只股票只取最近一个有效信号，再在股票池内做横截面排序。
        candidates.sort(key=lambda item: item[0].klu.idx, reverse=True)
        bsp, score = candidates[0]
        return {
            'code': chan.code,
            'score': float(score),
            'bsp': bsp,
            'signal_time': bsp.klu.time,
            'signal_idx': bsp.klu.idx,
            'price': float(bsp.klu.close),
        }

    @staticmethod
    def _get_reference_klu(chan: CChan, as_of: Optional[CTime]):
        if as_of is None:
            return Predictor._get_last_klu(chan)
        for klu in reversed(list(chan[0].klu_iter())):
            if klu.time.ts <= as_of.ts:
                return klu
        return None

    @staticmethod
    def _get_last_klu(chan: CChan):
        kl_list = chan[0]
        if len(kl_list) == 0:
            return None
        last_item = kl_list[-1]
        if hasattr(last_item, '__len__') and hasattr(last_item, '__getitem__'):
            if len(last_item) == 0:
                return None
            return last_item[-1]
        return last_item
