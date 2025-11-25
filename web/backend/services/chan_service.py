"""
Chan calculation service
Wraps original chan.py code and exposes business logic
"""
from datetime import datetime
from typing import Dict, Any, List, Optional

# Import original chan.py modules
from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, DATA_FIELD, DATA_SRC, KL_TYPE
from Math.MACD import CMACD
from Math.KDJ import KDJ
from Math.RSI import RSI


class ChanService:
    """Service class for Chan theory calculations"""
    
    # Map string enums to original enums (only A-share support)
    DATA_SRC_MAP = {
        "BAO_STOCK": DATA_SRC.BAO_STOCK,
        "CSV": DATA_SRC.CSV
    }
    
    KL_TYPE_MAP = {
        "1m": KL_TYPE.K_1M,
        "5m": KL_TYPE.K_5M,
        "15m": KL_TYPE.K_15M,
        "30m": KL_TYPE.K_30M,
        "60m": KL_TYPE.K_60M,
        "day": KL_TYPE.K_DAY,
        "week": KL_TYPE.K_WEEK,
        "mon": KL_TYPE.K_MON
    }
    
    def calculate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate Chan theory indicators
        
        Args:
            params: Analysis parameters including stock code, time range, config, etc.
            
        Returns:
            Dict containing K-line data and Chan indicators
        """
        # Parse parameters
        code = params["code"]
        begin_time = params.get("begin_time", "2020-01-01")
        end_time = params.get("end_time")
        data_src = self.DATA_SRC_MAP.get(params["data_src"], DATA_SRC.BAO_STOCK)
        limit_kl_count = params.get("limit_kl_count")
        try:
            limit_kl_count = int(limit_kl_count)
            if limit_kl_count <= 0:
                limit_kl_count = None
            else:
                limit_kl_count = min(limit_kl_count, 5000)
        except (TypeError, ValueError):
            limit_kl_count = None
        
        print(f"📊 Received lv_list from frontend: {params['lv_list']}")
        lv_list = [self.KL_TYPE_MAP[lv] for lv in params["lv_list"]]
        print(f"📊 Mapped lv_list for CChan: {lv_list}")
        
        print(f"🔧 Creating CChanConfig...")
        chan_config_data = dict(params.get("chan_config") or {})

        # Backwards compatibility with legacy top-level parameters
        legacy_keys = [
            "bi_strict",
            "seg_algo",
            "zs_algo",
            "print_warning",
            "boll_n",
        ]
        for key in legacy_keys:
            if key in params and key not in chan_config_data:
                chan_config_data[key] = params[key]

        # Ensure nested defaults exist
        for array_key in ("mean_metrics", "trend_metrics"):
            if array_key in chan_config_data and chan_config_data[array_key] is not None:
                chan_config_data[array_key] = [int(v) for v in chan_config_data[array_key] if v is not None]
        if "skip_step" in chan_config_data and chan_config_data["skip_step"] is not None:
            chan_config_data["skip_step"] = int(chan_config_data["skip_step"])
        if "boll_n" in chan_config_data and chan_config_data["boll_n"] is not None:
            chan_config_data["boll_n"] = int(chan_config_data["boll_n"])
        if "macd" not in chan_config_data:
            chan_config_data["macd"] = {"fast": 12, "slow": 26, "signal": 9}
        else:
            macd_conf = chan_config_data["macd"] or {}
            chan_config_data["macd"] = {
                "fast": int(macd_conf.get("fast", 12)),
                "slow": int(macd_conf.get("slow", 26)),
                "signal": int(macd_conf.get("signal", 9)),
            }
        divergence_rate = chan_config_data.get("divergence_rate")
        if divergence_rate is None:
            chan_config_data["divergence_rate"] = float("inf")
        else:
            try:
                parsed = float(divergence_rate)
            except (TypeError, ValueError):
                parsed = float("inf")
            chan_config_data["divergence_rate"] = parsed if parsed > 0 else float("inf")

        if "demark" not in chan_config_data:
            chan_config_data["demark"] = {
                "demark_len": 9,
                "setup_bias": 4,
                "countdown_bias": 2,
                "max_countdown": 13,
                "tiaokong_st": True,
                "setup_cmp2close": True,
                "countdown_cmp2close": True,
            }
        else:
            demark_conf = chan_config_data["demark"] or {}
            chan_config_data["demark"] = {
                "demark_len": int(demark_conf.get("demark_len", 9)),
                "setup_bias": int(demark_conf.get("setup_bias", 4)),
                "countdown_bias": int(demark_conf.get("countdown_bias", 2)),
                "max_countdown": int(demark_conf.get("max_countdown", 13)),
                "tiaokong_st": bool(demark_conf.get("tiaokong_st", True)),
                "setup_cmp2close": bool(demark_conf.get("setup_cmp2close", True)),
                "countdown_cmp2close": bool(demark_conf.get("countdown_cmp2close", True)),
            }

        print(f"🔧 Config dict: {chan_config_data}")
        chan_config = CChanConfig(chan_config_data)
        
        print(f"🔧 Executing CChan calculation for {code}...")
        # Execute Chan calculation
        try:
            chan = CChan(
                code=code,
                begin_time=begin_time,
                end_time=end_time,
                data_src=data_src,
                lv_list=lv_list,
                config=chan_config,
                autype=AUTYPE.QFQ,
            )
            print(f"✅ CChan calculation complete")
        except Exception as e:
            print(f"❌ CChan calculation failed: {str(e)}")
            raise
        
        # Extract results
        kline_data = self._extract_kline_data(chan)

        result = {
            "code": code,
            "data_source": params["data_src"],
            "kline_data": kline_data,
            "meta": {
                "total_klines": len(kline_data),
                "begin_time": begin_time,
                "end_time": end_time,
            }
        }
        
        # Add optional indicators based on request
        if params.get("plot_bi", True):
            result["bi_list"] = self._extract_bi_list(chan)
            result["meta"]["bi_count"] = len(result["bi_list"])
            
        if params.get("plot_seg", True):
            result["seg_list"] = self._extract_seg_list(chan)
            result["meta"]["seg_count"] = len(result["seg_list"])
            
        if params.get("plot_zs", True):
            bi_zs_list = self._extract_zs_list(chan, source="bi")
            seg_zs_list = self._extract_zs_list(chan, source="seg")
            result["zs_list"] = bi_zs_list
            result["bi_zs_list"] = bi_zs_list
            result["meta"]["zs_count"] = len(bi_zs_list)
            result["meta"]["bi_zs_count"] = len(bi_zs_list)
            if seg_zs_list:
                result["seg_zs_list"] = seg_zs_list
                result["meta"]["seg_zs_count"] = len(seg_zs_list)
            
        if params.get("plot_bsp", True):
            result["bsp_list"] = self._extract_bsp_list(chan)
            result["meta"]["bsp_count"] = len(result["bsp_list"])
        
        # Add MACD indicator
        if params.get("plot_macd", True):
            macd_result = self._calculate_macd(chan)
            result["macd_data"] = macd_result
            print(f"📊 Added MACD data to result: {len(macd_result)} points")
        
        # Add MA indicator
        print(f"🔍 Checking MA: plot_ma={params.get('plot_ma', False)}")
        if params.get("plot_ma", False):
            ma_params = params.get("ma_params", [5, 10, 20, 60])
            if isinstance(ma_params, list):
                ma_params = [int(v) for v in ma_params if v is not None]
            print(f"📊 Extracting MA data with params: {ma_params}")
            ma_result = self._extract_ma_data(chan, ma_params)
            result["ma_data"] = ma_result
            print(f"📊 Added MA data to result: {ma_result.keys() if ma_result else 'Empty'}")
        else:
            print(f"⚠️  MA not requested in params")
        
        # Add BOLL indicator
        print(f"🔍 Checking BOLL: plot_boll={params.get('plot_boll', False)}")
        if params.get("plot_boll", False):
            print(f"📊 Extracting BOLL data...")
            boll_result = self._extract_boll_data(chan)
            result["boll_data"] = boll_result
            print(f"📊 Added BOLL data to result: {len(boll_result)} points")
        else:
            print(f"⚠️  BOLL not requested in params")
        
        # Add KDJ indicator
        print(f"🔍 Checking KDJ: plot_kdj={params.get('plot_kdj', False)}")
        if params.get("plot_kdj", False):
            kdj_period = params.get("kdj_period", 9)
            try:
                kdj_period = int(kdj_period)
            except (TypeError, ValueError):
                kdj_period = 9
            print(f"📊 Calculating KDJ data with period={kdj_period}...")
            kdj_result = self._calculate_kdj(chan, kdj_period)
            result["kdj_data"] = kdj_result
            print(f"📊 Added KDJ data to result: {len(kdj_result)} points")
        else:
            print(f"⚠️  KDJ not requested in params")
        
        # Add RSI indicator
        print(f"🔍 Checking RSI: plot_rsi={params.get('plot_rsi', False)}")
        if params.get("plot_rsi", False):
            rsi_period = params.get("rsi_period", 14)
            try:
                rsi_period = int(rsi_period)
            except (TypeError, ValueError):
                rsi_period = 14
            print(f"📊 Calculating RSI data with period={rsi_period}...")
            rsi_result = self._calculate_rsi(chan, rsi_period)
            result["rsi_data"] = rsi_result
            print(f"📊 Added RSI data to result: {len(rsi_result)} points")
        else:
            print(f"⚠️  RSI not requested in params")
        if limit_kl_count:
            original_count = len(result.get("kline_data", []))
            if original_count > limit_kl_count:
                trimmed_kl = result["kline_data"][-limit_kl_count:]
                cutoff_time_str = trimmed_kl[0].get("time")

                def _parse_time(value: Any):
                    if not value or not isinstance(value, str):
                        return None
                    try:
                        return datetime.fromisoformat(value)
                    except ValueError:
                        try:
                            return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
                        except ValueError:
                            return None

                cutoff_dt = _parse_time(cutoff_time_str)

                def _within_cutoff(value: Any) -> bool:
                    if value is None:
                        return False
                    if cutoff_dt:
                        parsed = _parse_time(value)
                        if parsed:
                            return parsed >= cutoff_dt
                    if cutoff_time_str:
                        if isinstance(value, str):
                            return value >= cutoff_time_str
                    return True

                result["kline_data"] = trimmed_kl
                result["meta"]["total_klines"] = len(trimmed_kl)
                if cutoff_time_str:
                    result["meta"]["begin_time"] = cutoff_time_str

                if "bi_list" in result:
                    result["bi_list"] = [bi for bi in result["bi_list"] if _within_cutoff(bi.get("end_time"))]
                    result["meta"]["bi_count"] = len(result["bi_list"])
                if "seg_list" in result:
                    result["seg_list"] = [seg for seg in result["seg_list"] if _within_cutoff(seg.get("end_time"))]
                    result["meta"]["seg_count"] = len(result["seg_list"])
                if "zs_list" in result:
                    result["zs_list"] = [zs for zs in result["zs_list"] if _within_cutoff(zs.get("end_time"))]
                    result["meta"]["zs_count"] = len(result["zs_list"])
                if "bsp_list" in result:
                    result["bsp_list"] = [bsp for bsp in result["bsp_list"] if _within_cutoff(bsp.get("time"))]
                    result["meta"]["bsp_count"] = len(result["bsp_list"])
                if "macd_data" in result:
                    result["macd_data"] = [item for item in result["macd_data"] if _within_cutoff(item.get("time"))]
                if "boll_data" in result:
                    result["boll_data"] = [item for item in result["boll_data"] if _within_cutoff(item.get("time"))]
                if "kdj_data" in result:
                    result["kdj_data"] = [item for item in result["kdj_data"] if _within_cutoff(item.get("time"))]
                if "rsi_data" in result:
                    result["rsi_data"] = [item for item in result["rsi_data"] if _within_cutoff(item.get("time"))]
                if "ma_data" in result and isinstance(result["ma_data"], dict):
                    result["ma_data"] = {
                        key: [item for item in values if _within_cutoff(item.get("time"))]
                        for key, values in result["ma_data"].items()
                    }
                print(f"✂️  Trimmed output from {original_count} to last {limit_kl_count} K-lines starting at {cutoff_time_str}")
            else:
                print(f"ℹ️  Requested limit_kl_count={limit_kl_count}, but only {original_count} K-lines available. Skipping trim.")
        
        print(f"📦 Final result keys: {list(result.keys())}")
        print(f"📦 Has macd_data in result: {'macd_data' in result}")
        
        return result
    
    def _time_to_str(self, time_obj) -> str:
        if time_obj is None:
            return ""
        if hasattr(time_obj, "to_str"):
            return time_obj.to_str()
        return str(time_obj)

    def _extract_kline_data(self, chan: CChan) -> List[Dict]:
        """Extract Chan-merged K-line data enriched with constituent K-line details"""
        klines: List[Dict] = []
        try:
            for idx, klc in enumerate(chan[0]):
                # klc is CKLine (merged K-line)
                # Get first and last klu for open/close
                first_klu = klc.lst[0] if klc.lst else None
                last_klu = klc.lst[-1] if klc.lst else None
                
                if first_klu and last_klu:
                    time_str = str(klc.time_begin)
                    
                    # Validate and convert price data
                    try:
                        open_price = float(first_klu.open)
                        high_price = float(klc.high)
                        low_price = float(klc.low)
                        close_price = float(last_klu.close)
                        volume = float(first_klu.trade_info.volume) if first_klu.trade_info and hasattr(first_klu.trade_info, 'volume') else 0
                        
                        # Validate all prices are positive numbers
                        if not all([open_price > 0, high_price > 0, low_price > 0, close_price > 0]):
                            print(f"⚠️  Warning: Invalid prices at idx {idx}: open={open_price}, high={high_price}, low={low_price}, close={close_price}")
                            continue
                        
                        # Validate high >= low
                        if high_price < low_price:
                            print(f"⚠️  Warning: high < low at idx {idx}: high={high_price}, low={low_price}, swapping")
                            high_price, low_price = low_price, high_price
                        
                        # Preserve Chan merged extremes exactly as computed on backend
                        actual_high = high_price
                        actual_low = low_price

                        # Clamp open/close into merged range so downstream charts stay consistent
                        if open_price > actual_high:
                            open_price = actual_high
                        elif open_price < actual_low:
                            open_price = actual_low
                        if close_price > actual_high:
                            close_price = actual_high
                        elif close_price < actual_low:
                            close_price = actual_low
                        
                        # Debug: log first K-line
                        if idx == 0:
                            print(f"🕐 Backend: First K-line time_begin = {time_str}")
                            print(f"🕐 Backend: time_begin type = {type(klc.time_begin)}")
                            print(f"💰 Backend: First K-line prices: open={open_price}, high={actual_high}, low={actual_low}, close={close_price}")
                        
                        end_time_raw = getattr(klc, "time_end", None)
                        if end_time_raw is None and last_klu is not None:
                            end_time_raw = getattr(last_klu, "time_end", None)
                        if end_time_raw is None and last_klu is not None:
                            end_time_raw = getattr(last_klu, "time", None)

                        sub_count = len(klc.lst) if hasattr(klc, "lst") else 1
                        child_klines: List[Dict] = []
                        if hasattr(klc, "lst"):
                            for child in klc.lst:
                                child_time_obj = getattr(child, "time", None)
                                child_end_obj = getattr(child, "time_end", None)
                                child_klines.append({
                                    "time": self._time_to_str(child_time_obj),
                                    "end_time": self._time_to_str(child_end_obj),
                                    "high": float(child.high),
                                    "low": float(child.low),
                                    "open": float(child.open),
                                    "close": float(child.close),
                                })

                        klines.append({
                            "time": time_str,
                            "end_time": self._time_to_str(end_time_raw) or time_str,
                            "open": open_price,
                            "high": actual_high,
                            "low": actual_low,
                            "close": close_price,
                            "volume": max(0, volume),  # Ensure volume is non-negative
                            "chan_sub_count": sub_count,
                            "chan_is_composed": bool(sub_count and sub_count > 1),
                            "chan_children": child_klines,
                        })
                    except (ValueError, TypeError) as e:
                        print(f"⚠️  Warning: Failed to convert prices at idx {idx}: {e}")
                        continue
        except Exception as e:
                print(f"⚠️  Warning extracting K-line data: {e}")
                import traceback
                traceback.print_exc()

        if klines:
            print(f"📊 Backend: Extracted {len(klines)} Chan K-lines, first time = {klines[0]['time']}, last time = {klines[-1]['time']}")
            print(f"📊 Backend: First Chan K-line sample: {klines[:1]}")

        return klines
    
    def _extract_bi_list(self, chan: CChan) -> List[Dict]:
        """Extract Bi (笔) list from CChan object"""
        bi_list = []
        if hasattr(chan[0], 'bi_list'):
            for bi in chan[0].bi_list:
                bi_list.append({
                    "idx": bi.idx,
                    "dir": self._normalize_dir(bi.dir),
                    "begin_time": str(bi.begin_klc.time_begin),
                    "end_time": str(bi.end_klc.time_end),
                    "begin_price": float(bi.get_begin_val()),
                    "end_price": float(bi.get_end_val()),
                })
        return bi_list
    
    def _extract_seg_list(self, chan: CChan) -> List[Dict]:
        """Extract Seg (线段) list from CChan object"""
        seg_list = []
        if hasattr(chan[0], 'seg_list'):
            for seg in chan[0].seg_list:
                seg_list.append({
                    "idx": seg.idx,
                    "dir": self._normalize_dir(seg.dir),
                    "begin_time": str(seg.start_bi.begin_klc.time_begin),
                    "end_time": str(seg.end_bi.end_klc.time_end),
                    "begin_price": float(seg.start_bi.get_begin_val()),
                    "end_price": float(seg.end_bi.get_end_val()),
                })
        return seg_list

    @staticmethod
    def _normalize_dir(dir_val) -> str:
        """将后端方向枚举/数字统一成前端可识别的 'up'/'down' 文本"""
        if isinstance(dir_val, str):
            lower = dir_val.lower()
            if lower in ("up", "down"):
                return lower
        name = getattr(dir_val, "name", None)
        if isinstance(name, str):
            lower = name.lower()
            if lower in ("up", "down"):
                return lower
        value = getattr(dir_val, "value", None)
        if isinstance(value, str):
            lower = value.lower()
            if lower in ("up", "down"):
                return lower
        if isinstance(value, (int, float)):
            if value == 1:
                return "up"
            if value == 2:
                return "down"
        return "unknown"
    
    @staticmethod
    def _format_time(value) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        for attr in ("time_begin", "time", "time_end"):
            attr_val = getattr(value, attr, None)
            if attr_val is not None:
                return str(attr_val)
        return None

    def _extract_zs_list(self, chan: CChan, source: str = "bi") -> List[Dict]:
        """Extract ZhongShu (中枢) list from CChan object.

        Args:
            chan: Calculated Chan instance.
            source: 'bi' to return pen-based ZhongShu, 'seg' for segment-based.

        Returns:
            A list of ZhongShu dictionaries sorted by their starting index.
        """
        zs_items: List[Dict[str, Any]] = []
        try:
            kl_list = chan[0]
        except Exception:
            return zs_items

        container = []
        component_unit = None

        if source == "bi" and hasattr(kl_list, "zs_list"):
            container = getattr(kl_list.zs_list, "zs_lst", [])
            component_unit = "笔"
        elif source == "seg" and hasattr(kl_list, "segzs_list"):
            container = getattr(kl_list.segzs_list, "zs_lst", [])
            component_unit = "段"

        raw_items = list(container) if container else []

        # Fallback to legacy collection to avoid returning empty lists when
        # direct containers are unavailable (older snapshots / partial calc).
        if not raw_items and hasattr(kl_list, "seg_list"):
            seen_ids = set()
            for seg in getattr(kl_list, "seg_list"):
                zs_lst = getattr(seg, "zs_lst", []) or []
                for zs in zs_lst:
                    if id(zs) in seen_ids:
                        continue
                    raw_items.append(zs)
                    seen_ids.add(id(zs))
            if component_unit is None:
                component_unit = "段" if source == "seg" else "笔"

        for zs in raw_items:
            begin_time = self._format_time(getattr(zs, "begin", None))
            end_time = self._format_time(getattr(zs, "end", None))
            if begin_time is None or end_time is None:
                continue

            try:
                high_val = float(getattr(zs, "high", None))
                low_val = float(getattr(zs, "low", None))
            except (TypeError, ValueError):
                continue

            component_lst = getattr(zs, "bi_lst", []) or []
            component_count = len(component_lst) if component_lst else None
            bi_count = component_count if source == "bi" else None

            entry: Dict[str, Any] = {
                "low": low_val,
                "high": high_val,
                "begin_time": begin_time,
                "end_time": end_time,
                "bi_count": bi_count,
                "component_count": component_count,
                "component_unit": component_unit,
                "source": source,
                "is_sure": bool(getattr(zs, "is_sure", False)),
            }

            try:
                entry["is_one_bi"] = bool(zs.is_one_bi_zs())
            except Exception:
                entry["is_one_bi"] = False

            begin_idx = getattr(getattr(zs, "begin", None), "idx", None)
            end_idx = getattr(getattr(zs, "end", None), "idx", None)
            if begin_idx is not None:
                entry["begin_index"] = begin_idx
            if end_idx is not None:
                entry["end_index"] = end_idx

            level = getattr(zs, "level", None)
            if level is None and hasattr(zs, "lv"):
                level = getattr(zs, "lv", None)
            if level is not None:
                if hasattr(level, "name"):
                    level = level.name
                elif hasattr(level, "value"):
                    level = level.value
                entry["level"] = level

            zs_items.append(entry)

        if not zs_items:
            return zs_items

        return sorted(zs_items, key=lambda item: item.get("begin_index", 0))
    
    def _extract_bsp_list(self, chan: CChan) -> List[Dict]:
        """Extract BuySellPoint (买卖点) list from CChan object"""
        bsp_list: List[Dict[str, Any]] = []
        bs_point_list = getattr(chan[0], 'bs_point_lst', None)
        if not bs_point_list:
            print("⚠️  CChan result has no bs_point_lst attribute")
            return bsp_list

        iter_fn = getattr(bs_point_list, 'bsp_iter', None)
        if not callable(iter_fn):
            print("⚠️  bs_point_lst has no bsp_iter method")
            return bsp_list

        buy_cnt = 0
        sell_cnt = 0

        def safe_float(value):
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        def extract_features(bsp):
            feature_info = {}
            features = getattr(bsp, 'features', None)
            if features and hasattr(features, 'items'):
                for key, val in features.items():
                    feature_info[key] = safe_float(val) if isinstance(val, (int, float)) else val
            return feature_info

        def describe_context(bsp):
            context: Dict[str, Any] = {}

            def safe_time(target, attrs):
                for attr in attrs:
                    value = getattr(target, attr, None)
                    if value is not None:
                        return str(value)
                return None

            bi = getattr(bsp, 'bi', None)
            if bi:
                context['bi_index'] = getattr(bi, 'idx', None)
                try:
                    if bi.is_down():
                        context['bi_direction'] = 'down'
                    elif bi.is_up():
                        context['bi_direction'] = 'up'
                except Exception:
                    pass
                try:
                    context['bi_amplitude'] = safe_float(bi.amp())
                except Exception:
                    pass
                try:
                    context['bi_high'] = safe_float(bi._high())
                    context['bi_low'] = safe_float(bi._low())
                    context['bi_mid'] = safe_float(bi._mid())
                except Exception:
                    pass
                try:
                    context['bi_begin_val'] = safe_float(bi.get_begin_val())
                    context['bi_end_val'] = safe_float(bi.get_end_val())
                except Exception:
                    pass
                try:
                    context['bi_klc_count'] = getattr(bi, 'get_klc_cnt', lambda: None)()
                except Exception:
                    pass
                try:
                    context['bi_klu_count'] = getattr(bi, 'get_klu_cnt', lambda: None)()
                except Exception:
                    pass
                begin_klc = getattr(bi, 'begin_klc', None)
                end_klc = getattr(bi, 'end_klc', None)
                if begin_klc:
                    context['bi_begin_time'] = safe_time(begin_klc, ['time_begin', 'time'])
                if end_klc:
                    context['bi_end_time'] = safe_time(end_klc, ['time_end', 'time'])
                prev_bi = getattr(bi, 'pre', None)
                next_bi = getattr(bi, 'next', None)
                if prev_bi:
                    context['bi_prev_idx'] = getattr(prev_bi, 'idx', None)
                    try:
                        if prev_bi.is_down():
                            context['bi_prev_dir'] = 'down'
                        elif prev_bi.is_up():
                            context['bi_prev_dir'] = 'up'
                    except Exception:
                        pass
                    try:
                        context['bi_prev_amplitude'] = safe_float(prev_bi.amp())
                    except Exception:
                        pass
                    try:
                        prev_end = safe_float(prev_bi.get_end_val())
                        curr_end = safe_float(bi.get_end_val())
                        prev_amp = safe_float(prev_bi.amp())
                        if prev_end is not None and curr_end is not None and prev_amp not in (None, 0):
                            context['retrace_rate_vs_prev'] = abs(curr_end - prev_end)/prev_amp
                    except Exception:
                        pass
                if next_bi:
                    context['bi_next_idx'] = getattr(next_bi, 'idx', None)
                    try:
                        if next_bi.is_down():
                            context['bi_next_dir'] = 'down'
                        elif next_bi.is_up():
                            context['bi_next_dir'] = 'up'
                    except Exception:
                        pass
                    try:
                        context['bi_next_amplitude'] = safe_float(next_bi.amp())
                    except Exception:
                        pass
                seg = getattr(bi, 'parent_seg', None)
                if seg:
                    context['segment_index'] = getattr(seg, 'idx', None)
                    try:
                        if seg.is_down():
                            context['segment_direction'] = 'down'
                        elif seg.is_up():
                            context['segment_direction'] = 'up'
                    except Exception:
                        pass
                    context['segment_bounds'] = {
                        "start_bi_idx": getattr(getattr(seg, 'start_bi', None), 'idx', None),
                        "end_bi_idx": getattr(getattr(seg, 'end_bi', None), 'idx', None),
                    }
                    try:
                        context['segment_bi_count'] = seg.cal_bi_cnt()
                    except Exception:
                        pass
                    try:
                        context['segment_amplitude'] = safe_float(seg.amp())
                    except Exception:
                        pass
                    zs_range = None
                    zs_details = []
                    bi_idx = context.get('bi_index')
                    for zs in getattr(seg, 'zs_lst', []) or []:
                        try:
                            if zs.is_one_bi_zs():
                                continue
                        except Exception:
                            continue
                        begin_idx = getattr(getattr(zs, 'begin_bi', None), 'idx', None)
                        end_idx = getattr(getattr(zs, 'end_bi', None), 'idx', None)
                        entry = {
                            "begin_bi_idx": begin_idx,
                            "end_bi_idx": end_idx,
                            "low": safe_float(getattr(zs, 'low', None)),
                            "high": safe_float(getattr(zs, 'high', None)),
                            "mid": safe_float(getattr(zs, 'mid', None)),
                            "peak_low": safe_float(getattr(zs, 'peak_low', None)),
                            "peak_high": safe_float(getattr(zs, 'peak_high', None)),
                            "bi_in_idx": getattr(getattr(zs, 'bi_in', None), 'idx', None),
                            "bi_out_idx": getattr(getattr(zs, 'bi_out', None), 'idx', None),
                        }
                        try:
                            if bi_idx is not None:
                                peak_flag, peak_rate = zs.out_bi_is_peak(bi_idx)
                                entry['is_peak_out'] = peak_flag
                                entry['peak_rate'] = safe_float(peak_rate)
                        except Exception:
                            pass
                        zs_details.append(entry)
                        if zs_range is None:
                            zs_range = {
                                "begin_bi_idx": begin_idx,
                                "end_bi_idx": end_idx,
                            }
                    if zs_range:
                        context['segment_zs_range'] = zs_range
                        context['pen_zs_range'] = zs_range
                    if zs_details:
                        context['segment_zs_list'] = zs_details
                        context['pen_zs_list'] = zs_details
                        context['segment_multi_zs'] = len(zs_details)
                        context['pen_multi_zs'] = len(zs_details)
                        active_zs = None
                        if bi_idx is not None:
                            for candidate in reversed(zs_details):
                                begin_idx = candidate.get('begin_bi_idx')
                                if begin_idx is None:
                                    continue
                                if begin_idx <= bi_idx:
                                    active_zs = candidate
                                    break
                        active_zs = active_zs or zs_details[-1]
                        context['active_zs'] = active_zs
                        context['active_pen_zs'] = active_zs
            relate = getattr(bsp, 'relate_bsp1', None)
            if relate and getattr(relate, 'bi', None):
                context['relate_bsp1_bi_idx'] = getattr(relate.bi, 'idx', None)
            return context

        def direction_label(direction):
            if direction == 'down':
                return '向下'
            if direction == 'up':
                return '向上'
            return '未知方向'

        def normalize_bsp_type_label(raw_value: Optional[str]) -> str:
            if not raw_value:
                return ''
            value = raw_value.strip().lower()
            mapping = {
                '1': 't1',
                't1': 't1',
                '1p': 't1p',
                't1p': 't1p',
                '2': 't2',
                't2': 't2',
                '2s': 't2s',
                't2s': 't2s',
                '3a': 't3a',
                't3a': 't3a',
                '3b': 't3b',
                't3b': 't3b',
            }
            return mapping.get(value, value)

        def build_reason_segments(bsp, feature_info, context):
            type_label = bsp.type2str() if hasattr(bsp, 'type2str') else ''
            types = [normalize_bsp_type_label(item) for item in type_label.split(',') if item.strip()]
            details: List[str] = []
            bi_idx = context.get('bi_index')
            bi_label = f"笔{bi_idx}" if bi_idx is not None else "当前笔"
            prev_idx = context.get('bi_prev_idx')
            prev_label = f"笔{prev_idx}" if prev_idx is not None else "上一笔"
            relate_idx = context.get('relate_bsp1_bi_idx')
            relate_label = f"笔{relate_idx}" if relate_idx is not None else "--"
            active_zs = context.get('active_pen_zs') or context.get('active_zs')
            zs_range = context.get('pen_zs_range') or context.get('segment_zs_range')
            zs_count = context.get('pen_multi_zs') or context.get('segment_multi_zs')

            def fmt_price(value, digits=2):
                val = safe_float(value)
                if val is None:
                    return "--"
                return f"{val:.{digits}f}"

            def fmt_ratio(value):
                val = safe_float(value)
                if val is None:
                    return "--"
                return f"{val:.4f}"

            def describe_zs_label():
                if active_zs:
                    begin_idx = active_zs.get('begin_bi_idx')
                    end_idx = active_zs.get('end_bi_idx')
                elif zs_range:
                    begin_idx = zs_range.get('begin_bi_idx')
                    end_idx = zs_range.get('end_bi_idx')
                else:
                    begin_idx = end_idx = None
                if begin_idx is None and end_idx is None:
                    return "笔中枢"
                return f"笔中枢(笔{begin_idx if begin_idx is not None else '-'}~笔{end_idx if end_idx is not None else '-'})"

            def append_zs_detail():
                if not active_zs:
                    return
                label = describe_zs_label()
                details.append(
                    f"{label}区间[{fmt_price(active_zs.get('low')), fmt_price(active_zs.get('high'))}]，峰值[{fmt_price(active_zs.get('peak_low')), fmt_price(active_zs.get('peak_high'))}]"
                )
                bi_in_idx = active_zs.get('bi_in_idx')
                bi_out_idx = active_zs.get('bi_out_idx')
                if bi_in_idx is not None or bi_out_idx is not None:
                    details.append(f"{label}进出笔：{bi_in_idx if bi_in_idx is not None else '--'} → {bi_out_idx if bi_out_idx is not None else '--'}")

            def append_bi_detail():
                bi_amp = context.get('bi_amplitude')
                bi_low = context.get('bi_low')
                bi_high = context.get('bi_high')
                if bi_amp is not None or bi_low is not None or bi_high is not None:
                    details.append(f"{bi_label}振幅≈{fmt_price(bi_amp, 4)}，价格区间[{fmt_price(bi_low)}, {fmt_price(bi_high)}]")
                seg_amp = context.get('segment_amplitude')
                if seg_amp is not None:
                    details.append(f"相关笔序整体振幅≈{fmt_price(seg_amp, 4)}")

            zs_label = describe_zs_label()

            for tp in types:
                if tp == 't1':
                    detail = f"T1：{bi_label}脱离{zs_label}"
                    if zs_count:
                        detail += f"，该区域包含{zs_count}个多笔中枢"
                    div_rate = feature_info.get('divergence_rate')
                    if div_rate is not None:
                        in_idx = active_zs.get('bi_in_idx') if active_zs else None
                        detail += f"，与进中枢笔{in_idx if in_idx is not None else '--'}相比背驰≈{fmt_ratio(div_rate)}"
                    details.append(detail)
                    append_zs_detail()
                    append_bi_detail()
                elif tp == 't1p':
                    detail = f"T1P：同向笔{prev_label} → {bi_label}盘整背驰"
                    div_rate = feature_info.get('divergence_rate')
                    if div_rate is not None:
                        detail += f"，背驰率≈{fmt_ratio(div_rate)}"
                    details.append(detail)
                    prev_amp = context.get('bi_prev_amplitude')
                    curr_amp = context.get('bi_amplitude')
                    if prev_amp not in (None, 0) and curr_amp is not None:
                        ratio = curr_amp / prev_amp
                        details.append(f"振幅比值≈{fmt_ratio(ratio)}（当前/上一同向笔）")
                    append_bi_detail()
                elif tp == 't2':
                    break_label = f"笔{prev_idx}" if prev_idx is not None else "突破笔"
                    if relate_idx is None:
                        detail = f"T2：未找到关联的T1 笔，当前{break_label}后{bi_label}回抽确认"
                    else:
                        detail = f"T2：以T1 {relate_label}为基准，{break_label}突破后{bi_label}回抽确认"
                    if zs_label:
                        detail += f"，回抽保持在{zs_label}"
                    details.append(detail)
                    retrace_rate = feature_info.get('retrace_rate')
                    if retrace_rate is None:
                        retrace_rate = context.get('retrace_rate_vs_prev')
                    if retrace_rate is not None:
                        details.append(f"回撤比例≈{fmt_ratio(retrace_rate)}，需小于配置阈值")
                    if relate_idx is None:
                        details.append("提示：未找到对应 T1 笔，可能被当前过滤或截断隐藏。")
                    append_bi_detail()
                elif tp == 't2s':
                    level_offset = feature_info.get('level_offset')
                    if relate_idx is None:
                        detail = "T2S：未找到关联的T1 笔，继续沿当前类二结构扩展共振"
                    else:
                        detail = f"T2S：沿T2结构继续共振，参考T1 {relate_label}"
                    if level_offset is not None:
                        detail += f"，扩展层级={level_offset}"
                    if zs_label:
                        detail += f"，仍围绕{zs_label}"
                    details.append(detail)
                    if relate_idx is None:
                        details.append("提示：未找到对应 T1 笔，可能被当前过滤或截断隐藏。")
                    append_bi_detail()
                elif tp == 't3a':
                    detail = f"T3A：后续笔中枢{zs_label}被{bi_label}突破，延续趋势"
                    details.append(detail)
                    append_zs_detail()
                    append_bi_detail()
                elif tp == 't3b':
                    detail = f"T3B：{bi_label}回抽到{zs_label}并离开，完成第三类确认"
                    details.append(detail)
                    append_zs_detail()
                    append_bi_detail()

            if not details:
                dir_label = '买' if getattr(bsp, 'is_buy', False) else '卖'
                detail = f"{dir_label}{type_label}".strip()
                div_rate = feature_info.get('divergence_rate')
                if div_rate is not None:
                    detail += f" 背驰率≈{fmt_ratio(div_rate)}"
                retrace_rate = feature_info.get('retrace_rate')
                if retrace_rate is not None:
                    detail += f" 回撤比例≈{fmt_ratio(retrace_rate)}"
                if detail:
                    details.append(detail)
            return details

        def build_reason_bundle(bsp, feature_info, relate_info):
            context = describe_context(bsp)
            if relate_info and relate_info.get('relate_bsp1_bi_idx') is not None:
                context.setdefault('relate_bsp1_bi_idx', relate_info['relate_bsp1_bi_idx'])
            detail_list = build_reason_segments(bsp, feature_info, context)
            text = '；'.join(detail_list) if detail_list else None
            return text, detail_list, context

        for bsp in iter_fn():
            klu = getattr(bsp, 'klu', None)
            if not klu:
                continue
            if hasattr(klu, 'time_begin'):
                time_str = str(klu.time_begin)
            elif hasattr(klu, 'time'):
                time_str = str(klu.time)
            else:
                time_str = ''

            price = getattr(klu, 'close', None)
            if price is None and hasattr(klu, 'price'):
                price = getattr(klu, 'price')

            try:
                price_val = float(price) if price is not None else None
            except (TypeError, ValueError):
                price_val = None

            bi = getattr(bsp, 'bi', None)
            bi_info = {}
            if bi:
                bi_info = {
                    "index": getattr(bi, 'idx', None),
                    "direction": "up" if (hasattr(bi, 'is_up') and bi.is_up()) else ("down" if hasattr(bi, 'is_down') and bi.is_down() else None),
                    "seg_index": getattr(bi, 'seg_idx', None),
                    "begin_time": str(getattr(getattr(bi, 'begin_klc', None), 'time_begin', '')) if getattr(bi, 'begin_klc', None) else None,
                    "end_time": str(getattr(getattr(bi, 'end_klc', None), 'time_end', '')) if getattr(bi, 'end_klc', None) else None,
                }

            relate_info = None
            if getattr(bsp, 'relate_bsp1', None):
                relate = bsp.relate_bsp1
                relate_info = {
                    "type": relate.type2str() if hasattr(relate, 'type2str') else None,
                    "time": str(getattr(getattr(relate, 'klu', None), 'time_begin', '')),
                    "price": safe_float(getattr(getattr(relate, 'klu', None), 'close', None)),
                    "relate_bsp1_bi_idx": getattr(getattr(relate, 'bi', None), 'idx', None),
                }

            feature_info = extract_features(bsp)
            reason_text, reason_details, context_info = build_reason_bundle(bsp, feature_info, relate_info)

            bsp_list.append({
                "is_buy": bool(getattr(bsp, 'is_buy', False)),
                "type": bsp.type2str() if hasattr(bsp, 'type2str') else None,
                "time": time_str,
                "price": price_val,
                "bi": bi_info,
                "relate_bsp1": relate_info,
                "features": feature_info or None,
                "reason": reason_text or None,
                "reason_details": reason_details or None,
                "context": context_info or None,
            })

            if getattr(bsp, 'is_buy', False):
                buy_cnt += 1
            else:
                sell_cnt += 1

        print(f"📌 Extracted BSP list: total={len(bsp_list)} buy={buy_cnt} sell={sell_cnt}")
        if len(bsp_list) > 0:
            sample = bsp_list[:3]
            print(f"   Sample BSP entries: {sample}")

        return bsp_list
    
    def _calculate_macd(self, chan: CChan, fastperiod=12, slowperiod=26, signalperiod=9) -> List[Dict]:
        """Calculate MACD indicator from K-line data"""
        print(f"📊 Starting MACD calculation...")
        macd_calculator = CMACD(fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
        macd_data = []
        
        try:
            kline_count = 0
            for klc in chan[0]:
                kline_count += 1
                # Use close price of last klu in the merged K-line
                last_klu = klc.lst[-1] if klc.lst else None
                if last_klu:
                    macd_item = macd_calculator.add(float(last_klu.close))
                    
                    # Get time from the merged K-line
                    time_str = str(klc.time_begin)
                    
                    macd_data.append({
                        "time": time_str,
                        "dif": float(macd_item.DIF),
                        "dea": float(macd_item.DEA),
                        "macd": float(macd_item.macd),  # MACD histogram = 2*(DIF-DEA)
                    })
            
            print(f"✅ MACD calculation complete: {len(macd_data)} points from {kline_count} K-lines")
            if macd_data:
                print(f"   First MACD: {macd_data[0]}")
                print(f"   Last MACD: {macd_data[-1]}")
        except Exception as e:
            print(f"⚠️  Warning calculating MACD: {e}")
            import traceback
            traceback.print_exc()
        
        return macd_data
    
    def _extract_ma_data(self, chan: CChan, ma_params: List[int] = [5, 10, 20, 60]) -> Dict[str, List[Dict]]:
        """
        Extract Moving Average data from K-line
        
        Args:
            chan: CChan object
            ma_params: List of MA periods (e.g., [5, 10, 20, 60])
            
        Returns:
            Dict with MA data for each period
        """
        print(f"📊 Starting MA calculation with periods: {ma_params}...")
        ma_data = {f"ma{period}": [] for period in ma_params}
        
        try:
            # Use simple moving average calculation for each period
            from collections import deque
            ma_queues = {period: deque(maxlen=period) for period in ma_params}
            
            for klc in chan[0]:
                last_klu = klc.lst[-1] if klc.lst else None
                if last_klu:
                    close_price = float(last_klu.close)
                    time_str = str(klc.time_begin)
                    
                    for period in ma_params:
                        ma_queues[period].append(close_price)
                        
                        # Calculate MA (only if we have enough data points)
                        if len(ma_queues[period]) >= period:
                            ma_value = sum(ma_queues[period]) / len(ma_queues[period])
                        else:
                            # For initial periods, use available data
                            ma_value = sum(ma_queues[period]) / len(ma_queues[period]) if len(ma_queues[period]) > 0 else close_price
                        
                        ma_data[f"ma{period}"].append({
                            "time": time_str,
                            "value": float(ma_value)
                        })
            
            # Build status string
            ma_status = ', '.join([f'MA{p}={len(ma_data[f"ma{p}"])} points' for p in ma_params])
            print(f"✅ MA calculation complete: {ma_status}")
        except Exception as e:
            print(f"⚠️  Warning calculating MA: {e}")
            import traceback
            traceback.print_exc()
        
        return ma_data
    
    def _extract_boll_data(self, chan: CChan) -> List[Dict]:
        """
        Extract Bollinger Bands data from K-line
        BOLL data should already be calculated if boll_n is set in CChanConfig
        
        Returns:
            List of dicts with time, upper, middle, lower bands
        """
        print(f"📊 Starting BOLL extraction...")
        boll_data = []
        
        try:
            for klc in chan[0]:
                last_klu = klc.lst[-1] if klc.lst else None
                if last_klu and hasattr(last_klu, 'boll') and last_klu.boll:
                    time_str = str(klc.time_begin)
                    
                    boll_data.append({
                        "time": time_str,
                        "upper": float(last_klu.boll.UP),
                        "middle": float(last_klu.boll.MID),
                        "lower": float(last_klu.boll.DOWN)
                    })
            
            print(f"✅ BOLL extraction complete: {len(boll_data)} points")
            if boll_data:
                print(f"   First BOLL: {boll_data[0]}")
                print(f"   Last BOLL: {boll_data[-1]}")
        except Exception as e:
            print(f"⚠️  Warning extracting BOLL: {e}")
            print(f"   Make sure boll_n is configured in CChanConfig")
            import traceback
            traceback.print_exc()
        
        return boll_data
    
    def _calculate_kdj(self, chan: CChan, period: int = 9) -> List[Dict]:
        """
        Calculate KDJ indicator from K-line data
        
        Args:
            chan: CChan object
            period: KDJ calculation period (default: 9)
            
        Returns:
            List of dicts with time, k, d, j values
        """
        print(f"📊 Starting KDJ calculation with period={period}...")
        kdj_calculator = KDJ(period=period)
        kdj_data = []
        
        try:
            for klc in chan[0]:
                # Get high, low, close from the merged K-line
                last_klu = klc.lst[-1] if klc.lst else None
                if last_klu:
                    high = float(klc.high)
                    low = float(klc.low)
                    close = float(last_klu.close)
                    
                    kdj_item = kdj_calculator.add(high, low, close)
                    time_str = str(klc.time_begin)
                    
                    kdj_data.append({
                        "time": time_str,
                        "k": float(kdj_item.k),
                        "d": float(kdj_item.d),
                        "j": float(kdj_item.j),
                    })
            
            print(f"✅ KDJ calculation complete: {len(kdj_data)} points")
            if kdj_data:
                print(f"   First KDJ: {kdj_data[0]}")
                print(f"   Last KDJ: {kdj_data[-1]}")
        except Exception as e:
            print(f"⚠️  Warning calculating KDJ: {e}")
            import traceback
            traceback.print_exc()
        
        return kdj_data
    
    def _calculate_rsi(self, chan: CChan, period: int = 14) -> List[Dict]:
        """
        Calculate RSI indicator from K-line data
        
        Args:
            chan: CChan object
            period: RSI calculation period (default: 14)
            
        Returns:
            List of dicts with time, rsi values
        """
        print(f"📊 Starting RSI calculation with period={period}...")
        rsi_calculator = RSI(period=period)
        rsi_data = []
        
        try:
            for klc in chan[0]:
                last_klu = klc.lst[-1] if klc.lst else None
                if last_klu:
                    close = float(last_klu.close)
                    rsi_value = rsi_calculator.add(close)
                    time_str = str(klc.time_begin)
                    
                    rsi_data.append({
                        "time": time_str,
                        "rsi": float(rsi_value),
                    })
            
            print(f"✅ RSI calculation complete: {len(rsi_data)} points")
            if rsi_data:
                print(f"   First RSI: {rsi_data[0]}")
                print(f"   Last RSI: {rsi_data[-1]}")
        except Exception as e:
            print(f"⚠️  Warning calculating RSI: {e}")
            import traceback
            traceback.print_exc()
        
        return rsi_data
