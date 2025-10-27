"""
Chan calculation service
Wraps original chan.py code and exposes business logic
"""
from typing import Dict, Any, List

# Import original chan.py modules
from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE
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
        
        print(f"📊 Received lv_list from frontend: {params['lv_list']}")
        lv_list = [self.KL_TYPE_MAP[lv] for lv in params["lv_list"]]
        print(f"📊 Mapped lv_list for CChan: {lv_list}")
        
        print(f"🔧 Creating CChanConfig...")
        # Build Chan config
        config_dict = {
            "bi_strict": params.get("bi_strict", True),
            "seg_algo": params.get("seg_algo", "chan"),
            "zs_algo": params.get("zs_algo", "normal"),
            "print_warning": False,  # Disable warnings for cleaner API response
            "boll_n": 20,  # Enable BOLL calculation with 20 periods
        }
        
        print(f"🔧 Config dict: {config_dict}")
        chan_config = CChanConfig(config_dict)
        
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
        result = {
            "code": code,
            "data_source": params["data_src"],
            "kline_data": self._extract_kline_data(chan),
            "meta": {
                "total_klines": len(chan[0]),
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
            result["zs_list"] = self._extract_zs_list(chan)
            result["meta"]["zs_count"] = len(result["zs_list"])
            
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
            print(f"📊 Calculating RSI data with period={rsi_period}...")
            rsi_result = self._calculate_rsi(chan, rsi_period)
            result["rsi_data"] = rsi_result
            print(f"📊 Added RSI data to result: {len(rsi_result)} points")
        else:
            print(f"⚠️  RSI not requested in params")
        
        print(f"📦 Final result keys: {list(result.keys())}")
        print(f"📦 Has macd_data in result: {'macd_data' in result}")
        
        return result
    
    def _extract_kline_data(self, chan: CChan) -> List[Dict]:
        """Extract K-line data from CChan object"""
        klines = []
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
                        
                        # Ensure high is the maximum and low is the minimum
                        actual_high = max(high_price, low_price, open_price, close_price)
                        actual_low = min(high_price, low_price, open_price, close_price)
                        
                        # Debug: log first K-line
                        if idx == 0:
                            print(f"🕐 Backend: First K-line time_begin = {time_str}")
                            print(f"🕐 Backend: time_begin type = {type(klc.time_begin)}")
                            print(f"💰 Backend: First K-line prices: open={open_price}, high={actual_high}, low={actual_low}, close={close_price}")
                        
                        klines.append({
                            "time": time_str,
                            "open": open_price,
                            "high": actual_high,
                            "low": actual_low,
                            "close": close_price,
                            "volume": max(0, volume),  # Ensure volume is non-negative
                        })
                    except (ValueError, TypeError) as e:
                        print(f"⚠️  Warning: Failed to convert prices at idx {idx}: {e}")
                        continue
        except Exception as e:
            print(f"⚠️  Warning extracting K-line data: {e}")
            import traceback
            traceback.print_exc()
        
        if klines:
            print(f"📊 Backend: Extracted {len(klines)} K-lines, first time = {klines[0]['time']}, last time = {klines[-1]['time']}")
            # Log sample data
            print(f"📊 Backend: First 3 K-lines: {klines[:3]}")
        
        return klines
    
    def _extract_bi_list(self, chan: CChan) -> List[Dict]:
        """Extract Bi (笔) list from CChan object"""
        bi_list = []
        if hasattr(chan[0], 'bi_list'):
            for bi in chan[0].bi_list:
                bi_list.append({
                    "idx": bi.idx,
                    "dir": bi.dir.value,
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
                    "dir": seg.dir.value,
                    "begin_time": str(seg.start_bi.begin_klc.time_begin),
                    "end_time": str(seg.end_bi.end_klc.time_end),
                    "begin_price": float(seg.start_bi.get_begin_val()),
                    "end_price": float(seg.end_bi.get_end_val()),
                })
        return seg_list
    
    def _extract_zs_list(self, chan: CChan) -> List[Dict]:
        """Extract ZhongShu (中枢) list from CChan object"""
        zs_list = []
        if hasattr(chan[0], 'seg_list'):
            for seg in chan[0].seg_list:
                if hasattr(seg, 'zs_lst'):
                    for zs in seg.zs_lst:
                        # zs.begin and zs.end could be CKLine or CKLine_Unit
                        begin_time = str(zs.begin.time_begin) if hasattr(zs.begin, 'time_begin') else str(zs.begin.time)
                        end_time = str(zs.end.time_end) if hasattr(zs.end, 'time_end') else str(zs.end.time)
                        
                        level = getattr(zs, "level", None)
                        if hasattr(level, "name"):
                            level = level.name
                        elif hasattr(level, "value"):
                            level = level.value
                        if level is None and hasattr(zs, "lv"):
                            level = getattr(zs, "lv", None)
                            if hasattr(level, "name"):
                                level = level.name
                        zs_list.append({
                            "low": float(zs.low),
                            "high": float(zs.high),
                            "begin_time": begin_time,
                            "end_time": end_time,
                            "bi_count": len(zs.bi_lst),
                            "level": level,
                        })
        return zs_list
    
    def _extract_bsp_list(self, chan: CChan) -> List[Dict]:
        """Extract BuySellPoint (买卖点) list from CChan object"""
        bsp_list = []
        if hasattr(chan[0], 'bs_point_lst') and hasattr(chan[0].bs_point_lst, 'lst'):
            for bsp in chan[0].bs_point_lst.lst:
                # bsp.klu could be CKLine or CKLine_Unit
                time_str = str(bsp.klu.time_begin) if hasattr(bsp.klu, 'time_begin') else str(bsp.klu.time)
                
                bsp_list.append({
                    "is_buy": bsp.is_buy,
                    "type": bsp.type2str(),
                    "time": time_str,
                    "price": float(bsp.klu.close),
                })
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
