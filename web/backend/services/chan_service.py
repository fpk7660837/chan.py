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
        chan_config = CChanConfig({
            "bi_strict": params.get("bi_strict", True),
            "seg_algo": params.get("seg_algo", "chan"),
            "zs_algo": params.get("zs_algo", "normal"),
            "print_warning": False,  # Disable warnings for cleaner API response
        })
        
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
                    # Debug: log first K-line time
                    if idx == 0:
                        print(f"🕐 Backend: First K-line time_begin = {time_str}")
                        print(f"🕐 Backend: time_begin type = {type(klc.time_begin)}")
                    
                    klines.append({
                        "time": time_str,
                        "open": float(first_klu.open),
                        "high": float(klc.high),
                        "low": float(klc.low),
                        "close": float(last_klu.close),
                        "volume": float(first_klu.trade_info.volume) if first_klu.trade_info and hasattr(first_klu.trade_info, 'volume') else 0,
                    })
        except Exception as e:
            print(f"⚠️  Warning extracting K-line data: {e}")
            import traceback
            traceback.print_exc()
        
        if klines:
            print(f"📊 Backend: Extracted {len(klines)} K-lines, first time = {klines[0]['time']}, last time = {klines[-1]['time']}")
        
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
                        
                        zs_list.append({
                            "low": float(zs.low),
                            "high": float(zs.high),
                            "begin_time": begin_time,
                            "end_time": end_time,
                            "bi_count": len(zs.bi_lst),
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

