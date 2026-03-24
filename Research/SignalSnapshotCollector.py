from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from Chan import CChan

from .SignalSnapshot import SignalSnapshot


class SignalSnapshotCollector:
    """把缠论买卖点转成稳定的研究事件。"""

    def collect_chan(
        self,
        chan: CChan,
        *,
        direction: str = "buy",
        include_seg_bs: bool = False,
        signal_lv_idx: int = 0,
        child_confirm_main_bars: int = 1,
    ) -> List[SignalSnapshot]:
        snapshots: List[SignalSnapshot] = []
        snapshots.extend(
            self._collect_from_bsp_iter(
                chan,
                chan[signal_lv_idx].bs_point_lst.getSortedBspList(),
                direction=direction,
                is_seg_signal=False,
                signal_lv_idx=signal_lv_idx,
                child_confirm_main_bars=child_confirm_main_bars,
            )
        )
        if include_seg_bs:
            snapshots.extend(
                self._collect_from_bsp_iter(
                    chan,
                    chan[signal_lv_idx].seg_bs_point_lst.getSortedBspList(),
                    direction=direction,
                    is_seg_signal=True,
                    signal_lv_idx=signal_lv_idx,
                    child_confirm_main_bars=child_confirm_main_bars,
                )
            )
        snapshots.sort(key=lambda item: (item.signal_time.ts, item.signal_idx, item.code))
        return snapshots

    def collect(
        self,
        chan_list: List[CChan],
        *,
        direction: str = "buy",
        include_seg_bs: bool = False,
        signal_lv_idx: int = 0,
        child_confirm_main_bars: int = 1,
    ) -> List[SignalSnapshot]:
        snapshots: List[SignalSnapshot] = []
        for chan in chan_list:
            snapshots.extend(
                self.collect_chan(
                    chan,
                    direction=direction,
                    include_seg_bs=include_seg_bs,
                    signal_lv_idx=signal_lv_idx,
                    child_confirm_main_bars=child_confirm_main_bars,
                )
            )
        snapshots.sort(key=lambda item: (item.signal_time.ts, item.signal_idx, item.code))
        return snapshots

    def _collect_from_bsp_iter(
        self,
        chan: CChan,
        bsp_iter: Iterable,
        *,
        direction: str,
        is_seg_signal: bool,
        signal_lv_idx: int,
        child_confirm_main_bars: int,
    ) -> List[SignalSnapshot]:
        snapshots: List[SignalSnapshot] = []
        seen: Set[Tuple[str, int, str, bool, bool]] = set()
        for bsp in bsp_iter:
            if direction == "buy" and not bsp.is_buy:
                continue
            if direction == "sell" and bsp.is_buy:
                continue
            snapshot = SignalSnapshot.from_bsp(
                chan,
                bsp,
                lv_idx=signal_lv_idx,
                is_seg_signal=is_seg_signal,
            )
            self._attach_multilevel_context(
                snapshot,
                signal_lv_idx=signal_lv_idx,
                child_confirm_main_bars=child_confirm_main_bars,
            )
            dedupe_key = (
                snapshot.code,
                snapshot.signal_idx,
                snapshot.bsp_type,
                snapshot.is_buy,
                snapshot.is_seg_bsp,
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            snapshots.append(snapshot)
        return snapshots

    @staticmethod
    def _attach_multilevel_context(
        snapshot: SignalSnapshot,
        *,
        signal_lv_idx: int,
        child_confirm_main_bars: int,
    ) -> None:
        chan = snapshot.chan
        signal_klu = snapshot.signal_klu
        if chan is None or signal_klu is None:
            return

        if signal_lv_idx > 0:
            SignalSnapshotCollector._attach_parent_context(
                snapshot,
                chan=chan,
                parent_lv_idx=signal_lv_idx - 1,
                signal_klu=signal_klu,
            )

        if signal_lv_idx < len(chan.lv_list) - 1:
            SignalSnapshotCollector._attach_child_context(
                snapshot,
                chan=chan,
                signal_lv_idx=signal_lv_idx,
                signal_klu=signal_klu,
                child_confirm_main_bars=child_confirm_main_bars,
            )

        snapshot.joint_setup_key = SignalSnapshotCollector._build_joint_setup_key(snapshot)
        snapshot.joint_setup_label = SignalSnapshotCollector._build_joint_setup_label(snapshot)

    @staticmethod
    def _attach_parent_context(
        snapshot: SignalSnapshot,
        *,
        chan: CChan,
        parent_lv_idx: int,
        signal_klu,
    ) -> None:
        snapshot.parent_level = chan.lv_list[parent_lv_idx].name
        snapshot.parent_trend_state = "unknown"
        snapshot.parent_seg_position = "no_parent_klu"
        snapshot.parent_context = "no_seg|no_parent_klu|none"

        parent_klu = getattr(signal_klu, "sup_kl", None)
        latest_parent_bsp = None
        if parent_klu is None:
            return

        parent_history = []
        cursor = parent_klu
        while cursor is not None and len(parent_history) < 10:
            parent_history.append(cursor)
            cursor = getattr(cursor, "pre", None)
        if len(parent_history) >= 10:
            start_close = float(getattr(parent_history[-1], "close", 0.0) or 0.0)
            end_close = float(getattr(parent_klu, "close", 0.0) or 0.0)
            if start_close != 0:
                trend_return = (end_close - start_close) / start_close
                snapshot.parent_trend_return_10 = trend_return
                if trend_return >= 0.05:
                    snapshot.parent_trend_state = "bullish"
                elif trend_return <= -0.05:
                    snapshot.parent_trend_state = "bearish"
                else:
                    snapshot.parent_trend_state = "ranging"

        parent_seg = SignalSnapshotCollector._find_covering_line(
            chan[parent_lv_idx].seg_list,
            int(getattr(parent_klu, "idx", -1)),
        )
        if parent_seg is not None:
            snapshot.parent_seg_idx = int(getattr(parent_seg, "idx", -1))
            snapshot.parent_seg_dir = "down_seg" if parent_seg.is_down() else "up_seg"
            snapshot.parent_seg_is_sure = bool(getattr(parent_seg, "is_sure", False))
            snapshot.parent_seg_zs_count = len(getattr(parent_seg, "zs_lst", []))
            snapshot.parent_seg_multi_zs_count = int(parent_seg.get_multi_bi_zs_cnt())
            snapshot.parent_seg_position = SignalSnapshotCollector._classify_parent_seg_position(parent_seg, parent_klu)
        else:
            snapshot.parent_seg_dir = "no_seg"
            snapshot.parent_seg_position = "no_seg"

        for bsp in chan[parent_lv_idx].bs_point_lst.getSortedBspList():
            bsp_idx = getattr(getattr(bsp, "klu", None), "idx", -1)
            if bsp_idx <= parent_klu.idx:
                latest_parent_bsp = bsp
            else:
                break

        if latest_parent_bsp is not None:
            snapshot.parent_bsp_type = latest_parent_bsp.type2str()
            snapshot.parent_bsp_family = SignalSnapshotCollector._normalize_bsp_family(snapshot.parent_bsp_type)
            snapshot.parent_bsp_time = latest_parent_bsp.klu.time.to_str()
            snapshot.parent_bias_state = (
                "aligned" if bool(latest_parent_bsp.is_buy) == snapshot.is_buy else "counter"
            )
        else:
            snapshot.parent_bias_state = "none"

        snapshot.parent_context = SignalSnapshotCollector._build_parent_context(snapshot)

    @staticmethod
    def _attach_child_context(
        snapshot: SignalSnapshot,
        *,
        chan: CChan,
        signal_lv_idx: int,
        signal_klu,
        child_confirm_main_bars: int,
    ) -> None:
        snapshot.child_level = chan.lv_list[signal_lv_idx + 1].name
        child_events = SignalSnapshotCollector._collect_child_signal_events(
            chan=chan,
            child_lv_idx=signal_lv_idx + 1,
        )
        confirm = SignalSnapshotCollector._find_child_confirmation(
            signal_klu=signal_klu,
            child_events=child_events,
            is_buy=snapshot.is_buy,
            child_confirm_main_bars=child_confirm_main_bars,
        )
        if confirm is None:
            snapshot.child_interval_pattern = "no_signal"
            return

        snapshot.child_confirmed = confirm["confirmed"]
        snapshot.child_interval_state = confirm["interval_state"]
        snapshot.child_confirm_type = confirm["confirm_type"]
        snapshot.child_confirm_time = confirm["confirm_time"]
        snapshot.child_confirm_idx = confirm["confirm_idx"]
        snapshot.child_confirm_delay = confirm["confirm_delay"]
        snapshot.child_opposite_first = confirm["opposite_first"]
        snapshot.child_first_signal_side = confirm["first_signal_side"]
        snapshot.child_first_signal_type = confirm["first_signal_type"]
        snapshot.child_first_signal_family = confirm["first_signal_family"]
        snapshot.child_first_signal_delay = confirm["first_signal_delay"]
        snapshot.child_same_first_type = confirm["same_first_type"]
        snapshot.child_same_first_family = confirm["same_first_family"]
        snapshot.child_opposite_first_type = confirm["opposite_first_type"]
        snapshot.child_opposite_first_family = confirm["opposite_first_family"]
        snapshot.child_same_signal_count = confirm["same_signal_count"]
        snapshot.child_opposite_signal_count = confirm["opposite_signal_count"]
        snapshot.child_interval_pattern = confirm["interval_pattern"]

    @staticmethod
    def _collect_child_signal_events(
        *,
        chan: CChan,
        child_lv_idx: int,
    ) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        for source_kind, bsp_list, is_seg_signal in [
            ("bsp", chan[child_lv_idx].bs_point_lst.getSortedBspList(), False),
            ("seg_bsp", chan[child_lv_idx].seg_bs_point_lst.getSortedBspList(), True),
        ]:
            for bsp in bsp_list:
                klu = getattr(bsp, "klu", None)
                sup_kl = getattr(klu, "sup_kl", None)
                if klu is None or sup_kl is None:
                    continue
                bsp_type = bsp.type2str()
                events.append({
                    "time_ts": klu.time.ts,
                    "klu": klu,
                    "main_idx": int(getattr(sup_kl, "idx", -1)),
                    "is_buy": bool(bsp.is_buy),
                    "type": bsp_type,
                    "family": SignalSnapshotCollector._normalize_bsp_family(bsp_type),
                    "source_kind": source_kind,
                    "is_seg_signal": is_seg_signal,
                })
        events.sort(key=lambda item: (item["main_idx"], item["time_ts"]))
        return events

    @staticmethod
    def _find_child_confirmation(
        *,
        signal_klu,
        child_events: List[Dict[str, Any]],
        is_buy: bool,
        child_confirm_main_bars: int,
    ) -> Dict[str, Any]:
        signal_idx = int(getattr(signal_klu, "idx", -1))
        same_count = 0
        opposite_count = 0
        first_event = None
        first_same = None
        first_opposite = None

        for event in child_events:
            main_delay = int(event["main_idx"]) - signal_idx
            if main_delay <= 0:
                continue
            if main_delay > child_confirm_main_bars:
                break

            side = "same" if event["is_buy"] == is_buy else "opposite"
            event_info = {
                "side": side,
                "type": str(event["type"]),
                "family": str(event["family"]),
                "time": event["klu"].time.to_str(),
                "idx": int(getattr(event["klu"], "idx", -1)),
                "delay": main_delay,
            }

            if first_event is None:
                first_event = event_info

            if side == "same":
                same_count += 1
                if first_same is None:
                    first_same = event_info
            else:
                opposite_count += 1
                if first_opposite is None:
                    first_opposite = event_info

        interval_state = "no_signal"
        if first_event is not None:
            interval_state = "same_first" if first_event["side"] == "same" else "opposite_first"

        return {
            "confirmed": same_count > 0,
            "interval_state": interval_state,
            "confirm_type": first_same["type"] if first_same else "",
            "confirm_time": first_same["time"] if first_same else "",
            "confirm_idx": first_same["idx"] if first_same else -1,
            "confirm_delay": first_same["delay"] if first_same else -1,
            "opposite_first": interval_state == "opposite_first",
            "first_signal_side": first_event["side"] if first_event else "",
            "first_signal_type": first_event["type"] if first_event else "",
            "first_signal_family": first_event["family"] if first_event else "",
            "first_signal_delay": first_event["delay"] if first_event else -1,
            "same_first_type": first_same["type"] if first_same else "",
            "same_first_family": first_same["family"] if first_same else "",
            "opposite_first_type": first_opposite["type"] if first_opposite else "",
            "opposite_first_family": first_opposite["family"] if first_opposite else "",
            "same_signal_count": same_count,
            "opposite_signal_count": opposite_count,
            "interval_pattern": SignalSnapshotCollector._classify_child_interval_pattern(
                first_event=first_event,
                same_count=same_count,
                opposite_count=opposite_count,
            ),
        }

    @staticmethod
    def _normalize_bsp_family(bsp_type: str) -> str:
        if not bsp_type:
            return "unknown"
        parts = {part.strip() for part in str(bsp_type).split(",") if part.strip()}
        families = set()
        for part in parts:
            if part in {"1", "1p"}:
                families.add("1")
            elif part in {"2", "2s"}:
                families.add("2")
            elif part in {"3a", "3b"}:
                families.add("3")
        if not families:
            return "unknown"
        if families == {"1"}:
            return "1_like"
        if families == {"2"}:
            return "2_like"
        if families == {"3"}:
            return "3_like"
        if families == {"2", "3"}:
            return "2_3_like"
        if families == {"1", "2"}:
            return "1_2_like"
        if families == {"1", "3"}:
            return "1_3_like"
        return "mixed"

    @staticmethod
    def _find_covering_line(lines: Iterable[Any], klu_idx: int):
        active = None
        for line in lines:
            begin_klu = getattr(line, "get_begin_klu", lambda: None)()
            end_klu = getattr(line, "get_end_klu", lambda: None)()
            begin_idx = int(getattr(begin_klu, "idx", -1))
            end_idx = int(getattr(end_klu, "idx", -1))
            if begin_idx <= klu_idx <= end_idx:
                return line
            if begin_idx <= klu_idx:
                active = line
            elif begin_idx > klu_idx:
                break
        return active

    @staticmethod
    def _classify_parent_seg_position(parent_seg, parent_klu) -> str:
        latest_zs = parent_seg.get_final_multi_bi_zs() if parent_seg is not None else None
        if latest_zs is None:
            return "no_zs"
        if float(getattr(parent_klu, "high", 0.0) or 0.0) < float(getattr(latest_zs, "low", 0.0) or 0.0):
            return "below_zs"
        if float(getattr(parent_klu, "low", 0.0) or 0.0) > float(getattr(latest_zs, "high", 0.0) or 0.0):
            return "above_zs"
        return "inside_zs"

    @staticmethod
    def _build_parent_context(snapshot: SignalSnapshot) -> str:
        seg_dir = snapshot.parent_seg_dir or "no_seg"
        seg_position = snapshot.parent_seg_position or "no_zs"
        bias_state = snapshot.parent_bias_state or "none"
        return f"{seg_dir}|{seg_position}|{bias_state}"

    @staticmethod
    def _classify_child_interval_pattern(
        *,
        first_event: Optional[Dict[str, Any]],
        same_count: int,
        opposite_count: int,
    ) -> str:
        if first_event is None:
            return "no_signal"
        family = first_event.get("family", "unknown") or "unknown"
        speed = SignalSnapshotCollector._bucket_delay(int(first_event.get("delay", -1)))
        if first_event.get("side") == "same":
            noise = "clean" if opposite_count == 0 else "mixed"
            return f"same|{family}|{speed}|{noise}"
        recovery = "then_same" if same_count > 0 else "only"
        return f"opposite|{family}|{speed}|{recovery}"

    @staticmethod
    def _bucket_delay(delay: int) -> str:
        if delay < 0:
            return "unknown"
        if delay <= 2:
            return "fast"
        return "delayed"

    @staticmethod
    def _build_joint_setup_key(snapshot: SignalSnapshot) -> str:
        parent_context = snapshot.parent_context or "no_seg|no_parent_klu|none"
        child_pattern = snapshot.child_interval_pattern or "no_signal"
        return f"{snapshot.bsp_type}||{parent_context}||{child_pattern}"

    @staticmethod
    def _build_joint_setup_label(snapshot: SignalSnapshot) -> str:
        parent_context = snapshot.parent_context or "no_seg|no_parent_klu|none"
        child_pattern = snapshot.child_interval_pattern or "no_signal"
        return f"{parent_context} + {child_pattern}"
