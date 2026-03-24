from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Plot.PlotDriver import CPlotDriver

from ..SignalSnapshot import SignalSnapshot


class SignalVisualizer:
    """输出带结构和事件说明的单信号图。"""

    DEFAULT_PLOT_CONFIG = {
        "plot_kline": True,
        "plot_kline_combine": True,
        "plot_bi": True,
        "plot_seg": True,
        "plot_zs": True,
        "plot_bsp": True,
        "plot_marker": True,
        "plot_macd": False,
        "plot_segzs": False,
        "plot_eigen": False,
        "plot_segbsp": False,
        "plot_demark": False,
        "plot_rsi": False,
        "plot_kdj": False,
    }

    def render_signal(
        self,
        snapshot: SignalSnapshot,
        *,
        output_path: Path,
        evaluation: Optional[Dict[str, Any]] = None,
        x_range: int = 200,
        plot_config: Optional[Dict[str, Any]] = None,
    ) -> Path:
        if snapshot.chan is None:
            raise ValueError("Signal snapshot has no chan reference, cannot render chart.")

        marker_label = snapshot.bsp_type or snapshot.signal_kind
        marker_color = "red" if snapshot.is_buy else "green"
        plot_para = {
            "figure": {
                "x_range": x_range,
                "w": 18,
                "h": 8,
            },
            "marker": {
                "markers": {
                    snapshot.signal_time.to_str(): (marker_label, "down" if snapshot.is_buy else "up", marker_color),
                }
            },
        }
        driver = CPlotDriver(snapshot.chan, plot_config=plot_config or self.DEFAULT_PLOT_CONFIG, plot_para=plot_para)
        title = f"{snapshot.code} {snapshot.signal_time_str} {snapshot.signal_kind} {snapshot.direction}"
        driver.figure.suptitle(title, fontsize=16)
        self._add_annotation(driver.figure, snapshot, evaluation)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        driver.save2img(str(output_path))
        plt.close(driver.figure)
        return output_path

    def _add_annotation(
        self,
        figure,
        snapshot: SignalSnapshot,
        evaluation: Optional[Dict[str, Any]],
    ):
        lines = [
            f"source={snapshot.source}",
            f"bsp_type={snapshot.bsp_type or '-'}",
            f"signal_idx={snapshot.signal_idx}",
            f"is_buy={int(snapshot.is_buy)} is_seg_bsp={int(snapshot.is_seg_bsp)}",
            f"is_sure={int(snapshot.is_sure)} bi_is_sure={int(snapshot.bi_is_sure)} seg_is_sure={int(snapshot.seg_is_sure)}",
            f"zs_count={snapshot.zs_count} divergence={snapshot.divergence_rate:.4f}",
        ]
        if snapshot.parent_level:
            lines.append(
                f"parent={snapshot.parent_level} context={snapshot.parent_context or '-'} "
                f"trend={snapshot.parent_trend_state or '-'} "
                f"bsp={snapshot.parent_bsp_type or '-'}"
            )
        if snapshot.child_level:
            lines.append(
                f"child={snapshot.child_level} pattern={snapshot.child_interval_pattern or '-'} "
                f"state={snapshot.child_interval_state or '-'} confirmed={int(snapshot.child_confirmed)} "
                f"type={snapshot.child_confirm_type or '-'} delay={snapshot.child_confirm_delay}"
            )
        if snapshot.joint_setup_key:
            lines.append(f"joint_setup={snapshot.joint_setup_key}")
        if evaluation:
            for horizon in (5, 10, 20):
                prefix = f"h{horizon}"
                if evaluation.get(f"{prefix}_available") != 1:
                    continue
                lines.append(
                    f"{prefix}: ret={evaluation.get(f'{prefix}_return', 0.0):.2%} "
                    f"mfe={evaluation.get(f'{prefix}_mfe', 0.0):.2%} "
                    f"mae={evaluation.get(f'{prefix}_mae', 0.0):.2%} "
                    f"sl={int(evaluation.get(f'{prefix}_stop_loss_hit', 0))} "
                    f"rev={int(evaluation.get(f'{prefix}_reverse_signal_hit', 0))}"
                )
        figure.text(
            0.01,
            0.98,
            "\n".join(lines),
            va="top",
            ha="left",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#999999"},
        )
