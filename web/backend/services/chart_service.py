"""
Chart generation service
Generate interactive Plotly charts from Chan analysis results
"""
from typing import Dict, Any, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ChartService:
    """Service class for chart generation"""
    
    def generate_plotly_chart(self, data: Dict[str, Any]) -> Dict:
        """
        Generate Plotly chart configuration
        
        Args:
            data: Chart data including K-lines and indicators
            
        Returns:
            Plotly JSON configuration
        """
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=(f'{data["code"]} - Chan Theory Analysis', 'Volume')
        )
        
        # Add candlestick chart
        kline_data = data["kline_data"]
        fig.add_trace(
            go.Candlestick(
                x=[kl["time"] for kl in kline_data],
                open=[kl["open"] for kl in kline_data],
                high=[kl["high"] for kl in kline_data],
                low=[kl["low"] for kl in kline_data],
                close=[kl["close"] for kl in kline_data],
                name="K-Line",
                increasing_line_color='red',
                decreasing_line_color='green',
            ),
            row=1, col=1
        )
        
        # Add Bi (笔) lines
        if data.get("plot_bi") and data.get("bi_list"):
            bi_times = []
            bi_prices = []
            for bi in data["bi_list"]:
                bi_times.extend([bi["begin_time"], bi["end_time"], None])
                bi_prices.extend([bi["begin_price"], bi["end_price"], None])
            
            fig.add_trace(
                go.Scatter(
                    x=bi_times,
                    y=bi_prices,
                    mode='lines',
                    name='Bi (笔)',
                    line=dict(color='blue', width=2),
                    hovertemplate='Bi<br>Time: %{x}<br>Price: %{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Add Seg (线段) lines
        if data.get("plot_seg") and data.get("seg_list"):
            seg_times = []
            seg_prices = []
            for seg in data["seg_list"]:
                seg_times.extend([seg["begin_time"], seg["end_time"], None])
                seg_prices.extend([seg["begin_price"], seg["end_price"], None])
            
            fig.add_trace(
                go.Scatter(
                    x=seg_times,
                    y=seg_prices,
                    mode='lines',
                    name='Seg (线段)',
                    line=dict(color='purple', width=3),
                    hovertemplate='Seg<br>Time: %{x}<br>Price: %{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Add ZhongShu (中枢) rectangles
        if data.get("plot_zs") and data.get("zs_list"):
            for zs in data["zs_list"]:
                fig.add_shape(
                    type="rect",
                    x0=zs["begin_time"],
                    x1=zs["end_time"],
                    y0=zs["low"],
                    y1=zs["high"],
                    fillcolor="orange",
                    opacity=0.2,
                    line=dict(color="orange", width=2),
                    row=1, col=1
                )
        
        # Add BuySellPoints (买卖点)
        if data.get("plot_bsp") and data.get("bsp_list"):
            buy_points = [bsp for bsp in data["bsp_list"] if bsp["is_buy"]]
            sell_points = [bsp for bsp in data["bsp_list"] if not bsp["is_buy"]]
            
            if buy_points:
                fig.add_trace(
                    go.Scatter(
                        x=[bp["time"] for bp in buy_points],
                        y=[bp["price"] for bp in buy_points],
                        mode='markers+text',
                        name='Buy Point',
                        marker=dict(symbol='triangle-up', size=12, color='red'),
                        text=[bp["type"] for bp in buy_points],
                        textposition="bottom center",
                        hovertemplate='Buy<br>%{text}<br>Time: %{x}<br>Price: %{y:.2f}<extra></extra>'
                    ),
                    row=1, col=1
                )
            
            if sell_points:
                fig.add_trace(
                    go.Scatter(
                        x=[sp["time"] for sp in sell_points],
                        y=[sp["price"] for sp in sell_points],
                        mode='markers+text',
                        name='Sell Point',
                        marker=dict(symbol='triangle-down', size=12, color='green'),
                        text=[sp["type"] for sp in sell_points],
                        textposition="top center",
                        hovertemplate='Sell<br>%{text}<br>Time: %{x}<br>Price: %{y:.2f}<extra></extra>'
                    ),
                    row=1, col=1
                )
        
        # Add volume bars
        colors = ['red' if kl["close"] >= kl["open"] else 'green' for kl in kline_data]
        fig.add_trace(
            go.Bar(
                x=[kl["time"] for kl in kline_data],
                y=[kl["volume"] for kl in kline_data],
                name="Volume",
                marker_color=colors,
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f'{data["code"]} Chan Theory Analysis',
            xaxis_title="Time",
            yaxis_title="Price",
            height=data.get("height", 800),
            width=data.get("width", 1200),
            hovermode='x unified',
            xaxis_rangeslider_visible=False,
            template="plotly_white"
        )
        
        # Return JSON for frontend
        return fig.to_json()

