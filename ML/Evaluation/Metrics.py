"""
评估指标

同时支持分类指标与交易/组合指标。
"""

from typing import Dict, List

import numpy as np


class Metrics:
    """评估指标计算器"""

    @staticmethod
    def calculate_classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray = None,
    ) -> Dict[str, float]:
        try:
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
        except ImportError as exc:
            raise ImportError("scikit-learn not installed. Please install with: pip install scikit-learn") from exc

        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        }

        if y_proba is not None:
            try:
                metrics['auc'] = float(roc_auc_score(y_true, y_proba))
            except Exception:
                metrics['auc'] = 0.0

        return metrics

    @staticmethod
    def calculate_trading_metrics(
        returns: np.ndarray,
        trades: List[Dict] = None,
        periods_per_year: int = 252,
    ) -> Dict[str, float]:
        returns = np.asarray(returns, dtype=float)
        if returns.size == 0:
            returns = np.array([0.0], dtype=float)

        metrics: Dict[str, float] = {}
        equity_curve = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = 1 - np.divide(equity_curve, running_max, out=np.ones_like(equity_curve), where=running_max != 0)

        total_return = float(equity_curve[-1] - 1)
        mean_return = float(np.mean(returns))
        std_return = float(np.std(returns))
        annualized_return = float((1 + total_return) ** (periods_per_year / max(len(returns), 1)) - 1) if equity_curve[-1] > 0 else -1.0

        metrics['total_return'] = total_return
        metrics['mean_return'] = mean_return
        metrics['std_return'] = std_return
        metrics['annualized_return'] = annualized_return
        metrics['max_drawdown'] = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

        if std_return > 0:
            metrics['sharpe_ratio'] = mean_return / std_return * np.sqrt(periods_per_year)
        else:
            metrics['sharpe_ratio'] = 0.0

        if metrics['max_drawdown'] > 0:
            metrics['calmar_ratio'] = annualized_return / metrics['max_drawdown']
        else:
            metrics['calmar_ratio'] = 0.0

        if trades:
            profits = [float(trade.get('profit', 0.0)) for trade in trades]
            winning_trades = [p for p in profits if p > 0]
            losing_trades = [p for p in profits if p < 0]

            metrics['total_trades'] = float(len(trades))
            metrics['winning_trades'] = float(len(winning_trades))
            metrics['losing_trades'] = float(len(losing_trades))
            metrics['win_rate'] = len(winning_trades) / len(trades) if trades else 0.0

            avg_profit = float(np.mean(winning_trades)) if winning_trades else 0.0
            avg_loss = abs(float(np.mean(losing_trades))) if losing_trades else 0.0
            metrics['avg_profit'] = avg_profit
            metrics['avg_loss'] = avg_loss
            metrics['profit_loss_ratio'] = avg_profit / avg_loss if avg_loss > 0 else 0.0

        return metrics

    @staticmethod
    def calculate_all_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray = None,
        returns: np.ndarray = None,
        trades: List[Dict] = None,
        periods_per_year: int = 252,
    ) -> Dict[str, float]:
        metrics = {}
        metrics.update(Metrics.calculate_classification_metrics(y_true, y_pred, y_proba))

        if returns is not None:
            metrics.update(Metrics.calculate_trading_metrics(returns, trades, periods_per_year=periods_per_year))

        return metrics

    @staticmethod
    def print_metrics(metrics: Dict[str, float], title: str = "Evaluation Metrics"):
        print(f"\n{'=' * 60}")
        print(f"{title:^60}")
        print(f"{'=' * 60}")

        classification_keys = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        print("\nClassification Metrics:")
        for key in classification_keys:
            if key in metrics:
                print(f"  {key.capitalize():20s}: {metrics[key]:.4f}")

        trading_keys = [
            'total_return',
            'annualized_return',
            'sharpe_ratio',
            'calmar_ratio',
            'max_drawdown',
            'win_rate',
            'profit_loss_ratio',
            'total_trades',
        ]
        print("\nTrading Metrics:")
        for key in trading_keys:
            if key not in metrics:
                continue
            value = metrics[key]
            if key in {'total_return', 'annualized_return', 'max_drawdown', 'win_rate'}:
                print(f"  {key.replace('_', ' ').title():20s}: {value:.2%}")
            elif key == 'total_trades':
                print(f"  {key.replace('_', ' ').title():20s}: {int(value)}")
            else:
                print(f"  {key.replace('_', ' ').title():20s}: {value:.4f}")

        print(f"{'=' * 60}\n")
