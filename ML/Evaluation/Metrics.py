"""
评估指标

计算各种模型评估指标：AUC、精确率、夏普比率、卡玛比率、胜率、盈亏比等
"""

from typing import Dict, Any, List
import numpy as np


class Metrics:
    """评估指标计算器"""

    @staticmethod
    def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> Dict[str, float]:
        """
        计算分类指标

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_proba: 预测概率

        Returns:
            指标字典
        """
        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        except ImportError:
            raise ImportError("scikit-learn not installed. Please install with: pip install scikit-learn")

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
        }

        # AUC需要概率
        if y_proba is not None:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_proba)
            except:
                metrics['auc'] = 0.0

        return metrics

    @staticmethod
    def calculate_trading_metrics(returns: np.ndarray, trades: List[Dict] = None) -> Dict[str, float]:
        """
        计算交易指标

        Args:
            returns: 收益率序列
            trades: 交易记录列表，每个元素包含：
                - entry_price: 入场价格
                - exit_price: 出场价格
                - direction: 方向（'buy'或'sell'）
                - profit: 盈亏

        Returns:
            指标字典
        """
        metrics = {}

        # 基础统计
        metrics['total_return'] = np.sum(returns)
        metrics['mean_return'] = np.mean(returns)
        metrics['std_return'] = np.std(returns)

        # 夏普比率
        if metrics['std_return'] > 0:
            metrics['sharpe_ratio'] = metrics['mean_return'] / metrics['std_return'] * np.sqrt(252)  # 年化
        else:
            metrics['sharpe_ratio'] = 0.0

        # 最大回撤
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = running_max - cumulative_returns
        metrics['max_drawdown'] = np.max(drawdown) if len(drawdown) > 0 else 0.0

        # 卡玛比率
        if metrics['max_drawdown'] > 0:
            metrics['calmar_ratio'] = metrics['total_return'] / metrics['max_drawdown']
        else:
            metrics['calmar_ratio'] = 0.0

        # 如果有交易记录，计算胜率和盈亏比
        if trades is not None and len(trades) > 0:
            profits = [trade.get('profit', 0) for trade in trades]
            winning_trades = [p for p in profits if p > 0]
            losing_trades = [p for p in profits if p < 0]

            metrics['total_trades'] = len(trades)
            metrics['winning_trades'] = len(winning_trades)
            metrics['losing_trades'] = len(losing_trades)

            # 胜率
            metrics['win_rate'] = len(winning_trades) / len(trades) if len(trades) > 0 else 0.0

            # 盈亏比
            avg_profit = np.mean(winning_trades) if len(winning_trades) > 0 else 0.0
            avg_loss = abs(np.mean(losing_trades)) if len(losing_trades) > 0 else 0.0

            if avg_loss > 0:
                metrics['profit_loss_ratio'] = avg_profit / avg_loss
            else:
                metrics['profit_loss_ratio'] = 0.0

            metrics['avg_profit'] = avg_profit
            metrics['avg_loss'] = avg_loss

        return metrics

    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                              y_proba: np.ndarray = None,
                              returns: np.ndarray = None,
                              trades: List[Dict] = None) -> Dict[str, float]:
        """
        计算所有指标

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_proba: 预测概率
            returns: 收益率序列
            trades: 交易记录

        Returns:
            完整指标字典
        """
        metrics = {}

        # 分类指标
        classification_metrics = Metrics.calculate_classification_metrics(y_true, y_pred, y_proba)
        metrics.update(classification_metrics)

        # 交易指标
        if returns is not None:
            trading_metrics = Metrics.calculate_trading_metrics(returns, trades)
            metrics.update(trading_metrics)

        return metrics

    @staticmethod
    def print_metrics(metrics: Dict[str, float], title: str = "Evaluation Metrics"):
        """
        打印指标

        Args:
            metrics: 指标字典
            title: 标题
        """
        print(f"\n{'='*60}")
        print(f"{title:^60}")
        print(f"{'='*60}")

        # 分类指标
        classification_keys = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        print("\nClassification Metrics:")
        for key in classification_keys:
            if key in metrics:
                print(f"  {key.capitalize():20s}: {metrics[key]:.4f}")

        # 交易指标
        trading_keys = ['total_return', 'sharpe_ratio', 'calmar_ratio', 'max_drawdown',
                       'win_rate', 'profit_loss_ratio', 'total_trades']
        print("\nTrading Metrics:")
        for key in trading_keys:
            if key in metrics:
                value = metrics[key]
                if key in ['total_return', 'max_drawdown']:
                    print(f"  {key.replace('_', ' ').title():20s}: {value:.2%}")
                elif key == 'total_trades':
                    print(f"  {key.replace('_', ' ').title():20s}: {int(value)}")
                else:
                    print(f"  {key.replace('_', ' ').title():20s}: {value:.4f}")

        print(f"{'='*60}\n")
