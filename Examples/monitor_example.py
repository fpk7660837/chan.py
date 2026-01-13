"""
监控示例 - 买卖点监控

演示如何使用监控系统监控股票池的买卖点信号
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
from Monitor.MonitorEngine import CMonitorEngine
from Monitor.MonitorConfig import CMonitorConfig
from Monitor.EventDetector import CBSPDetector, CPriceBreakDetector
from Notification.NotificationService import CNotificationService
from Common.CEnum import DATA_SRC, KL_TYPE, BSP_TYPE

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def main():
    """主函数"""

    # 配置监控参数
    config = CMonitorConfig(
        scan_interval=300,  # 每5分钟扫描一次（实际使用时可以设置为60秒）
        stock_pool=[
            "000001",  # 平安银行
            "600519",  # 贵州茅台
            "000858",  # 五粮液
        ],
        data_src=DATA_SRC.AKSHARE,
        lv_list=[KL_TYPE.K_DAY],
        lookback_days=365,
        chan_config={
            "bi_strict": True,
            "divergence_rate": 0.9,
            "bs_type": "1,1p,2",
            "print_warning": False,
        },
        # 通知配置（需要替换为实际的webhook地址）
        notification_config={
            "dingtalk": {
                "webhook_url": "https://oapi.dingtalk.com/robot/send?access_token=YOUR_TOKEN",
                "secret": "YOUR_SECRET"  # 可选
            },
            # "wechat": {
            #     "webhook_url": "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=YOUR_KEY"
            # },
            # "feishu": {
            #     "webhook_url": "https://open.feishu.cn/open-apis/bot/v2/hook/YOUR_TOKEN"
            # }
        },
        work_hours=((9, 30), (15, 0)),  # 交易时间
    )

    # 创建监控引擎
    engine = CMonitorEngine(config)

    # 添加买卖点检测器
    stock_names = {
        "000001": "平安银行",
        "600519": "贵州茅台",
        "000858": "五粮液",
    }

    bsp_detector = CBSPDetector(
        bsp_types=[BSP_TYPE.T1, BSP_TYPE.T1P, BSP_TYPE.T2],
        time_window_days=3,
        stock_names=stock_names
    )
    engine.add_detector(bsp_detector)

    # 添加价格突破检测器
    price_detector = CPriceBreakDetector(
        break_type="both",
        lookback_days=20,
        stock_names=stock_names
    )
    engine.add_detector(price_detector)

    # 配置通知服务
    notification_service = CNotificationService(config.notification_config)
    engine.set_notification_service(notification_service)

    # 测试通知渠道
    print("\n=== 测试通知渠道 ===")
    notification_service.test_all()

    # 执行单次扫描测试
    print("\n=== 执行单次扫描测试 ===")
    result = engine.scan_once()
    print(f"扫描完成，检测到 {result['event_count']} 个事件")

    for event in result['events']:
        print(f"  - {event}")

    # 启动持续监控（可选）
    print("\n=== 启动监控引擎 ===")
    print("监控引擎将在后台运行，按 Ctrl+C 停止")

    try:
        engine.start()

        # 保持主线程运行
        import time
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n正在停止监控...")
        engine.stop()
        print("监控已停止")


if __name__ == "__main__":
    main()
