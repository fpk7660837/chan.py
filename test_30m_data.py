"""
测试30分钟K线数据源可用性
"""

import sys
sys.path.insert(0, '.')

from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE

# 测试股票
test_code = "sz.000001"
begin_time = "2023-01-01"
end_time = "2023-12-31"

print("测试30分钟K线数据源可用性")
print("="*60)

# 测试BaoStock
print("\n1. 测试 BaoStock - 30分钟数据...")
try:
    config = CChanConfig({"trigger_step": True})
    chan = CChan(
        code=test_code,
        begin_time=begin_time,
        end_time=end_time,
        data_src=DATA_SRC.BAO_STOCK,
        lv_list=[KL_TYPE.K_30M],
        config=config,
        autype=AUTYPE.QFQ,
    )

    count = 0
    for snapshot in chan.step_load():
        count += 1
        if count >= 5:
            break

    if count > 0:
        print(f"   ✓ BaoStock支持30分钟数据 (测试加载了{count}条)")
    else:
        print("   ✗ BaoStock不支持30分钟数据")
except Exception as e:
    print(f"   ✗ BaoStock错误: {e}")

# 测试AkShare
print("\n2. 测试 AkShare - 30分钟数据...")
try:
    config = CChanConfig({"trigger_step": True})
    chan = CChan(
        code=test_code,
        begin_time=begin_time,
        end_time=end_time,
        data_src=DATA_SRC.AKSHARE,
        lv_list=[KL_TYPE.K_30M],
        config=config,
        autype=AUTYPE.QFQ,
    )

    count = 0
    for snapshot in chan.step_load():
        count += 1
        if count >= 5:
            break

    if count > 0:
        print(f"   ✓ AkShare支持30分钟数据 (测试加载了{count}条)")
    else:
        print("   ✗ AkShare不支持30分钟数据")
except Exception as e:
    print(f"   ✗ AkShare错误: {e}")

print("\n" + "="*60)
print("测试结论:")
print("由于30分钟数据获取受限，建议:")
print("1. 使用日线数据进行回测（数据最稳定）")
print("2. 或者使用60分钟数据（如果支持）")
print("3. 或者准备CSV格式的30分钟数据")
