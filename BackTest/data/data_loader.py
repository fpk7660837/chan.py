import baostock as bs
import pandas as pd
from datetime import datetime
import os

class DataLoader:
    def __init__(self, data_dir='data'):
        """
        数据加载器
        :param data_dir: 数据存储目录
        """
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

    def get_stock_data(self, code, start_date='2018-01-01', end_date=None, force_download=False, frequency='d'):
        """
        获取股票数据，优先从本地读取，如果本地没有则从baostock下载
        :param frequency: K线周期，默认为日K，可选值：d=日k线、w=周、m=月、5=5分钟、15=15分钟、30=30分钟、60=60分钟
        """
        # 构建本地文件路径
        file_name = f"{code}_{start_date}_{end_date if end_date else 'latest'}_{frequency}.csv"
        file_path = os.path.join(self.data_dir, file_name)
        
        # 如果本地文件存在且不强制下载，则直接读取
        if os.path.exists(file_path) and not force_download:
            df = pd.read_csv(file_path)
            if frequency != 'd':  # 分钟级别数据需要特殊处理
                # 将time列转换为正确的时间格式
                df['time'] = df['time'].apply(lambda x: pd.to_datetime(x, format='%Y%m%d%H%M%S%f').strftime('%H:%M:%S'))
                df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
                df.set_index('datetime', inplace=True)
                df = df.drop(['date', 'time', 'code'], axis=1)  # 删除不需要的列
            else:
                df.set_index('date', inplace=True)
                df.index = pd.to_datetime(df.index)
                df = df.drop(['code'], axis=1)  # 删除不需要的列
            return df
            
        # 否则从baostock下载
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        # 登录baostock
        lg = bs.login()
        if lg.error_code != '0':
            print('login respond error_code:'+lg.error_code)
            print('login respond  error_msg:'+lg.error_msg)
        
        # 获取数据
        fields = "date,time,code,open,high,low,close,volume,amount" if frequency != 'd' else "date,code,open,high,low,close,volume,amount"
        rs = bs.query_history_k_data_plus(
            code,
            fields,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            adjustflag="3"  # 复权类型，3：后复权
        )
        
        # 转换为DataFrame
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        df = pd.DataFrame(data_list, columns=rs.fields)
        
        # 打印数据格式以便调试
        print("Data columns:", df.columns)
        print("\nFirst few rows:")
        print(df.head())
        
        # 转换数据类型
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
        
        # 设置索引
        if frequency != 'd':  # 分钟级别数据需要特殊处理
            # 将time列转换为正确的时间格式
            df['time'] = df['time'].apply(lambda x: pd.to_datetime(x, format='%Y%m%d%H%M%S%f').strftime('%H:%M:%S'))
            df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
            df.set_index('datetime', inplace=True)
            df = df.drop(['date', 'time', 'code'], axis=1)  # 删除不需要的列
        else:
            df.set_index('date', inplace=True)
            df.index = pd.to_datetime(df.index)
            df = df.drop(['code'], axis=1)  # 删除不需要的列
        
        # 保存到本地
        df.to_csv(file_path)
        
        # 登出系统
        bs.logout()
        
        return df
