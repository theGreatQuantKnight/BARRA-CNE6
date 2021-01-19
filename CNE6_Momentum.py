import pymysql
import pandas as pd
import numpy as np
from datetime import datetime
from functools import reduce
from itertools import  chain
SENTINEL = 1e10

# 定义计算函数类
class CalcFunc(object):
    def __init__(self):
        pass

    def _align(self, df1, df2, *dfs):
        dfs_all = [ df for df in chain([df1, df2], dfs)]
        if any(len(df.shape) == 1 or 1 in df.shape for df in dfs_all):
            dims = 1
        else:
            dims = 2
        mut_date_range = sorted(reduce(lambda x, y: x.intersection(y), (df.index for df in dfs_all)))
        mut_codes = sorted(reduce(lambda x, y: x.intersection(y), (df.columns for df in dfs_all)))
        if dims == 2:
            dfs_all = [df.loc[mut_date_range, mut_codes] for df in dfs_all]
        elif dims == 1:
            dfs_all = [df.loc[mut_date_range, :] for df in dfs_all]
        return dfs_all

    def __drop_invalid_and_fill_val(self, series, val=None, method=None):
        valid_idx = np.argwhere(series.notna()).flatten()
        try:
            series_valid = series.iloc[valid_idx[0]:]
        except IndexError:
            return series
        if val:
            series_valid = series_valid.fillna(val)
        elif method:
            series_valid = series_valid.fillna(method=method)
        else:
            median = np.nanmedian(series_valid)
            series_valid = series_valid.fillna(median)
        series = series.iloc[:valid_idx[0]].append(series_valid)
        return series

    def _rolling_apply(self, datdf, func, args=None, axis=0, window=None):
        if window:
            res = datdf.rolling(window=window).apply(func, args=args)
        else:
            res = datdf.apply(func, args=args, axis=axis)
        return res

    def _get_exp_weight(self,window, half_life):
        exp_wt = np.asarray([0.5 ** (1 / half_life)] * window) ** np.arange(window)
        return exp_wt[::-1] / np.sum(exp_wt)

    def weighted_std(self,series, weights):
        return np.sqrt(np.sum((series - np.mean(series)) ** 2 * weights))

    def weighted_func(self, func, series, weights):
        weights /= np.sum(weights)
        if func.__name__ == 'std':
            return self.weighted_std(series, weights)
        else:
            return func(series * weights)

    def nanfunc(self, series, func, sentinel=SENTINEL, weights=None):
        series = series.values
        valid_idx = np.argwhere(series != sentinel)
        if weights is not None:
            return self.weighted_func(func, series[valid_idx],weights=weights[valid_idx])
        else:
            return func(series[valid_idx])

    def _rolling(self, datdf, window, half_life=None,func_name='sum', weights=None):
        global SENTINEL
        datdf = datdf.where(pd.notnull(datdf), SENTINEL)
        datdf = datdf.astype(float)
        func = getattr(np, func_name, )

        if half_life or (weights is not None):
            exp_wt = self._get_exp_weight(window, half_life) if half_life else weights
            args = func, SENTINEL, exp_wt
        else:
            args = func, SENTINEL

        res = self._rolling_apply(datdf, self.nanfunc, args=args,axis=0, window=window)
        return res.T

# 定义动量因子对象
class Momentum(CalcFunc):
    def __init__(self,changepct,hfq_close,indexquote_changepct,negotiablemv,firstind):
        super().__init__()
        # self.changepct = changepct.where(pd.notnull(changepct), SENTINEL)
        self.changepct = changepct.astype('float')
        self.hfq_close = hfq_close.astype('float')
        self.indexquote_changepct = indexquote_changepct.astype('float')
        self.tdays = sorted(changepct.index)
        self.negotiablemv = negotiablemv.astype('float')
        self.firstind = firstind

    def STREV(self):
        '''
        :return: 最近一个月的加权累积对数日收益率，时间窗口21个交易日，半衰期5个交易日
        '''
        return self._rolling(self.changepct, window=21, half_life=5, func_name='sum')

    def SEASON(self):
        '''
        :return: 过去5年的已实现次月收益率的平均值
        '''
        nyears = 5
        pct_chg_m_d = hfq_close.pct_change(periods=21)
        pct_chgs_shift = [pct_chg_m_d.shift(i * 21 * 12 - 21) for i in range(1, nyears + 1)]
        seasonality = sum(pct_chgs_shift) / nyears
        return seasonality.T

    def INDMOM(self):
        '''
        :return: 行业动量，相对于中信一级
        '''
        window = 6 * 21;
        half_life = 21
        logret = np.log(1 + self.changepct)
        rs = self._rolling(logret, window, half_life, 'sum').T

        cap_sqrt = np.sqrt(self.negotiablemv)
        ind_citic_lv1 = self.firstind

        rs.columns = rs.columns.astype(int).astype(str)
        cap_sqrt.columns = cap_sqrt.columns.astype(int).astype(str)

        rs, cap_sqrt, ind_citic_lv1 = self._align(rs, cap_sqrt, ind_citic_lv1)

        rs.index.name = 'time'
        rs.columns.name = 'code'
        df = rs.unstack().copy()
        dat = df.reset_index()
        dat.columns = ['time', 'code', 'rs']

        cap_sqrt.index.name = 'time'
        cap_sqrt.columns.name = 'code'
        df = cap_sqrt.unstack().copy()
        df = df.reset_index()
        df.columns = ['time','code','weight']
        dat = pd.concat([dat,df['weight']],axis=1)

        ind_citic_lv1.index.name = 'time'
        ind_citic_lv1.columns.name = 'code'
        df = ind_citic_lv1.unstack().copy()
        df = df.reset_index()
        df.columns = ['time','code','ind']
        dat = pd.concat([dat,df['ind']],axis=1)

        rs_ind = {(time, ind): (df['weight'] * df['rs']).sum() / df['weight'].sum()
                  for time, df_gp in dat.groupby(['time'])
                  for ind, df in df_gp.groupby(['ind'])}

        def _get(key):
            try:
                return rs_ind[key]
            except:
                return np.nan

        dat['rs_ind'] = [_get((date, ind)) for date, ind in zip(dat['time'], dat['ind'])]
        dat['indmom'] = dat['rs_ind'] - dat['rs'] * dat['weight'] / dat['weight'].sum()
        indmom = pd.pivot_table(dat, values='indmom', index=['code'], columns=['time'])
        return indmom

    def RSTR(self):
        '''
        :return: 首先，对股票的对数收益率进行半衰指数加权求和，时间窗口252日，半衰期126日。然后以11个交易日为窗口，滞后11个交易日
        取费之后相对强度的等权平均值
        '''
        benchmark_ret = self.indexquote_changepct
        stock_ret = self.changepct

        benchmark_ret, stock_ret = self._align(benchmark_ret, stock_ret)
        benchmark_ret_np = np.asarray(benchmark_ret)
        benchmark_ret_np = np.repeat(benchmark_ret_np, stock_ret.shape[1], axis=1)
        benchmark_ret = pd.DataFrame(data = benchmark_ret_np,index = stock_ret.index, columns='windindex')

        excess_ret = np.log((1 + stock_ret).divide((1 + benchmark_ret), axis=0))
        rstr = self._rolling(excess_ret, window=252, half_life=126, func_name='sum')
        rstr = rstr.rolling(window=11, min_periods=1).mean()
        return rstr



if __name__ == '__main__':
    ####################################################################################################################
    # 连接数据库
    db = pymysql.connect()
    # 提取数据
    sql = """
    select TICKER_SYMBOL,TRADE_DATE,PRE_CLOSE_PRICE_2,CLOSE_PRICE_2
    from mkt_equd_adj_af 
    where TRADE_DATE >= '2015-01-05'and TRADE_DATE <= '2019-12-28';
    """
    cursor = db.cursor()
    cursor.execute(sql)
    data = pd.DataFrame(cursor.fetchall(), columns=['stock', 'date', 'pre_hfq_close','hfq_close'])
    # 昨收盘后复权
    pre_hfq_close = data.pivot(index='date', columns='stock', values='pre_hfq_close')
    # 收盘价后复权
    hfq_close = data.pivot(index='date', columns='stock', values='hfq_close')
    # 收益率
    changepct = hfq_close / pre_hfq_close - 1
    cursor.close()
    ####################################################################################################################
    # 提取数据
    sql = """
    select TICKER_SYMBOL,TRADE_DATE,NEG_MARKET_VALUE
    from mkt_equd
    where TRADE_DATE >= '2015-01-05'and TRADE_DATE <= '2019-12-28';
    """
    cursor = db.cursor()
    cursor.execute(sql)
    data = pd.DataFrame(cursor.fetchall(), columns=['stock', 'date', 'negmktv'])
    # 流通市值
    negotiablemv = data.pivot(index='date', columns='stock', values='negmktv')
    cursor.close()

    db.close()
    ####################################################################################################################
    # 连接数据库
    index = pymysql.connect()
    # 提取数据
    sql = """
    select date, Wind全A 
    from index_d_close
    where date >= '2014-12-30'and date <= '2019-12-28';
    """
    cursor1 = index.cursor()
    cursor1.execute(sql)
    data1 = pd.DataFrame(cursor1.fetchall(), columns = ['date','indexquote'])
    indequote = data1.set_index('date')

    indexquote_changepct = indequote.pct_change()
    indexquote_changepct = indexquote_changepct.loc[changepct.index.astype(str)]
    indexquote_changepct = indexquote_changepct.set_index(changepct.index)
    cursor1.close()

    index.close()
    ####################################################################################################################
    # 提取数据
    firstind = pd.read_csv(r'industry_citiccode.csv')
    firstind.iloc[:, 0] = firstind.iloc[:, 0].astype(str)
    def changetimetype(timeint):
        return datetime.date(datetime.strptime(timeint, '%Y%m%d'))
    firstind.iloc[:, 0] = firstind.iloc[:, 0].apply(changetimetype)
    firstind = firstind.set_index(firstind.iloc[:, 0]).iloc[:,1:]
    firstind = firstind.loc[changepct.index,:]
    ####################################################################################################################
    # 初始化动量因子因子
    MomObject = Momentum(changepct,hfq_close,indexquote_changepct,negotiablemv,firstind) #将收益率、后复权价格、指数收益率、流动市值、行业哑变量传入流动性因子对象

    # 短期反转
    short_term_reversal = MomObject.STREV()
    # 季节因子
    seasonality = MomObject.SEASON()
    # 行业动量
    industry_mom = MomObject.INDMOM()
    # 相对于市场的强度
    relative_strength = MomObject.RSTR()

    # 数据保存

    ####################################################################################################################







