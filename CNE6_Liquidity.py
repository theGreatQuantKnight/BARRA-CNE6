import pymysql
import pandas as pd
import numpy as np
SENTINEL = 1e10

# 定义计算函数类
class CalcFunc(object):
    def __init__(self):
        pass

    def _cal_liquidity(self,series, days_pm=21, sentinel=-SENTINEL):
        series = series.astype(float).values
        freq = len(series) // days_pm
        valid_idx = np.argwhere(series != SENTINEL)
        series = series[valid_idx]
        res = np.log(np.nansum(series) / freq)
        if np.isinf(res):
            return sentinel
        else:
            return res

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

# 定义流动性因子对象
class Liquidity(CalcFunc):
    def __init__(self,turnovervalue):
        super().__init__()
        self.share_turnover = turnovervalue.where(pd.notnull(turnovervalue), SENTINEL)

    def STOM(self):
        '''
        :return:对最近21个交易日的股票换手率求和，然后取对数
        '''
        return self._rolling_apply(self.share_turnover, self._cal_liquidity, axis=0, window=21)

    def STOQ(self):
        '''
        :return:对最近63个交易日的股票换手率求和，然后取对数
        '''
        return self._rolling_apply(self.share_turnover, self._cal_liquidity, axis=0, window=63)

    def STOA(self):
        '''
        :return:对最近252个交易日的股票换手率求和，然后取对数
        '''
        return self._rolling_apply(self.share_turnover, self._cal_liquidity, axis=0, window=252)

    def ATVR(self):
        '''
        :return:对换手率进行加权求和，时间窗口为252个交易日，半衰期63个交易日
        '''
        return self._rolling(self.share_turnover, window=252, half_life=63, func_name='sum').T


if __name__ == '__main__':
    ####################################################################################################################
    # 连接数据库
    db = pymysql.connect()

    # 提取数据
    sql = """
    select TICKER_SYMBOL,TRADE_DATE,TURNOVER_RATE
    from mkt_equd
    where TRADE_DATE >= '2015-01-05'and TRADE_DATE <= '2020-12-28';
    """
    cursor = db.cursor()
    cursor.execute(sql)
    data = pd.DataFrame(cursor.fetchall(), columns=['stock', 'date', 'turnovervalue'])
    data['date'] = data['date'].astype(str)
    turnovervalue = data.pivot(index='date', columns='stock', values='turnovervalue')
    cursor.close()
    db.close()
    ####################################################################################################################
    # 初始化流动性因子
    LiqObject = Liquidity(turnovervalue) #将换手率传入流动性因子对象

    # 月换手率
    stom = LiqObject.STOM()
    # 季换手率
    stoq = LiqObject.STOQ()
    # 年换手率
    stoa = LiqObject.STOA()
    # 年化交易量比率
    atvr = LiqObject.ATVR()

    # 数据保存

    ####################################################################################################################



