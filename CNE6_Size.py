import pymysql
import pandas as pd
import numpy as np
import statsmodels.api as sm

# 定义计算函数类
class CalcFunc(object):
    def _regress(self,y, X, intercept=True, weight=1, verbose=True):
        if not isinstance(y, (pd.Series, pd.DataFrame)):
            y = pd.DataFrame(y)
        if not isinstance(X, (pd.Series, pd.DataFrame)):
            X = pd.DataFrame(X)

        if intercept:
            cols = X.columns.tolist()
            X['const'] = 1
            X = X[['const'] + cols]

        model = sm.WLS(y, X, weights=weight)
        result = model.fit()
        params = result.params

        if verbose:
            resid = y - pd.DataFrame(np.dot(X, params), index=y.index,
                                     columns=y.columns)
            if intercept:
                return params.iloc[1:], params.iloc[0], resid
            else:
                return params, None, resid
        else:
            if intercept:
                return params.iloc[1:]
            else:
                return params

    def _cal_midcap(self, series):
        x = series.dropna().values
        y = x ** 3
        beta, alpha, _ = self._regress(y, x, intercept=True, weight=1, verbose=True)
        resid = series ** 3 - (alpha + beta[0] * series)
        return resid

    def _pandas_applycal(self, dat, myfunc, args=None, axis=1, window=None):
        if axis == 0 and window is None:
            dat = dat.T
        if window:
            dat = dat.rolling(window=window)
            if args is None:
                res = dat.apply(myfunc)
            else:
                res = dat.apply(myfunc, args=args)
        else:
            res = dat.apply(myfunc, args=args, axis=1)
        return res.T

# 定义规模因子对象
class Size(CalcFunc):
    def __init__(self,negotiablemv):
        self.negotiablemv = negotiablemv

    def lncap(self):
        '''
        :return: 流通市值的自然对数
        '''
        self.lncap = self.negotiablemv.apply(np.log)
        return self.lncap

    def midcap(self):
        '''
        :return:首先取Size因子暴露的立方，然后以加权回归的方式对Size因子正交
        '''
        return self._pandas_applycal(self.lncap, self._cal_midcap, axis=0)


if __name__ == '__main__':
    ####################################################################################################################
    # 连接数据库
    db = pymysql.connect()
    # 提取数据
    sql = """
    select TICKER_SYMBOL,TRADE_DATE,NEG_MARKET_VALUE
    from mkt_equd
    where TRADE_DATE >= '2015-01-05';
    """
    cursor = db.cursor()
    cursor.execute(sql)
    data = pd.DataFrame(cursor.fetchall(), columns=['stock', 'date', 'negmktv'])
    # 流动市值
    negotiablemv = data.pivot(index='date', columns='stock', values='negmktv')
    negotiablemv = negotiablemv.astype('float')
    cursor.close()
    db.close()
    ####################################################################################################################
    # 初始化规模因子
    SizeObject = Size(negotiablemv) #将流动市值传入规模因子对象

    # 规模
    lncap = SizeObject.lncap()
    # 中市值
    midcap = SizeObject.midcap()

    # 数据保存

    ####################################################################################################################
