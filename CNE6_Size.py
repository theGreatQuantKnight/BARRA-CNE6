import pymysql
import pandas as pd
import numpy as np
import statsmodels.api as sm
from dask import dataframe as dd
from dask.multiprocessing import get

db = pymysql.connect(host='192.168.254.77', port=32061, user='dengtao', password='Abcd1234', database='tonglian')


class Size(object):
    def __init__(self):
        sql = """
        select TICKER_SYMBOL,TRADE_DATE,NEG_MARKET_VALUE
        from mkt_equd
        where TRADE_DATE >= '2010-01-01';
        """
        cursor = db.cursor()
        cursor.execute(sql)
        data = pd.DataFrame(cursor.fetchall(), columns=['stock', 'date', 'negmktv'])
        self.negotiablemv = data.pivot(index='date', columns='stock', values='negmktv')
        cursor.close()

    def _regress(y, X, intercept=True, weight=1, verbose=True):
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

    def _pandas_parallelcal(self, dat, myfunc, ncores=6, args=None, axis=1, window=None):
        if axis == 0 and window is None:
            dat = dat.T
        dat = dd.from_pandas(dat, npartitions=ncores)
        if window:
            dat = dat.rolling(window=window)
            if args is None:
                res = dat.apply(myfunc)
            else:
                res = dat.apply(myfunc, args=args)
        else:
            res = dat.apply(myfunc, args=args, axis=1)
        return res.compute(get=get)

    def lncap(self):
        '''
        :return: 流通市值的自然对数
        '''
        self.lncap = np.log(self.negotiablemv)
        return self.lncap

    def midcap(self):
        '''
        :return:首先取Size因子暴露的立方，然后以加权回归的方式对Size因子正交
        '''
        self.midcap = self._pandas_parallelcal(self.lncap, self._cal_midcap, axis=0).T
        return self.midcap


db.close()
