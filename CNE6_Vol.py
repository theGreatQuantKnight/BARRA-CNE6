import pymysql
import pandas as pd
import numpy as np
import statsmodels.api as sm
from dask import dataframe as dd
from dask.multiprocessing import get
from functools import reduce
import pandas.tseries.offsets as toffsets
from itertools import dropwhile, chain, product

db = pymysql.connect(host='192.168.254.76', port=32060, user='dengtao', password='Abcd1234', database='tonglian')

index = pymysql.connect(host='192.168.254.76', port=32060, user='dengtao', password='Abcd1234', database='quant_db')

SENTINEL = 1e10

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

    def _get_exp_weight(self,window, half_life):
        exp_wt = np.asarray([0.5 ** (1 / half_life)] * window) ** np.arange(window)
        return exp_wt[::-1] / np.sum(exp_wt)

    def rolling_windows(self,a, window):
        if window > a.shape[0]:
            raise ValueError(
                "Specified `window` length of {0} exceeds length of"
                " `a`, {1}.".format(window, a.shape[0])
            )
        if isinstance(a, (pd.Series, pd.DataFrame)):
            a = a.values
        if a.ndim == 1:
            a = a.reshape(-1, 1)
        shape = (a.shape[0] - window + 1, window) + a.shape[1:]
        strides = (a.strides[0],) + a.strides
        windows = np.squeeze(
            np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
        )
        if windows.ndim == 1:
            windows = np.atleast_2d(windows)
        return windows

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

    def _rolling_regress(self, y, x, window=5, half_life=None,intercept=True, verbose=False, fill_na=0):
        fill_args = {'method': fill_na} if isinstance(fill_na, str) else {'value': fill_na}
        # 取x和y的并集
        x, y = self._align(x, y)
        # 取股票代码序列
        stocks = y.columns
        # 加权
        if half_life:
            weight = self._get_exp_weight(window, half_life)
        else:
            weight = 1
        # 去除前几行nan值
        start_idx = x.loc[pd.notnull(x).values.flatten()].index[0]
        x, y = x.loc[start_idx:], y.loc[start_idx:, :]

        rolling_ys = self.rolling_windows(y, window)
        rolling_xs = self.rolling_windows(x, window)

        beta = pd.DataFrame()
        alpha = pd.DataFrame()
        sigma = pd.DataFrame()

        for i, (rolling_x, rolling_y) in enumerate(zip(rolling_xs, rolling_ys)):
            rolling_y = pd.DataFrame(rolling_y, columns=y.columns,
                                     index=y.index[i:i + window])
            window_sdate, window_edate = rolling_y.index[0], rolling_y.index[-1]
            rolling_y = rolling_y.fillna(**fill_args)
            rolling_y = rolling_y.astype(float)
            try:
                b, a, resid = self._regress(rolling_y.values, rolling_x,
                                            intercept=True, weight=weight, verbose=True)
            except:
                print(i)
                raise
            vol = np.std(resid, axis=0)
            vol.index = a.index = b.columns = stocks
            b.index = [window_edate]
            vol.name = a.name = window_edate
            beta = pd.concat([beta, b], axis=0)
            alpha = pd.concat([alpha, a], axis=1)
            sigma = pd.concat([sigma, vol], axis=1)

        beta = beta.T
        beta = beta.reindex(y.index, axis=1).reindex(y.columns, axis=0)
        alpha = alpha.reindex(y.index, axis=1).reindex(y.columns, axis=0)
        sigma = sigma.reindex(y.index, axis=1).reindex(y.columns, axis=0)
        return beta, alpha, sigma

    def _rolling_apply(self, datdf, func, args=None, axis=0, window=None):
        if window:
            res = datdf.rolling(window=window).apply(func, args=args)
        else:
            res = datdf.apply(func, args=args, axis=axis)
        return res

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
        if func is None:
            msg = f"""Search func:{func_name} from numpy failed, 
                   only numpy ufunc is supported currently, please retry."""
            raise AttributeError(msg)

        if half_life or (weights is not None):
            exp_wt = self._get_exp_weight(window, half_life) if half_life else weights
            args = func, SENTINEL, exp_wt
        else:
            args = func, SENTINEL

        try:
            res = self._pandas_parallelcal(datdf, self.nanfunc, args=args,
                                           axis=0, window=window)
        except Exception:
            print('Calculating under single core mode...')
            res = self._rolling_apply(datdf, self.nanfunc, args=args,axis=0, window=window)
        return res.T

class Volatility(CalcFunc):
    def __init__(self):
        super().__init__()
        sql = """Select date, Wind全A
        from index_d_close
        where date >= '2010-01-04' and date <= '2020-12-28';"""
        cursor = index.cursor()
        cursor.execute(sql)
        data= pd.DataFrame(cursor.fetchall(), columns=['date','mktret'])
        data['date'] = data['date'].astype(str)
        mktret = data.set_index('date')
        self.mktret = mktret.pct_change().iloc[1:,:]
        cursor.close()

        sql = """
        select TICKER_SYMBOL,TRADE_DATE,DAILY_RETURN_REINV
        from equ_retud
        where TRADE_DATE >= '2010-01-05'and TRADE_DATE <= '2020-12-28';
        """
        cursor = db.cursor()
        cursor.execute(sql)
        data = pd.DataFrame(cursor.fetchall(), columns=['stock', 'date', 'dailyret'])
        data['date'] = data['date'].astype(str)
        self.dailyret = data.pivot(index='date', columns='stock', values='dailyret')
        cursor.close()



    def _capm_regress(self, window=252, half_life=63):
        y = self.dailyret
        x = self.mktret
        beta, alpha, sigma = self._rolling_regress(y, x, window=window, half_life=half_life)
        return beta, alpha, sigma

    def BETA(self):
        self.beta, self.alpha, self.hsigma = self._capm_regress(window=252, half_life=63)
        return self.beta

    def HSIGMA(self):
        return self.hsigma

    def HALPHA(self):
        return self.alpha

    def DASTD(self):
        dastd = self._rolling(self.dailyret, window=252,half_life=42, func_name='std')
        return dastd
    #
    #
    #
    #
    #
    #
    #
    #
    #
    # def CMRA(self, version=VERSION):
    #     stock_ret = self._fillna(self.changepct / 100, 0)
    #     if version == 6:
    #         ret = np.log(1 + stock_ret)
    #     elif version == 5:
    #         index_ret = self._get_benchmark_ret()
    #         index_ret = np.log(1 + index_ret)
    #         index_ret, stock_ret = self._align(index_ret, stock_ret)
    #         ret = np.log(1 + stock_ret).sub(index_ret[BENCHMARK], axis=0)
    #     cmra = self._pandas_parallelcal(ret, self._cal_cmra, args=(12, 21, version),
    #                                     window=252, axis=0).T
    #     return cmra
result = Volatility()

db.close()
index.close()


