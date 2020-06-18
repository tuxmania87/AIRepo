import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import acf,pacf
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
import scipy.stats as st
import math
import warnings 
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.model_selection import TimeSeriesSplit
warnings.simplefilter('ignore')
import sys
import time




class BobSARIMAX:
    def __init__(self, data, col=0, train_split=0.8):
        self.model = None
        self.transformation = None
        self.col = data.columns[col]
        self.data = data
        
        self.sentinel = int(len(data.index) * train_split)
        self.train_data = data.iloc[:self.sentinel]
        self.test_data = data.iloc[self.sentinel:]
        self.fitresult = None
        
    def mean_absolute_percentage_error(self,y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    def automode(self, win, adfuller_treshhold = 0.05, forcemode = -1):
        
        timeseries = self.train_data
        col = self.col
        
        # First evaluate if we can achieve stationarity by 
        # inspecting adfueller test
        best_trans, pval_adfuller = self.chose_best_transformation(win)
        
        if pval_adfuller > adfuller_treshhold:
            print("No Stationarity could be achieved by diff/difflog transformation. Best p-value: {}".format(pval_adfuller))
            pass
        
        print("Best transformation: {} with p-value {}".format(best_trans, pval_adfuller))
        
        if forcemode >= 0:
            print("Overwrite with set transformation: {}".format(forcemode))
            best_trans = forcemode
        
        
        if best_trans == 0:
            df = timeseries
        elif best_trans == 1:
            df = timeseries.diff().dropna()
        elif best_trans == 2:
            df = np.log(timeseries).diff().dropna()
            
        # plot choosen model
        self.plots(df)
                
        # find best q  acf
        
        lag_acf = acf(df, nlags=min(40,len(df.index)-2))
        lag_pacf = pacf(df, nlags=min(40,len(df.index)-2),method='ols')
        unten = -1.96/np.sqrt(len(df))
        oben = 1.96/np.sqrt(len(df))
        
        lag_acf = lag_acf[1:]
        q = (np.argwhere((lag_acf > oben) | (lag_acf < unten)) + 1)[0][0]
        
        lag_pacf = lag_pacf[1:]
        p = (np.argwhere((lag_pacf > oben) | (lag_pacf < unten)) + 1)[0][0]
        
        temp_arr = np.argsort(lag_acf)
        temp_arr = temp_arr[::-1]

        seasonal = temp_arr[1] - temp_arr[0]
        
        print("Suggested p: {} q: {} s: {}".format(p,q,seasonal))

        if best_trans == 2:
            df = np.log(timeseries).dropna()
        else:
            df = timeseries
        
        print("Run grid search...")
        
        # set parameters
        d = [0,1]
        parr = np.unique([max(p-1,0), p, p+1]).tolist()
        qarr = np.unique([max(q-1,0), q, q+1]).tolist()
        Parr = [0,1,2]
        Qarr = [0,1,2]
        s = [seasonal]
        
        
        minp,mind,minq,minP,minD,minQ,mins = self.cust_sarimax(df,parr,d,qarr,Parr,d,Qarr,s,maxorder=8)
        
        #minp,mind,minq,minP,minD,minQ,mins = 1,1,0,1,0,1,12
        
        if(minp < 200):
            print("Optimal SARIMA Model found!")
        else:
            return
    
        print("Model set: order=({},{},{}), seasonal_order=({},{},{},{})".format(minp,mind,minq,minP,minD,minQ,mins))
        self.model = SARIMAX(df, order=(minp,mind,minq) , seasonal_order=(minP,minD,minQ,mins), enforce_invertibility= False, enforce_stationarity=False)
        self.transformation = best_trans
        
    def fit(self):
        if self.model == None:
            print("No model set. Use setModel() or automode() to set a model first.")
            return
        self.fitresult = self.model.fit(disp=0)
    
    def cross_val_score(self,splits=5):
        # needs to be fitted
        if self.fitresult == None:
            print("You need a fitted model to be able to cross validate.")
            return
        
        tss = TimeSeriesSplit(splits)
        
        if self.transformation == 2:
            scaling_function = np.log
            inverse_scaling_function = np.exp
        else:
            scaling_function = lambda x: x 
            inverse_scaling_function = lambda x: x 
            

        threedays = []
        sixdays = []
        allldays = []
        
        
        # store original model parameters 
        p,d,q = self.model.order
        P,D,Q,s = self.model.seasonal_order
        
        
        
        for train_index, test_index in tss.split(self.data):
            
            df_temp_train = self.data.iloc[train_index]
            df_temp_test = self.data.iloc[test_index]
            
            #create model and fit
            model = SARIMAX(scaling_function(df_temp_train),order=(p,d,q), seasonal_order=(P,D,Q,s), enforce_invertibility=False, enforce_stationarity=False)
            res = model.fit(disp=0)
            
            # predict 
            df_pred = inverse_scaling_function(res.forecast(len(df_temp_test.index)))
            df_pred = pd.DataFrame(df_pred)
            df_pred.columns = ["forecasts"]
            
            #plot 
            pd.concat([df_temp_train, df_temp_test]).join(df_pred,how="left").plot(figsize=(15,7))
            
            #calculate next 3 forecast rmse
            threedays.append(np.sqrt(np.mean((df_pred.values[:3]-df_temp_test.values[:3])**2)))
            
            #calculate next 6 forecast rmse
            sixdays.append(np.sqrt(np.mean((df_pred.values[:6]-df_temp_test.values[:6])**2)))
            
            #calculate all
            allldays.append(np.sqrt(np.mean((df_pred.values-df_temp_test.values)**2)))
            
        print("3-days RMSE: {} {}".format(np.mean(threedays),threedays))
        print("6-days RMSE: {} {}".format(np.mean(sixdays),sixdays))
        print("all-days RMSE: {} {}".format(np.mean(allldays),allldays))
        
    
    def predict(self, timesteps):
        
        X = self.train_data
        
        if self.transformation == 2:
            scaling_function = np.exp
        else:
            scaling_function = lambda x: x
        
        confidence = scaling_function(self.fitresult.get_forecast(timesteps).conf_int(alpha=.05))
        confidence.columns = ["lowerCI","upperCI"]
        
        series_SARIMAX = scaling_function(self.fitresult.forecast(timesteps))
        
        df_sarimax = pd.DataFrame(series_SARIMAX)
        df_sarimax.columns = ["forecasted_values"]
        
        df_sarimax =  df_sarimax.join(confidence)
        
        df_alltogether = self.data[[self.col]].join(df_sarimax, how="outer")
        # cut alltogether only to predicted time stamp
        
        df_alltogether = df_alltogether.loc[:df_sarimax.index.max()]
        
        df_alltogether[[self.col, "forecasted_values"]].plot(figsize=(20,10))
        plt.fill_between(df_alltogether.index, df_alltogether.lowerCI, df_alltogether.upperCI, color="lightgrey")
        
        #for rmse calculation consider inner join from forecasts with data
        df_rsmecalc = df_sarimax.join(self.data, how="inner")
        print("RMSE", np.sqrt(np.mean((df_rsmecalc["forecasted_values"].values-df_rsmecalc[self.col].values)**2)))
        print("MAPE", self.mean_absolute_percentage_error(df_rsmecalc["forecasted_values"].values,df_rsmecalc[self.col].values))
        
        
        return df_sarimax
        
    def setModel(self, X,p,d,q,P,D,Q,s,logtransformed=False):
        if logtransformed:
            self.transformation = 2
            _df = np.log(X).dropna()
        else:
            self.logtransformed = 0
            _df = X
            
        self.model = SARIMAX(_df,order=(p,d,q,), seasonal_order=(P,D,Q,s), enforce_invertibility=False, enforce_stationarity=False)
    
    def chose_best_transformation(self, win):
        
        # try original, diff and logdiff
        results = []
        
        # original
        pval = self.test_adfuller(self.train_data,self.col,win)["p-value"]
        results.append(pval)
        
        # diff
        pval = self.test_adfuller(self.train_data.diff().dropna(), self.col, win)["p-value"]
        results.append(pval)
        
        # logdiff 
        pval = self.test_adfuller(np.log(self.train_data).diff().dropna(), self.col, win)["p-value"]
        results.append(pval)
        
        return np.argmin(results), results[np.argmin(results)]
    
    def test_adfuller(self, timeseries, col, win):
        import pandas as pd
        dftest = adfuller(timeseries[col], autolag="AIC")
        
        dfoutput = pd.Series(dftest[0:4], index = ["Test Statistic","p-value","#Lags Used","Number of Observations Used"])
        for key, value in dftest[4].items():
            dfoutput["Critical Value (%s)"%key] = value
        return dfoutput
    
    # functions
    def test_stationarity(self,timeseries, col, win):
        movingAverage = timeseries.rolling(window=win).mean()
        movingSTD = timeseries.rolling(window=win).std()
    
        orig = plt.plot(timeseries, color="blue", label="Original")
        mean = plt.plot(movingAverage, color ="red", label="Rolling Mean")
        std = plt.plot(movingSTD, color = "black", label = "Rolling Std")
        plt.legend(loc="best")
        plt.title("Rolling Mean & Standard Deviation")
        plt.show(block=True)
    
        print(self.test_adfuller(timeseries,col,win))
        
    def plots(self,df):
        
        ax1 = plt.subplot(311)
        
        df.plot(ax=ax1)
        
        #ax2 = plt.subplot(312)
        #plot_acf(df, lags=40, ax=ax2, zero=False)
        
        #ax3 = plt.subplot(313)
        #plot_pacf(df, lags=40, ax=ax3, zero=False)
        
        lag_acf = acf(df, nlags=min(40,len(df.index)-2))
        lag_pacf = pacf(df, nlags=min(40,len(df.index)-2),method='ols')
        
        ax2 = plt.subplot(312)
        ax2.stem(lag_acf)
        ax2.axhline(y=0,linestyle="--",color="gray")
        ax2.axhline(y=-1.96/np.sqrt(len(df)), linestyle="--",color="gray")
        ax2.axhline(y=1.96/np.sqrt(len(df)), linestyle="--",color="gray")
        ax2.title.set_text("ACF")
        
        
        ax3 = plt.subplot(313)
        ax3.stem(lag_pacf)
        ax3.axhline(y=0,linestyle="--",color="gray")
        ax3.axhline(y=-1.96/np.sqrt(len(df)), linestyle="--",color="gray")
        ax3.axhline(y=1.96/np.sqrt(len(df)), linestyle="--",color="gray")
        ax3.title.set_text("PACF")
        
        f = plt.gcf()
        f.set_figheight(15)
        f.set_figwidth(15)
        
        
    def cust_sarimax(self,df,p,d,q,P,D,Q,s,maxorder=6):
        min = 200000000
        minp=minq=mind=minD=minP=minQ=mins = 2000000
        strmin = ""
        it = 0
        allLen = len(p) * len(q) * len(d) * len(P) * len(Q) * len(D) * len(s)
        print("to test {} combinations...".format(allLen))
        startTime = time.time()
        for _p in p:
            for _q in q:
                for _d in d:
                    for _D in D:
                        for _P in P:
                            for _Q in Q:
                                for _s in s:
                                    try:
                                        if _p + _q + _d + _P + _D + _Q <= maxorder:
                                            model = SARIMAX(df, order=(_p,_d,_q) , seasonal_order=(_P,_D,_Q,_s), enforce_invertibility= False, enforce_stationarity=False)
                                            model_result = model.fit(disp=0)
                                            it += 1
                                            SSE = np.sum(model_result.resid**2)
                                            pvals = acorr_ljungbox(model_result.resid, lags=math.ceil(math.log(len(model_result.resid))))[1]
                                            curstr = str(_p)+" "+str(_d)+" "+str(_q)+" "+str(_P)+" "+str(_D)+" "+str(_Q)+" "+str(_s)+": AIC: "+str(model_result.aic)+" SSE: "+str(SSE)+" p-value: "+str(pvals[-1])
                                            if pvals[-1] > 0.05:
                                                #print("{} p {}".format(curstr,pvals[-1]))
                                                pass
                                            if model_result.bic < min and pvals[-1] > 0.05:
                                                min = model_result.aic
                                                strmin = curstr
                                                minp = _p 
                                                minq = _q
                                                mind = _d
                                                minD = _D
                                                minP = _P
                                                minQ = _Q
                                                mins = _s
                                            # progress print
                                            
                                            #calulate seconds per item
                                            spi = int((time.time() - startTime) / it)
                                            
                                            # rest time needed 
                                            remainingSeconds = (allLen - it) * spi
                                            
                                            print("\rGrid Searching... {} %  Remaining Seconds {}".format( int((float(it) / (allLen) ) * 10000) / 100 , remainingSeconds  ), end='')
                                            sys.stdout.flush()
                                            
                                    except:
                                        foo="bar"
        #print("MIN: "+strmin)
        return minp,mind,minq,minP,minD,minQ,mins