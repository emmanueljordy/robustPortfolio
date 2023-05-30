import numpy as np 
import pandas as pd 
import os
original_dir = os.getcwd()
os.chdir('..\\Utils')
from helper_funcs import *
os.chdir(original_dir)
import scipy








class backtest_crypto_portfolios(): 
    """
    Class to run the bactest as explained in the article.
    
    Parameters
    ----------
    
        -portfolio_list : list of object  of class portfolio, default=None 
            represents the portfolio to backtest
            
        - data:  pandas DataFrame, default=None 
          data containing the price or returns for the whole backtesting period 
          
        volumes : pandas DataFrame, 
            data containing the trading volumes for each of the cryptocurrencies 
            
        K : int, 
            The K highest cryptos in mean trading volume will be used during each training period.  
            
          
        - is_price_data: bool, default=True
            If True, the inputted data represents prices, if False, the data represents returns
            
        - risk_free_rate: int or pandas DataFrame, 
            risk free rate data to be used 
        
        - dates: pandas DataFrame, containing datetime object 
            date corresponding to each row in the input data
            
        -train_window_size: int, Number of observations to be used for training, default=750 (250*3)
            Number of observations to be used to estimate the weights
            
        -test_window_size: int, default=250
            Number of observations to be used in the rolling window for estimating portfolio characteristics
            
        -rebalance_window_size: int, default=20
            Time before  the portfolio will be rebalanced in the test window size 
            
        verbose: bool, 
            If to print during the backtest
        
    
    
    
    Attributes
    ----------
    
        
    
    
    
    """
    
    def __init__(self, 
                portfolio_list = None, 
                returns_data = None, 
                log_returns_data = None,  
                tradA_data = None,
                volumes = None,
                K = 20,
                is_price_data = True, 
                risk_free_rate = 0.0, 
                dates = None, 
                train_window_size = 252,
                test_window_size = 30,
                rebalance_window_size=30,
                 transaction_cost = 0.0001 *25,
                 verbose = True
                ): 
        self.portfolio_list=portfolio_list
        self.returns_data= returns_data
        self.log_returns_data = log_returns_data
        self.tradA_data = tradA_data
        self.volumes = volumes 
        self.K = K 
        self.is_price_data= is_price_data
        self.dates=dates
        self.risk_free_rate= risk_free_rate
        self.train_window_size= train_window_size
        self.test_window_size= test_window_size
        self.rebalance_window_size= rebalance_window_size
        self.transaction_cost = transaction_cost
        self.verbose = verbose
        
        if not(isinstance(self.risk_free_rate,(int,float,pd.DataFrame))):
            raise(ValueError('only int, floats, or pandasDataframe are allowed for the risk free rate'))
            
        if(isinstance(self.risk_free_rate, pd.DataFrame) and self.risk_free_rate.shape[1]>1):
            raise(ValueError('only one dimensional  pandasDataframe are allowed for the risk free rate'))
            
        if(isinstance(self.risk_free_rate, (int, float))):
            self.adjust_risk_free_length =   - is_price_data
            self.risk_free_rate=pd.DataFrame(np.repeat(self.risk_free_rate,self.returns_data.shape[0] +self.adjust_risk_free_length))
            self.risk_free_rate.index = self.dates
    
    def run_backtest(self): 
        """
        run the bactest
        """
        self.portfolio_names = [current_portfolio.name for current_portfolio in self.portfolio_list]
        self.train_weights_list = {}
        self.arithmetic_mean_returns_list = {}
        self.geometric_mean_returns_list = {}
        self.realised_volatility_list = {}
        self.realised_skewness_list ={}
        self.realised_kurtosis_list = {}
        self.sharpe_ratio_list = {}
        self.value_at_risk_list ={}
        self.dates_list=[]
        self.dates_start_list = []
        self.turnover_list = {}
        self.adjusted_sharpe_ratio_list = {}
        self.maximum_drawdown_list = {}
        self.certainty_equivalence_list = {}
        self.concentration_list ={}
        self.sum_squared_weights_list = {}
        self.test_realised_returns_list ={}
        self.sortino_ratio_list = {}
        self.omega_ratio_list = {}
        self.asset_names = []
        
        
        backtest_items_list = [self.arithmetic_mean_returns_list,self.geometric_mean_returns_list,
                             self.realised_volatility_list,self.realised_skewness_list,
                             self.realised_kurtosis_list,self.sharpe_ratio_list,
                             self.value_at_risk_list,self.dates_list,
                             self.turnover_list]
        
        N_obs = self.returns_data.shape[0]
        number_of_rebalances_in_test_window = round(self.test_window_size/self.rebalance_window_size)
        
        
        for start_train_index in range(0,N_obs-self.train_window_size-self.test_window_size,self.test_window_size):
            dates_train = self.dates[start_train_index: start_train_index+self.train_window_size+1]
            if self.verbose:
                print('start_train_index: ',start_train_index)
                print('start_train_date: ', dates_train[0])
                print('end_train_date: ', dates_train[-1])
            
            if not self.tradA_data is None:
                
                train_data_period = self.log_returns_data.iloc[start_train_index: start_train_index+self.train_window_size+1,:]
                tradA_data_period = self.tradA_data.loc[ dates_train[0]:  dates_train[-1]]
                train_volumes_period = self.volumes.iloc[start_train_index: start_train_index+self.train_window_size+1,:]
                train_data = train_data_period[get_highest_trading_volume(train_volumes_period,self.K)]
                train_data = pd.merge( train_data,tradA_data_period,left_index=True, right_index=True)
                self.asset_names.append(train_data.columns.values)
                risk_free_train = self.risk_free_rate.loc[self.risk_free_rate.index.isin(train_data.index)]
            
            else: 
                train_data_period = self.log_returns_data.iloc[start_train_index: start_train_index+self.train_window_size+1,:]
                train_volumes_period = self.volumes.iloc[start_train_index: start_train_index+self.train_window_size+1,:]
                train_data = train_data_period[get_highest_trading_volume(train_volumes_period,self.K)]
                self.asset_names.append(train_data.columns.values)
                risk_free_train = self.risk_free_rate.loc[self.risk_free_rate.index.isin(train_data.index)]
                
                
            for current_portfolio in self.portfolio_list:
                print(current_portfolio.name)
                current_portfolio.is_price_data = self.is_price_data
                current_portfolio.data = train_data
                current_portfolio.asset_names = train_data.columns.values
                current_portfolio.risk_free_rate = risk_free_train
                current_portfolio.compute_returns()
                current_portfolio.compute_excess_returns()
                current_portfolio.previous_weights = current_portfolio.weights
                current_portfolio.optimize()
                print(train_data.columns.values)
                print(current_portfolio.weights)
                current_portfolio.compute_sum_squared_weights()
                
                
                if current_portfolio.name in  self.train_weights_list:
                    self.train_weights_list[current_portfolio.name].append(current_portfolio.weights)
                else:
                    self.train_weights_list[current_portfolio.name] =  [current_portfolio.weights]   
                    
                if current_portfolio.previous_weights is not None:
                    
                    
                    if current_portfolio.name in  self.turnover_list:
                        self.turnover_list[current_portfolio.name].append(compute_turnover(self.asset_names[-1],current_portfolio.weights,self.asset_names[-2],current_portfolio.previous_weights ))
                    else:
                        self.turnover_list[current_portfolio.name] =  [compute_turnover(self.asset_names[-1],current_portfolio.weights,self.asset_names[-2],current_portfolio.previous_weights)]

                    if current_portfolio.name in  self.sum_squared_weights_list:
                        self.sum_squared_weights_list[current_portfolio.name].append(current_portfolio.sum_squared_weights)
                    else:
                        self.sum_squared_weights_list[current_portfolio.name] =  [current_portfolio.sum_squared_weights]
                    
                    
            for rebalance_number in range(number_of_rebalances_in_test_window):
                print('rebalance number : ', rebalance_number)
                start_test_index = start_train_index+self.train_window_size+1 
                
                test_data_period = self.returns_data.iloc[start_test_index+(rebalance_number*self.rebalance_window_size):
                                         start_test_index+((rebalance_number+1)*self.rebalance_window_size),: ]
                
                dates_test = self.dates[start_test_index+(rebalance_number*self.rebalance_window_size):
                                         start_test_index+((rebalance_number+1)*self.rebalance_window_size)]
                print('start_test date: ', dates_test[0])
                print('end_test_date: ', dates_test[-1])
                
                if not self.tradA_data is None:
                    tradA_data_test = self.tradA_data.loc[ dates_test[0]:  dates_test[-1]]
                    test_data = test_data_period[get_highest_trading_volume(train_volumes_period,self.K)]
                    test_data = pd.merge( test_data,tradA_data_test,left_index=True, right_index=True)
                    risk_free_test = self.risk_free_rate.loc[self.risk_free_rate.index.isin(test_data.index)]
                
                else: 
                    test_data = test_data_period[get_highest_trading_volume(train_volumes_period,self.K)]
                    risk_free_test = self.risk_free_rate.loc[self.risk_free_rate.index.isin(test_data.index)]
                    
                    
                self.dates_list.append(dates_test)
                self.dates_start_list.append(dates_test[0])
                
                for current_portfolio in self.portfolio_list: 
                    current_portfolio.data = test_data
                    current_portfolio.risk_free_rate = risk_free_test
                    current_portfolio.compute_returns()
                    current_portfolio.compute_excess_returns()
                    current_portfolio.compute_realised_returns()
                    if current_portfolio.previous_weights is not None:
                        current_portfolio.realised_returns[0,0] = current_portfolio.realised_returns[0,0] -  ((self.turnover_list[current_portfolio.name][-1])*(self.transaction_cost))
                    current_portfolio.compute_portfolio_characteristics(include_realised_returns = False)
                                                        
                    if current_portfolio.name in  self.arithmetic_mean_returns_list:
                        self.arithmetic_mean_returns_list[current_portfolio.name].append(current_portfolio.arithmetic_mean_return)
                    else:
                        self.arithmetic_mean_returns_list[current_portfolio.name] =  [current_portfolio.arithmetic_mean_return]
                        
                    if current_portfolio.name in  self.geometric_mean_returns_list:
                        self.geometric_mean_returns_list[current_portfolio.name].append(current_portfolio.geometric_mean_return)
                    else:
                         self.geometric_mean_returns_list[current_portfolio.name] =  [current_portfolio.geometric_mean_return]
                        
                    if current_portfolio.name in  self.realised_volatility_list:
                        self.realised_volatility_list[current_portfolio.name].append(current_portfolio.realised_volatility)
                    else:
                        self.realised_volatility_list[current_portfolio.name] =  [current_portfolio.realised_volatility]
                        
                    if current_portfolio.name in  self.realised_skewness_list:
                        self.realised_skewness_list[current_portfolio.name].append(current_portfolio.realised_skewness)
                    else:
                        self.realised_skewness_list[current_portfolio.name] =  [current_portfolio.realised_skewness]
                        
                    if current_portfolio.name in  self.realised_kurtosis_list:
                        self.realised_kurtosis_list[current_portfolio.name].append(current_portfolio.realised_kurtosis)
                    else:
                        self.realised_kurtosis_list[current_portfolio.name] =  [current_portfolio.realised_kurtosis]
                    
                    if current_portfolio.name in  self.sharpe_ratio_list:
                        self.sharpe_ratio_list[current_portfolio.name].append(current_portfolio.sharpe_ratio)
                    else:
                        self.sharpe_ratio_list[current_portfolio.name] =  [current_portfolio.sharpe_ratio]
                        
                    if current_portfolio.name in  self.adjusted_sharpe_ratio_list:
                        self.adjusted_sharpe_ratio_list[current_portfolio.name].append(current_portfolio.adjusted_sharpe_ratio)
                    else:
                        self.adjusted_sharpe_ratio_list[current_portfolio.name] =  [current_portfolio.adjusted_sharpe_ratio]
                        
                    if current_portfolio.name in  self.maximum_drawdown_list:
                        self.maximum_drawdown_list[current_portfolio.name].append(np.max(np.abs(current_portfolio.max_drawdown))[0])
                    else:
                        self.maximum_drawdown_list[current_portfolio.name] =  [np.max(np.abs(current_portfolio.max_drawdown))[0]]
                        
                    if current_portfolio.name in  self.certainty_equivalence_list:
                        self.certainty_equivalence_list[current_portfolio.name].append(current_portfolio.certainty_equivalence)
                    else:
                        self.certainty_equivalence_list[current_portfolio.name] =  [current_portfolio.certainty_equivalence]
                        
                    if current_portfolio.name in  self.test_realised_returns_list:
                        self.test_realised_returns_list[current_portfolio.name].append(current_portfolio.realised_returns)
                    else:
                        self.test_realised_returns_list[current_portfolio.name] =  [current_portfolio.realised_returns]
                    
                    if current_portfolio.name in  self.sortino_ratio_list:
                        self.sortino_ratio_list[current_portfolio.name].append(current_portfolio.sortino_ratio)
                    else:
                        self.sortino_ratio_list[current_portfolio.name] =  [current_portfolio.sortino_ratio]
                        
                    if current_portfolio.name in  self.omega_ratio_list:
                        self.omega_ratio_list[current_portfolio.name].append(current_portfolio.omega_ratio)
                    else:
                        self.omega_ratio_list[current_portfolio.name] =  [current_portfolio.omega_ratio]

                        

        
        #create the relevant tables
        self.arithmetic_mean_returns_df = pd.DataFrame(self.arithmetic_mean_returns_list,index=self.dates_start_list)
        self.geometric_mean_returns_df = pd.DataFrame(self.geometric_mean_returns_list,index=self.dates_start_list)
        self.realised_volatility_df = pd.DataFrame(self.realised_volatility_list,index=self.dates_start_list)
        self.realised_skewness_df = pd.DataFrame(self.realised_skewness_list,index=self.dates_start_list)
        self.realised_kurtosis_df = pd.DataFrame(self.realised_kurtosis_list,index=self.dates_start_list)
        self.sharpe_ratio_df = pd.DataFrame(self.sharpe_ratio_list,index=self.dates_start_list)
        self.adjusted_sharpe_ratio_df = pd.DataFrame(self.adjusted_sharpe_ratio_list,index=self.dates_start_list)
        self.certainty_equivalence_df = pd.DataFrame(self.certainty_equivalence_list,index=self.dates_start_list)
        self.maximum_drawdown_df = pd.DataFrame(self.maximum_drawdown_list,index=self.dates_start_list)
        self.sortino_ratio_df = pd.DataFrame(self.sortino_ratio_list,index=self.dates_start_list)
        self.omega_ratio_df = pd.DataFrame(self.omega_ratio_list,index=self.dates_start_list)
        self.turnover_df = pd.DataFrame(self.turnover_list)
        self.sum_squared_weights_df = pd.DataFrame(self.sum_squared_weights_list)

        return
    
    def describe_arithmetic_returns(self):
        """
        return a description of the  arithmetic returns vector 
        """
        return(self.arithmetic_mean_returns.describe(percentiles=[0.1,0.25,0.5,0.75]))
    
    def backtest_mean_results(self):
        """
        returns the mean of the different performance measures of the given  backtest
        """
        indexes =  self.geometric_mean_returns_df.columns.values.tolist()
        mean_perfs_dict = ({'ASR' : self.adjusted_sharpe_ratio_df.mean(axis=0).values.tolist(),
                            'MR' : self.arithmetic_mean_returns_df.mean(axis=0).values.tolist(),
                            'CEQ': self.certainty_equivalence_df.mean(axis=0).values.tolist(),
                            'MD' : self.maximum_drawdown_df.mean(axis=0).values.tolist(),
                            'SSPW': self.sum_squared_weights_df.mean(axis=0).values.tolist(),
                            'SR': self.sharpe_ratio_df.mean(axis=0).values.tolist(),
                            'OR': self.omega_ratio_df.mean(axis=0).values.tolist(),
                            'SoR': self.sortino_ratio_df.mean(axis=0).values.tolist(),
                            'Kurtosis' : self.realised_kurtosis_df.mean(axis=0).values.tolist(),
                            'Skewness' : self.realised_skewness_df.mean(axis=0).values.tolist()})
                            
                            
                            
        
        self.mean_perfs_df = pd.DataFrame.from_dict(mean_perfs_dict)
        self.mean_perfs_df.index = indexes
        return(self.mean_perfs_df )
    
    def out_of_sample_results(self): 
        """
        returns the portfolio characteristics for the whole out of sample period
        """
        
        for current_portfolio in self.portfolio_list:
            current_portfolio.realised_returns = np.array(self.test_realised_returns_list[current_portfolio.name]).reshape((-1,1))
            current_portfolio.compute_portfolio_characteristics(include_realised_returns=False)
            
            
        indexes = [current_portfolio.name for current_portfolio in self.portfolio_list]
        ASR = [current_portfolio.adjusted_sharpe_ratio for current_portfolio in self.portfolio_list]
        MR = [current_portfolio.arithmetic_mean_return for current_portfolio in self.portfolio_list]
        CEQ = [current_portfolio.certainty_equivalence for current_portfolio in self.portfolio_list ]
        MD = [ 100*np.min(current_portfolio.max_drawdown)[0] for current_portfolio in self.portfolio_list]
        TO = self.turnover_df.mean(axis=0).values.tolist()
        SSPW = self.sum_squared_weights_df.mean(axis=0).values.tolist()
        SR = [current_portfolio.sharpe_ratio for current_portfolio in self.portfolio_list ]
        Kurto = [current_portfolio.realised_kurtosis for current_portfolio in self.portfolio_list ]
        Skew= [current_portfolio.realised_skewness for current_portfolio in self.portfolio_list]
        Vol = [current_portfolio.realised_volatility for current_portfolio in self.portfolio_list ]
        Geom_mean =[100*current_portfolio.geometric_mean_return for current_portfolio in self.portfolio_list]
        sortino_ratio = [current_portfolio.sortino_ratio for current_portfolio in self.portfolio_list ]
        omega_ratio = [current_portfolio.omega_ratio for current_portfolio in self.portfolio_list ]
        
        
        out_of_sample_dict = ({'ASR' : ASR,
                            'CEQ': CEQ,
                            'MD' : MD,
                            'TO' : TO,
                            'SSPW': SSPW,
                            'SR': SR,
                            'Kurtosis' : Kurto,
                            'Skewness' : Skew,
                              'MR' : MR, 
                              'OR' : sortino_ratio,
                              'SoR': omega_ratio,  
                            'Geom_mean':Geom_mean,
                              'Vol': Vol})
        
        self.out_of_sample_df = pd.DataFrame.from_dict(out_of_sample_dict)
        self.out_of_sample_df.index = indexes
        return(self.out_of_sample_df)
    
    
    def plot_backtest_max_drawdown(self):
        """
        returns a plot of the maximum drawdown during the backtest period 
        """
            
        for current_portfolio in self.portfolio_list:
            current_portfolio.dates = [item for sublist in self.dates_list for item in sublist]
            current_portfolio.plot_max_drawdown()
            
        plt.legend(loc="lower right")
            
        return
    
    def plot_backtest_equity_curve(self):
        """
        plot the equity curves during the bcktest period for each portfolio 
        """
        for current_portfolio in self.portfolio_list:
            current_portfolio.dates = [item for sublist in self.dates_list for item in sublist]
            current_portfolio.plot_equity_curve()
            
        plt.legend(loc="upper left")

        return
    
    def plot_weights_evolution(self,portfolio_name):
        """
        returns the evolution of each assets weights throughout the backtest for the required portfolio
        
        portfolio_name : str, name of the portfolio must be one of the baktestested portfolios
        """
        color=iter(cm.rainbow(np.linspace(0,1,self.returns_data.shape[1])))
        if portfolio_name in self.portfolio_names:
            for asset_index in range(self.returns_data.shape[1]):
                c=next(color)
                current_asset_weights=[weight[asset_index] for weight in self.train_weights_list[portfolio_name]]
                plt.plot(current_asset_weights, label=self.returns_data.columns.values[asset_index],c=c)
            plt.legend(loc="upper left")    
            plt.show()
        else:
            print(portfolio_name, ' is not in the list of portfolios')
            
        return
