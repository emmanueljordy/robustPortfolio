import numpy as np
import pandas as pd
import cython
from cvxopt import matrix
from cvxopt.blas import dot 
from cvxopt.solvers import qp, options 
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.base import RegressorMixin,BaseEstimator,TransformerMixin, defaultdict
import scipy




class Portfolio(_BaseComposition,BaseEstimator,TransformerMixin,RegressorMixin):
    """
    This class defines the structure to be followed by the different portfolio classes.
    
    necessary packages: pandas, numpy , scipy.stats
    TODO: 
        -implement the fact that the init_amount_invested can be an array
        - check if the initial amount is sufficient to be invested in the assets
        -solve the fact that sometimes weights are cvxopt and sometimes they are np arrays 
    
    Input parameters to class:
    
    asset_names: list, default=None
        tickers or names describing each asset
    
    data: pandas DataFrame, default=None
        price or return data.
        
    is_price_data: bool, default=True
        if True, then the data are prices if not the will be treated as returns
        
    data_frequency: string, one of 'daily', 'monthly', 'yearly';  default='daily'
        frequency of the  prices or return
    
    returns_frequency: string, one of 'daily', 'monthly', 'yearly';  default='daily'
        frequency of the  prices or return
        
    period_length: int, default = 250
                    number of days in a time period. can be used to annualize results
    
        
    risk_free_rate: float or  pandas DataFrame , default=0
         if int, assumes a constant risk_free_rate. if array, it needs to be of the same length as data 
         
    init_amount_invested: positive float , default=10000
        Amount to be invested in the portfolio at initialization
        
    portfolio_value: positive float,  default=None
        The current portfolio value
        
    weights: numpy.ndarray, default =None
        The array length should  be  the same as the asset_names
    
    
    realised_return: float, default=0.0
        Realised return of the portfolio 
        
    expected_return: float, default=0.0
        expected portfolio return
        
    realised_volatility: float, default=0.0
        realised volatility of the portfolio 
        
    expected_volatility: float, default=0.0
        expected portfolio volatility
        
    realised_skewness: float, default=0.0
        realised skewness of the portfolio returns 
        
    realised_kurtosis: float, default=0.0
        realised kurtosis of the portfolio returns
        
        
    sharpe_ratio: float, default=0.0
        sharpe ratio of the portfolio 
        
    value_at_risk : float, default=0.0
        Value at risk of the portfolio 
        
    expected_shortfall: float, default=0.0
        Expected shortfall of the portfolio. 
        
    time_taken: float, default=0.0
        time_taken for the optimization procedure. 
    
    """
    def __init__(self,
                 asset_names=None,
                 data=None,
                 is_price_data=True,
                 data_frequency='daily',
                 returns_frequency='daily',
                 period_length = 252,
                 risk_free_rate=0.0, 
                 init_amount_invested=1.0,
                 portfolio_value=0.0,
                 weights= None,
                 realised_returns=0.0,
                 expected_returns=0.0, 
                 realised_volatility=0.0, 
                 expected_volatility=0.0, 
                 realised_skewness=0.0, 
                 realised_kurtosis=0.0,
                 sharpe_ratio=0.0, 
                 value_at_risk=0.0,
                 expected_shortfall=0.0,
                 time_taken=0.0,
                 dates=None,
                 name='portf'
                ):
        self.asset_names=asset_names
        self.data = data
        self.is_price_data=is_price_data
        self.data_frequency=data_frequency
        self.returns_frequency=returns_frequency 
        self.period_length = period_length
        self.risk_free_rate=risk_free_rate
        self.init_amount_invested=init_amount_invested
        self.portfolio_value=portfolio_value
        self.weights= weights
        self.realised_returns=realised_returns
        self.expected_returns=expected_returns 
        self.realised_volatility=realised_volatility 
        self.expected_volatility=expected_volatility 
        self.realised_skewness=realised_skewness
        self.realised_kurtosis=realised_kurtosis 
        self.sharpe_ratio=sharpe_ratio 
        self.value_at_risk=value_at_risk
        self.expected_shortfall=expected_shortfall
        self.time_taken=time_taken
        self.name = name
        self.dates = dates
        
        self.list_data_frequency=['daily', 'weekly','monthly', 'quarterly', 'yearly']
        if not( str(self.data_frequency) in self.list_data_frequency):
            raise(ValueError('only data frequency allowed are daily, weekly,monthly, quarterly and yearly'))
        
        if not( self.returns_frequency in self.list_data_frequency):
            raise(ValueError('only data frequency allowed are daily, weekly,monthly, quarterly and yearly'))

        if self.init_amount_invested <0 : 
            raise(ValueError('please choose positive amount invested'))
            
        if not(isinstance(self.data,pd.DataFrame)):
            raise(ValueError('only pandasDataframe are allowed as data'))
            
        if not(isinstance(self.risk_free_rate,(int,float,pd.DataFrame))):
            raise(ValueError('only int, floats, or pandasDataframe are allowed for the risk free rate'))
            
        if(isinstance(self.risk_free_rate, pd.DataFrame) and self.risk_free_rate.shape[1]>1):
            raise(ValueError('only one dimensional  pandasDataframe are allowed for the risk free rate'))
            
     
    def compute_returns(self, method='realised'): 
        """
        TODO: complete function with different input frequencies. implement Different ways ie EWMA etc
        converts the price data to returns data  and if needed aggregates according to wanted frequency
        """
        if(self.is_price_data): 
            if(self.returns_frequency=='daily' and self.returns_frequency=='daily'):
                self.returns = self.data.pct_change(1)
                self.returns=self.returns.iloc[1:,:]
        else:
            self.returns=self.data
        return(self.returns)
     
    
    def compute_excess_returns(self): 
        """
        compute excess returns from the data  by subtracting the risk free rate  to the returns
        """
        if(isinstance(self.risk_free_rate, (int, float))):
            self.risk_free_rate=pd.DataFrame(np.repeat(self.risk_free_rate,self.returns.shape[0]))
        self.excess_returns = self.returns.sub(self.risk_free_rate.values.reshape(-1,1), axis=0)
        return(self.excess_returns)
        
    
    def compute_realised_returns(self): 
        """
        computes the realised returns of the portfolio for each row in the returns data
        """
        if self.weights is not None and self.returns is not None:
            self.realised_returns = np.matmul(self.excess_returns.values,np.array(self.weights).reshape(-1,1))
        else: 
            raise(ValueError('Missing weights'))
        
        return(self.realised_returns)
    
    def compute_arithmetic_mean_return(self):
        """
        computes the arithmetic mean return of the portfolio 
        """
        self.arithmetic_mean_return = np.mean(self.realised_returns*self.period_length)
        return(self.arithmetic_mean_return)
    
    def compute_geometric_mean_return(self):
        """
        computes the geometric mean return of the portfolio
        """
        self.geometric_mean_return = ((scipy.stats.mstats.gmean(1+self.realised_returns)-1)*self.period_length)[0]
        return(self.geometric_mean_return)
    
    def compute_pnl(self):
        """
        computes the pnl of the portfolio using the realised returns 
        """
        self.pnl= self.init_amount_invested* self.realised_returns
        
        return(self.pnl)
    
    def compute_equity_curve(self):
        """
        returns a vector corresponding to the equity curve of the portfolio 
        """
        self.equity_curve = pd.DataFrame(np.cumsum(np.concatenate((self.init_amount_invested,self.pnl),axis=None)))
        return(self.equity_curve)
    
    def compute_max_drawdown(self):
        """
        returns the maximum drawdown of the portfolio over the life of the investment
        """
        roll_max = self.equity_curve.cummax()
        self.period_drawdown = self.equity_curve/roll_max -1
        self.max_drawdown = self.period_drawdown.cummin()
        return(self.max_drawdown)    
            
    
    def compute_realised_volatility(self): 
        """
        TODO: implement different methods to compute realised volatility
        returns the realised volatility of the portfolio 
        """
        self.realised_volatility = np.std(self.pnl) *np.sqrt(self.period_length)
        return(self.realised_volatility)
        
    def compute_realised_skewness(self): 
        """
        returns the realised skewness of the portfolio returns 
        """
        self.realised_skewness = scipy.stats.skew(self.pnl,bias=False)[0]*(1/np.sqrt(self.period_length))
        return(self.realised_skewness)
        
    def compute_realised_kurtosis(self):
        """
        returns the realised  excess kurtosis of the portfolio returnss
        """
        self.realised_kurtosis= scipy.stats.kurtosis(self.pnl,bias=False)[0]*(1/self.period_length)
        return(self.realised_kurtosis)
    
    def compute_certainty_equivalence(self,gamma=1,rf=0):
        """
        computes the certainty equivalence 
        """
        self.certainty_equivalence = (self.arithmetic_mean_return-rf)-(0.5*gamma*self.realised_volatility)
        return(self.certainty_equivalence)
    
    def compute_sharpe_ratio(self): 
        """
        TODO: add the ability for geometric returns sharpe
        returns the sharpe ratio of the portfolio
        """
        self.sharpe_ratio = self.arithmetic_mean_return/self.realised_volatility
        return(self.sharpe_ratio)
    
    def compute_adjusted_sharpe_ratio(self):
        """
        returns the skewness and kurtosis adjusted sharpe ratio  of th eportfolio 
        """
        self.adjusted_sharpe_ratio = (self.sharpe_ratio *(1+((self.realised_skewness/6)*self.sharpe_ratio) -
                                                         (((self.realised_kurtosis)/24) *(self.sharpe_ratio**2))))
        return(self.adjusted_sharpe_ratio)
    
    def compute_VaR(self,confidence_level=0.99, method='nonparam', time_horizon=1):
        """
        TODO: implement different methods for computing the VaR 
        computes the value at risk of the portfolio using the realised returns
        """
        if(method=='nonparam'):
            self.value_at_risk= -1*np.quantile(self.pnl,q=1-confidence_level) *np.sqrt(time_horizon)
            return(self.value_at_risk)
        else:
            raise(ValueError('wrong method for the value at risk'))
            
    def compute_concentration(self, target=0.1):
        """
        computes the fraction of asset with weights greater than the target
        """
        
        
        self.concentration = np.mean( np.array(self.weights)>target) 
        
        return(self.concentration)
    
    def compute_sum_squared_weights(self):
        """
        computes the sum of the squared weights  of the portfolio
        """
        self.sum_squared_weights = np.sum(self.weights**2)
        
    def compute_LPM(self, MAR = 0, p= 1): 
        """
        compute the omega ratio based on the current return series
        inspired from Riskfolio-Lib
        
        MAR: float, optional 
            Minimum acceptable return 
        p : float, either 1 or 2
            1 is Omega ratio 
            2 is Sortino ratio 
        """
        a = np.array(self.realised_returns, ndmin=2)
        
        value =  a -MAR

        if p == 2:
            n = value.shape[0] - 1
        else:
            n = value.shape[0]

        value = np.sum(np.power(value[np.where(value >= 0)], p)) / n
        value = np.power(value, 1 / p).item()

        if p ==2: 
            self.sortino_ratio = value 
        else: 
            self.omega_ratio = value
        
            
    def compute_portfolio_characteristics(self,confidence_level=0.99,method='nonparam',VaR_time_horizon=1, include_realised_returns=True):
        """
        computes ortfolio characteristics : mean return, vol, skew, kurtosis, VaR, sharpe
        """
        if include_realised_returns:
            self.compute_returns()
            self.compute_excess_returns()
            self.compute_realised_returns()
        self.compute_pnl()
        self.compute_arithmetic_mean_return()
        self.compute_geometric_mean_return()
        self.compute_realised_volatility()
        self.compute_realised_skewness()
        self.compute_realised_kurtosis()
        self.compute_sharpe_ratio()
        self.compute_adjusted_sharpe_ratio()
        self.compute_VaR(confidence_level=0.99, method='nonparam',time_horizon=VaR_time_horizon)
        self.compute_equity_curve()
        self.compute_max_drawdown()
        self.compute_certainty_equivalence()
        self.compute_concentration()
        self.compute_sum_squared_weights()
        self.compute_LPM(p=1)
        self.compute_LPM(p=2)
        
        
        return
    
    def plot_equity_curve(self, dates=None):
        
        """
        plots the portfolio equity curve
        """
        start_adjustment = 1-self.is_price_data
        fig = plt.axes()
        if self.dates is None:
            fig.plot(self.equity_curve,label=self.name)
        else: 
            fig.plot(self.dates, self.equity_curve[start_adjustment:], label=self.name)
        return
        
    def plot_max_drawdown(self):
        """
        plots the portfolios maximum drawdown
        """
        fig = plt.axes()
        start_adjustment = 1-self.is_price_data
        if self.dates is None:
            
            fig.plot(self.period_drawdown,label=self.name)
            #fig.plot(self.max_drawdown)
        else : 
            fig.plot(self.dates,self.period_drawdown[start_adjustment:],label=self.name)
            #fig.plot(self.dates,self.max_drawdown)
        return
    
    
    
    
    
    
    
    
    def optimize(self):
        pass
    

    
#     @portfolio_value.setter
#     def init_amount_invested(self,new_portfolio_value):
#         if new_portfolio_value <0:
#             raise(ValueError('only positive potfolio values are allowed'))
#         else:
#             self._portfolio_value = new_portfolio_value
            
            
            
class MVPortfolio(Portfolio):
    """
    Generic class representing portfolio obtained through quadratic programming optimization
    please have a look at the cvxopt documentation for quadratic programming
    : http://cvxopt.org/userguide/coneprog.html#quadratic-programming
    it requires all input parameter from the portfolio class together with extra input parameters
    input parameters: 
    
    realised_risk_matrix:  numpy ndarray , default=None
        must be a square matrix of size p where p is the number of risky assets
    
    expected_risk_matrix :  numpy ndarray , default=None
        must be a square matrix of size p where p is the number of risky assets
        
    objective_inear_vector: numpy ndarray or cvxopt matrix, default=None
        cvxopt q vector in the objective
        
    weight_inequality_constraint_matrix: numpy ndarray or cvxopt matrix, default=None
        cvxopt G matrix
    
    weight_inequality_constraint_values: numpy ndarray or cvxopt matrix, default=None
        cvxopt h vector

    weight_equality_constraint_matrix: numpy ndarray or cvxopt matrix, default=None
        cvxopt A matrix
    
    weight_equality_constraint_values:
        cvxopt b vector
    """
    
    
    def __init__(self, 
                asset_names=None,
                 data=None,
                 is_price_data=True,
                 data_frequency='daily',
                 returns_frequency='daily',
                 risk_free_rate=0.0, 
                 period_length = 252,
                 init_amount_invested=1.0,
                realised_risk_matrix=None, 
                expected_risk_matrix=None,
                objective_linear_vector=None, 
                weight_inequality_constraint_matrix=None, 
                weight_inequality_constraint_values=None,
                weight_equality_constraint_matrix=None, 
                weight_equality_constraint_values=None,
                long_only = True,
                name='min_var',
                risk_method='sample_cov'):
            
            
            Portfolio.__init__(self,asset_names=asset_names,data=data,is_price_data=is_price_data,
                             data_frequency=data_frequency,returns_frequency=returns_frequency,
                             risk_free_rate=risk_free_rate, period_length=period_length,init_amount_invested=init_amount_invested)
            self.realised_risk_matrix=realised_risk_matrix
            self.expected_risk_matrix= expected_risk_matrix
            self.objective_linear_vector = objective_linear_vector
            self.n_assets= self.data.shape[1] 
            self.long_only = long_only
            self.weight_equality_constraint_matrix = weight_equality_constraint_matrix
            self.weight_equality_constraint_values = weight_equality_constraint_values
            self.name = name
            self.risk_method = risk_method
            
            
            
            
            
    def compute_realised_risk_matrix(self, method='sample_cov',rowvar=0):
        """
        TODO: implement different methods for the risk matrix and make a class of it 
        computes the risk matrix according to the chosen method.
        
        method: str, one of ['bcov', 'cov', 'dcov' 'mdd']; default='cov'
            mmethod to us for the estimation of the risk matrix 
            
        self.period_length: int, one of [1, 5,20,250]
            representing in what period to represent the risk matrix. 1 is daily, 5 is weekly, 20 monthly 250 is yearly
            
        rowvar: bool
            if the rows represent the variables
        """
        self.returns = self.compute_returns()
        self.excess_returns = self.compute_excess_returns()
        if(rowvar== True):
            excess_returns_values = self.excess_returns.values.T
        else:
            excess_returns_values =self.excess_returns.values
            
        
        list_methods = ['bcov', 'sample_cov', 'dcov', 'mdd','shrinkage']
        if not(str(method) in list_methods):
            raise(ValueError('The chosen method is not allowed'))
            
        if(method=='sample_cov'):
            self.realised_risk_matrix = np.cov(excess_returns_values,rowvar=rowvar)*self.period_length
        elif(method == 'shrinkage'):
            self.realised_risk_matrix = pd.DataFrame(ShrunkCovariance().fit(excess_returns_values).covariance_*self.period_length)
        elif(method=='dcov'):
            self.realised_risk_matrix = DCOV_pairwise(excess_returns_values) *self.period_length
        elif(method=='mdd'):
            self.realised_risk_matrix = MDD_pairwise(excess_returns_values)*self.period_length
        elif( method=='bcov'):
            self.realised_risk_matrix = BCOV_pairwise(excess_returns_values)*self.period_length
            
        else:
            raise(ValueError('implement the chosen method'))
            
        return(self.realised_risk_matrix)
        
            
    def add_weights_equality_constraint(self,constraint_matrix,constraint_values):
        """
        adds additional equality constraint to the optimization problem
        
        constraint_matrix: cvx opt matrix representing the constraints, denoted A in cvxopt doc 
        
        constraint_values: cvx opt vector representing the constraints, denoted b in cvxopt doc
        
        """
        if self.weight_equality_constraint_values is None and self.weight_equality_constraint_matrix is None : 
            self.weight_equality_constraint_values = constraint_values
            self.weight_equality_constraint_matrix = constraint_matrix
        else:
            self.weight_equality_constraint_values=matrix(np.concatenate([self.weight_equality_constraint_values, 
                                                                          constraint_values]))
            self.weight_equality_constraint_matrix = matrix(np.concatenate([self.weight_equality_constraint_matrix, 
                                                                          constraint_matrix]))
            return
            
    def add_weights_inequality_constraint(self, constraint_matrix, constraint_values): 
        """
        adds additional inequality constraints to the optimization problem
        constraint_matrix: cvx opt matrix representing the constraints, denoted G in cvxopt doc 
        
        constraint_values: cvx opt vector representing the constraints, denoted h in cvxopt doc
        
        """
        if self.weight_inequality_constraint_values is None and self.weight_inequality_constraint_matrix is None : 
            self.weight_inequality_constraint_values = constraint_values
            self.weight_inequality_constraint_matrix = constraint_matrix
        else:
            self.weight_inequality_constraint_values=matrix(np.concatenate([self.weight_inequality_constraint_values, 
                                                                          constraint_values]))
            self.weight_inequality_constraint_matrix = matrix(np.concatenate([self.weight_inequality_constraint_matrix, 
                                                                          constraint_matrix]))
            return
        
            
        
    def optimize(self):
        
        """
        Solve the quadratic programming problem specified by the portfolio 
        """
        self.n_assets= self.data.shape[1] 
        self.objective_linear_vector = matrix(0.0, (self.data.shape[1] ,1))# to be changed if mean variance portfolio 
            
        if self.long_only:
            self.weight_inequality_constraint_matrix = -matrix(np.eye(self.n_assets))
            self.weight_inequality_constraint_values = matrix(0.0, (self.n_assets ,1))
            self.weight_equality_constraint_matrix= matrix(np.ones((1,self.n_assets)))
            self.weight_equality_constraint_values = matrix(1.0)
        else: 
            self.weight_inequality_constraint_matrix=-matrix(np.eye(self.n_assets))
            self.weight_inequality_constraint_values=matrix(1.0, (self.n_assets ,1))
            
        self.realised_risk_matrix = self.compute_realised_risk_matrix(method=self.risk_method)
        #print(self.realised_risk_matrix.shape)
        #self.realised_risk_matrix = np.nan_to_num(self.realised_risk_matrix)
            
        options['show_progress'] = False
        sol = qp(matrix(self.realised_risk_matrix), self.objective_linear_vector,
                 self.weight_inequality_constraint_matrix, self.weight_inequality_constraint_values,
                 self.weight_equality_constraint_matrix, self.weight_equality_constraint_values)
        
        self.weights = np.round(np.array(sol['x']).reshape(-1,1),3)
        #print(dot(matrix(1.0, (self.n_assets ,1)),sol['x']))
        
        return(np.round(np.array(self.weights).reshape(-1,1)),3)
    
    
    
    
    
    
    
    
    
class EqualWeightPortfolio(Portfolio):
        """
        generic class implementing the equally weighted portfolio 
        
        please have a look at the cvxopt documentation for quadratic programming
        : http://cvxopt.org/userguide/coneprog.html#quadratic-programming
        it requires all input parameter from the portfolio class together with extra input parameters
        input parameters: 
    
        
        objective_inear_vector: numpy ndarray or cvxopt matrix, default=None
        cvxopt q vector in the objective
        
        weight_inequality_constraint_matrix: numpy ndarray or cvxopt matrix, default=None
        cvxopt G matrix
    
        weight_inequality_constraint_values: numpy ndarray or cvxopt matrix, default=None
        cvxopt h vector

        weight_equality_constraint_matrix: numpy ndarray or cvxopt matrix, default=None
        cvxopt A matrix
        
        weight_equality_constraint_values:
        cvxopt b vector
        
        """
        
        def __init__(self, 
                 asset_names=None,
                 data=None,
                 is_price_data=True,
                 data_frequency='daily',
                 returns_frequency='daily',
                 period_length = 252,
                 risk_free_rate=0.0, 
                 init_amount_invested=1.0,
                name='equal_weight'):
            
            
            Portfolio.__init__(self,asset_names=asset_names,data=data,is_price_data=is_price_data,
                             data_frequency=data_frequency,returns_frequency=returns_frequency,
                             risk_free_rate=risk_free_rate,period_length=period_length, init_amount_invested=init_amount_invested,name=name)

            self.returns = self.compute_returns()
            self.excess_returns = self.compute_excess_returns()
            self.name = name
            
        def optimize(self):
            """
            produces the equaly weighted weights were each asset receives weights 1/N with 
            N the number of assets
            """
            
            N = self.data.shape[1]
            
            self.weights = np.repeat(1/N,N)
            self.weights = np.array(self.weights).reshape(-1,1)
            return(np.array(self.weights).reshape(-1,1))
        
        
        
        
class max_sharpe_portfolio(Portfolio):
    """
    Generic class implementing the maximum sharpe ratio portfolio 
    please have a look at the cvxopt documentation for quadratic programming
        : http://cvxopt.org/userguide/coneprog.html#quadratic-programming
        it requires all input parameter from the portfolio class together with extra input parameters
        input parameters: 
    
        
        objective_linear_vector: numpy ndarray or cvxopt matrix, default=None
        cvxopt q vector in the objective
        
        weight_inequality_constraint_matrix: numpy ndarray or cvxopt matrix, default=None
        cvxopt G matrix
    
        weight_inequality_constraint_values: numpy ndarray or cvxopt matrix, default=None
        cvxopt h vector

        weight_equality_constraint_matrix: numpy ndarray or cvxopt matrix, default=None
        cvxopt A matrix
        
        weight_equality_constraint_values:
        cvxopt b vector
    
    """
    
    def __init__(self, 
                asset_names=None,
                 data=None,
                 is_price_data=True,
                 data_frequency='daily',
                 returns_frequency='daily',
                 risk_free_rate=0.0,
                 period_length=252,
                 init_amount_invested=1.0,
                realised_risk_matrix=None, 
                expected_risk_matrix=None,
                 objective_linear_vector=None,
                 long_only = True,
                 name='max_sharpe',
                 risk_method= 'sample_cov'
                ):
            
            
            Portfolio.__init__(self,asset_names=asset_names,data=data,is_price_data=is_price_data,
                             data_frequency=data_frequency,period_length=period_length,
                            returns_frequency=returns_frequency,
                             risk_free_rate=risk_free_rate,init_amount_invested=init_amount_invested)
            
            self.realised_risk_matrix=realised_risk_matrix
            self.expected_risk_matrix= expected_risk_matrix
            
            if self.is_price_data: 
                self.returns = self.compute_returns()
            else: 
                self.returns = self.data
            self.excess_returns = self.compute_excess_returns()
            self.n_assets= self.excess_returns.shape[1] 
            self.long_only= long_only
            

                
            self.weight_equality_constraint_matrix= matrix(np.array(np.mean(self.excess_returns,axis=0)).reshape(1,-1))
            self.weight_equality_constraint_values = matrix(1.0)
            self.objective_linear_vector = objective_linear_vector
            self.name = name
            self.risk_method=risk_method
            
            
            
            
    def compute_realised_risk_matrix(self, method='sample_cov',rowvar=0):
        """
        TODO: implement different methods for the risk matrix and make a class of it 
        computes the risk matrix according to the chosen method.
        
        method: str, one of ['bcov', 'cov', 'dcov' 'mdd']; default='cov'
            mmethod to us for the estimation of the risk matrix 
            
        self.period_length: int, one of [1, 5,20,250]
            representing in what period to represent the risk matrix. 1 is daily, 5 is weekly, 20 monthly 250 is yearly
            
        rowvar: bool
            if the rows represent the variables
        """
        if(rowvar== True):
            excess_returns_values = self.excess_returns.values.T
        else:
            excess_returns_values =self.excess_returns.values
            
        
        list_methods = ['bcov', 'sample_cov', 'dcov', 'mdd','shrinkage']
        if not(str(method) in list_methods):
            raise(ValueError('The chosen method is not allowed'))
            
        if(method=='sample_cov'):
            self.realised_risk_matrix = np.cov(excess_returns_values,rowvar=rowvar)*self.period_length
        elif(method=='dcov'):
            self.realised_risk_matrix = DCOV_pairwise(excess_returns_values) *self.period_length
        elif(method=='mdd'):
            self.realised_risk_matrix = MDD_pairwise(excess_returns_values)*self.period_length
        elif(method == 'shrinkage'):
            self.realised_risk_matrix= pd.DataFrame(ShrunkCovariance().fit(excess_returns_values).covariance_*self.period_length)                                
        elif( method=='bcov'):
            self.realised_risk_matrix = BCOV_pairwise(excess_returns_values)*self.period_length
            
        else:
            raise(ValueError('implement the chosen method'))
            
        return(self.realised_risk_matrix)
        
            
    def add_weights_equality_constraint(self,constraint_matrix,constraint_values):
        """
        adds additional equality constraint to the optimization problem
        
        constraint_matrix: cvx opt matrix representing the constraints, denoted A in cvxopt doc 
        
        constraint_values: cvx opt vector representing the constraints, denoted b in cvxopt doc
        
        """
        if self.weight_equality_constraint_values is None and self.weight_equality_constraint_matrix is None : 
            self.weight_equality_constraint_values = constraint_values
            self.weight_equality_constraint_matrix = constraint_matrix
        else:
            self.weight_equality_constraint_values=matrix(np.concatenate([self.weight_equality_constraint_values, 
                                                                          constraint_values]))
            self.weight_equality_constraint_matrix = matrix(np.concatenate([self.weight_equality_constraint_matrix, 
                                                                          constraint_matrix]))
            return
            
    def add_weights_inequality_constraint(self, constraint_matrix, constraint_values): 
        """
        adds additional inequality constraints to the optimization problem
        constraint_matrix: cvx opt matrix representing the constraints, denoted G in cvxopt doc 
        
        constraint_values: cvx opt vector representing the constraints, denoted h in cvxopt doc
        
        """
        if self.weight_inequality_constraint_values is None and self.weight_inequality_constraint_matrix is None : 
            self.weight_inequality_constraint_values = constraint_values
            self.weight_inequality_constraint_matrix = constraint_matrix
        else:
            self.weight_inequality_constraint_values=matrix(np.concatenate([self.weight_inequality_constraint_values, 
                                                                          constraint_values]))
            self.weight_inequality_constraint_matrix = matrix(np.concatenate([self.weight_inequality_constraint_matrix, 
                                                                          constraint_matrix]))
            return
        
            
        
    def optimize(self):
        
        """
        Solve the quadratic programming problem specified by the portfolio 
        """
        self.n_assets= self.data.shape[1] 
        print(self.n_assets)
        if self.is_price_data: 
                self.returns = self.compute_returns()
        else: 
            self.returns = self.data
        self.excess_returns = self.compute_excess_returns()
        
        self.weight_equality_constraint_matrix= matrix(np.array(np.mean(self.excess_returns,axis=0)).reshape(1,-1))
        print(self.weight_equality_constraint_matrix)
        print((self.weight_equality_constraint_matrix.size))
        self.weight_equality_constraint_values = matrix(1.0)
        
        if self.objective_linear_vector is None: 
            self.objective_linear_vector = matrix(0.0, (self.n_assets ,1))
            
        self.realised_risk_matrix = self.compute_realised_risk_matrix(method=self.risk_method)
        print(self.realised_risk_matrix)
        if self.long_only: 
            self.weight_inequality_constraint_matrix=-matrix(np.eye(self.n_assets))
            self.weight_inequality_constraint_values=matrix(0.0, (self.n_assets ,1))
            self.objective_linear_vector = matrix(0.0, (self.n_assets ,1))
        else:
            self.weight_inequality_constraint_matrix=-matrix(np.eye(self.n_assets)+np.ones((self.n_assets,self.n_assets)))
            self.weight_inequality_constraint_values=matrix(0.0, (self.n_assets ,1))
        
            
            
            
            
        options['show_progress'] = False
        sol = qp(matrix(2*self.realised_risk_matrix), self.objective_linear_vector,
                 self.weight_inequality_constraint_matrix, self.weight_inequality_constraint_values,
                 self.weight_equality_constraint_matrix, self.weight_equality_constraint_values)
        
        self.weights = sol['x'] *(1/dot(matrix(1.0, (self.n_assets ,1)),sol['x']))
        self.weights = np.array(self.weights).reshape(-1,1)
        return(np.array(self.weights).reshape(-1,1))
