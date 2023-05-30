import numpy as np 
import pandas as pd 
from scipy.linalg import block_diag
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from scipy.cluster import hierarchy
import fastcluster
from scipy.cluster.hierarchy import fcluster

import os
original_dir = os.getcwd()
os.chdir('..\\Utils')
from helper_funcs import *
os.chdir('..\\MVPortfolio')
from MVPortfolio_funcs import *
os.chdir('..\\wrapping')
from wrapping_funcs import *
os.chdir(original_dir)




class HRP_portfolio(Portfolio):
    """ Hierarchical Risk Parity portfolio 

    Parameters
    ----------
        asset_names: List  of strings
                    Names of the assets in the portfolio 
                    
        data : Pandas DataFrame 
                Data representing either the prices or the returns of the assets.  
                
        is_price_data: Logical 
                If the inputed data represents prices or returns 
                
        data_frequency: string, 
                Sampling frequency of the data. built in options are : 
                    daily : representing daily data 
                    
                    weekly: representing weekly data 
                    
                    monthly: representing monthly data 
                    
        returns_frequency : string, 
                Sampling frequency of the data. built in options are : 
                    daily : representing daily data 
                    
                    weekly: representing weekly data 
                    
                    monthly: representing monthly data 
                    
        risk_free_rate: float, 
                The chosen risk free rate 
                
        period_length: int, 
                Number of days in a period for data aggregation. This will be used to represent the aggregate statistics of the portfolio on a 
                yearly (252), monthly (30), or weekly (5) or daily (1) frequency. 
                
                
        init_amount_invested: float, 
                the initial amount invested. This will be used to compute the equity curve 
                
        cov_matrix: numpy array, 
                 the covariance matrix 
                 
        cor_matrix: numpy array, 
                the correlation matrix 
                
        name: "string 
                the name of the portfolio 
                
        cov_method: string 
                the method to estimate the covariance 
                
        cor_method:  string 
                the method to estimate the correlation. Possible methods: 
                sample: the sample correlation matrix 
                dcor: the sample pairwise distance correlation matrix 
                robust_cor: cellwise robust correlation matrix 
                robust_dcor: cellwise robust distance 
                
                
        linkage: string 
                the linkage method for the clustering. Available methods are "single", "average" or "complete"
        
                    
        
        
        
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
                 cov_matrix=None,
                 cor_matrix=None, 
                 name='HRP',
                 cov_method='sample',
                 cor_method='sample',
                 linkage_method='single'
                ):
        self.asset_names = asset_names 
        self.data = data 
        self.is_price_data = is_price_data
        self.data_frequency = data_frequency
        self.returns_frequency = returns_frequency 
        self.risk_free_rate = risk_free_rate 
        self.period_length = period_length
        self.init_amount_invested = init_amount_invested
        self.cov_matrix = cov_matrix 
        self.cor_matrix = cor_matrix
        self.name = name 
        self.cov_method = cov_method
        self.cor_method = cor_method
        self.linkage_method = linkage_method
        self.weights =  None
        
        
        
        
        
    def compute_cov(self,rowvar=0):
        """
        TODO: implement different methods for the risk matrix and make a class of it. This will avoid calling the same function in each portfolio.  
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
            
        list_methods = ['sample', 'robust_cov', 'robust_dcov','dcov','shrinkage', 'custom']
        if not (self.cov_method in list_methods): 
            raise(ValueError('The chosen cov or cor method is not allowed'))
            
        
            
        if(self.cov_method=='sample'):
            self.cov_matrix = self.excess_returns.cov()*self.period_length
        elif(self.cov_method == 'robust_cov'):
            self.cov_matrix = wrapped_covariance_correlation(excess_returns_values)[0]*self.period_length
        elif(self.cov_method=='dcov'):
            self.cov_matrix = pd.DataFrame(DCOV_pairwise(excess_returns_values) *self.period_length)
        elif(self.cov_method == 'shrinkage'):
            self.cov_matrix = pd.DataFrame(ShrunkCovariance().fit(excess_returns_values).covariance_*self.period_length)
        elif(self.cov_method == 'robust_dcov'):
            self.cov_matrix = wrapped_dcov(excess_returns_values)[0]*self.period_length
            
        else : 
            print('wrong method')

            
        return 
    
    def compute_cor(self,rowvar=0):
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
            
        list_methods = ['sample', 'robust_cor', 'robust_dcor','dcor', 'custom']
        if not(self.cor_method in list_methods):
            raise(ValueError('The chosen cov or cor method is not allowed'))
            
            
        if(self.cor_method =='sample'):
            self.cor_matrix = self.excess_returns.corr()*self.period_length
            
        elif(self.cor_method == 'dcor'):
            self.cor_matrix = pd.DataFrame(pairwise_dcor(excess_returns_values) *self.period_length)
            
        elif(self.cor_method == 'robust_cor'):
            self.cor_matrix = wrapped_covariance_correlation(excess_returns_values)[1]*self.period_length
            
        elif(self.cor_method == 'robust_dcor'):
            self.cor_matrix = wrapped_dcor(excess_returns_values)*self.period_length
            
        else : 
            print('wrong method')
            
        return self
      
            
    
    def optimize(self): 
        """
        Computes HRP weights
        """
        
        self.compute_cov()
        self.compute_cor()
        self.cov_matrix.fillna(0, inplace = True)
        self.cov_matrix =  pd.DataFrame(self.cov_matrix.values, index=self.asset_names, columns=self.asset_names)
        self.cor_matrix.fillna(0, inplace = True)
        self.cor_matrix =  pd.DataFrame(self.cor_matrix.values, index=self.asset_names, columns=self.asset_names)
        self.distances = ((1 - np.round(self.cor_matrix,6)) / 2)
        np.fill_diagonal(self.distances.values, 0)
        self.distances = pd.DataFrame(self.distances.values, index=self.asset_names, columns=self.asset_names)
        self.ordered_dist_mat, self.res_order, self.res_linkage = compute_serial_matrix(np.round(self.distances.values,6), method=self.linkage_method)
        self.weights = pd.Series(1, index = self.res_order)
        clustered_alphas = [self.res_order]
        self.cov_matrix.rename(index={x:y for x,y in zip(self.cov_matrix.columns,range(0,len(self.cov_matrix.columns)))},inplace=True)
        self.cov_matrix.rename(columns={x:y for x,y in zip(self.cov_matrix.columns,range(0,len(self.cov_matrix.columns)))},inplace=True)
        
        while len(clustered_alphas) > 0:
            clustered_alphas = [cluster[start:end] for cluster in clustered_alphas
                                for start, end in ((0, len(cluster) // 2),
                                                   (len(cluster) // 2, len(cluster)))
                                if len(cluster) > 1]# seperate the cluster in two if at least 2 elements
            for subcluster in range(0, len(clustered_alphas), 2):
                left_cluster = clustered_alphas[subcluster]#index of all elements in left cluster
                right_cluster = clustered_alphas[subcluster + 1]# index of all elements in right cluster
                left_subcovar = self.cov_matrix[left_cluster].loc[left_cluster]
                inv_diag = np.nan_to_num(1 / np.diag(left_subcovar.values))
                parity_w = np.nan_to_num(inv_diag * (1 / np.sum(inv_diag)))# weights are proportional to the inverse of the volatilities
                left_cluster_var = np.dot(parity_w, np.dot(left_subcovar, parity_w))

                right_subcovar = self.cov_matrix[right_cluster].loc[right_cluster]
                inv_diag = np.nan_to_num(1 / np.diag(right_subcovar.values))
                parity_w = np.nan_to_num(inv_diag * (1 / np.sum(inv_diag)))
                right_cluster_var = np.dot(parity_w, np.dot(right_subcovar, parity_w))

                alloc_factor = 1 - left_cluster_var / (left_cluster_var + right_cluster_var)

                self.weights[left_cluster] *= alloc_factor
                
                self.weights[right_cluster] *= 1 - alloc_factor
        
        self.weights.sort_index(inplace=True)  
        self.weights = np.array(self.weights).reshape(-1,1)
        return(np.array(self.weights).reshape(-1,1))
                
        
class HERC_portfolio(Portfolio): 
    """
    Generic class representing a hierarchical equal risk contribution portfolio 
    
    Parameters
    ----------
    
    """
    
    def __init__(self, 
                 asset_names=None,
                 data=None,
                 is_price_data=True,
                 data_frequency='daily',
                 returns_frequency='daily',
                 risk_free_rate=0.0,
                 period_length=1,
                 init_amount_invested=1.0,
                cov_matrix=None,
                 cor_matrix = None, 
                 name='HRP',
                 cov_method = 'sample',
                 cor_method = 'sample',
                 linkage_method = 'ward', 
                 risk_method = 'CVaR',
                 max_num_clusters = 10,
                 num_clusters_method = 'silhouette',
                 alpha = 0.01
                ):
        self.asset_names = asset_names 
        self.data = data 
        self.is_price_data = is_price_data
        self.data_frequency = data_frequency
        self.returns_frequency = returns_frequency 
        self.risk_free_rate = risk_free_rate 
        self.period_length = period_length
        self.init_amount_invested = init_amount_invested
        self.cov_matrix = cov_matrix 
        self.cor_matrix = cor_matrix
        self.name = name
        self.cov_method = cov_method
        self.cor_method = cor_method
        self.linkage_method = linkage_method
        self.risk_method = risk_method
        self.max_num_clusters = max_num_clusters 
        self.num_clusters_method = num_clusters_method
        self.alpha = alpha
        self.weights =  None
    
        
        
    def compute_cov(self,rowvar=0):
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
            
        list_methods = ['sample', 'robust_cov', 'robust_dcov','dcov', 'shrinkage', 'custom']
        if not(self.cov_method in list_methods):
            raise(ValueError('The chosen cov or cor method is not allowed'))
            
        
            
        if(self.cov_method=='sample'):
            self.cov_matrix = self.excess_returns.cov()*self.period_length
            
        elif(self.cov_method == 'shrinkage'):
            self.cov_matrix = pd.DataFrame(ShrunkCovariance().fit(excess_returns_values).covariance_*self.period_length)
            
        elif(self.cov_method == 'robust_cov'):
            self.cov_matrix = wrapped_covariance_correlation(excess_returns_values)[0]*self.period_length
            
        elif(self.cov_method == 'dcov'):
            self.cov_matrix = pd.DataFrame(DCOV_pairwise(excess_returns_values) *self.period_length)
            
        elif(self.cov_method == 'robust_dcov'):
            self.cov_matrix = wrapped_dcov_dcor(excess_returns_values)[0]*self.period_length
            
        else : 
            print('wrong method cov')

            
        return 
    
    def compute_cor(self,rowvar=0):
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
            
        list_methods = ['sample', 'robust_cor', 'robust_dcor', 'dcor', 'custom']
        if not(self.cor_method in list_methods):
            raise(ValueError('The chosen cov or cor method is not allowed'))
            
        
            
        if(self.cor_method =='sample'):
            self.cor_matrix = self.excess_returns.corr()*self.period_length
            
        elif(self.cor_method == 'dcor'):
            self.cor_matrix = pd.DataFrame(pairwise_dcor(excess_returns_values) *self.period_length)
            
        elif(self.cor_method == 'robust_cor'):
            self.cor_matrix = wrapped_covariance_correlation(excess_returns_values)[1]*self.period_length
            
        elif(self.cor_method == 'robust_dcor'):
            self.cor_matrix = wrapped_dcor(excess_returns_values)*self.period_length
            
        else : 
            print('wrong method cor')
            
            
        return 
    
    
    def _HERC_recursive_bisection(self): 
        """
        
        """
        
        root, nodes = hr.to_tree(self.clustering, rd=True)
        nodes = np.array(nodes)
        nodes_1 = np.array([i.dist for i in nodes])
        idx = np.argsort(nodes_1)
        nodes = nodes[idx][::-1].tolist()
        weight = pd.Series(1, index=self.asset_names)  # Set initial weights to 1
        clustering_inds = hr.fcluster(self.clustering, self.num_clusters, criterion="maxclust")
        clusters = {
            i: [] for i in range(min(clustering_inds), max(clustering_inds) + 1)
        }
        for i, v in enumerate(clustering_inds):
            clusters[v].append(i)
            
        for i in nodes[: self.num_clusters - 1]:
            if i.is_leaf() == False:  # skip leaf-nodes
                left = i.get_left().pre_order()  # lambda i: i.id) # get left cluster
                right = i.get_right().pre_order()  # lambda i: i.id) # get right cluster
                left_set = set(left)
                right_set = set(right)
                left_risk = 0
                right_risk = 0
                
                # Allocate weight to clusters
                for j in clusters.keys():
                    if set(clusters[j]).issubset(left_set):
                            
                            # Left cluster
                            left_cov = self.cov_matrix.iloc[clusters[j], clusters[j]]
                            left_returns = self.data.iloc[:, clusters[j]]
                            left_weight = naive_risk(left_returns, left_cov,self.risk_method, self.risk_free_rate,self.alpha)
                            if self.risk_method == 'CVaR':
                                left_risk_ = (CVaR_Hist(np.matmul(left_returns.values, left_weight),self.alpha))
                            else : 
                                left_risk_ = (CDaR_Abs(np.matmul(left_returns.values, left_weight),self.alpha))
                                
                            left_risk += left_risk_
                            
                    elif set(clusters[j]).issubset(right_set):
                               
                            # Right cluster
                            right_cov = self.cov_matrix.iloc[clusters[j], clusters[j]]
                            right_returns = self.returns.iloc[:, clusters[j]]
                            right_weight = naive_risk(right_returns, right_cov,self.risk_method, self.risk_free_rate,self.alpha)
                            if self.risk_method == 'CVaR':
                                right_risk_ = (CVaR_Hist(np.matmul(right_returns.values, right_weight),self.alpha))
                            else : 
                                right_risk_ = (CDaR_Abs(np.matmul(right_returns.values, right_weight),self.alpha))
                                
                            right_risk += right_risk_
                                                    
                alpha_1 = 1 - left_risk / (left_risk + right_risk)
                weight[left] *= alpha_1  # weight 1
                weight[right] *= 1 - alpha_1  # weight 2
    
    

        # Get constituents of k clusters
        clustered_assets = pd.Series(
            hr.cut_tree(self.clustering, n_clusters=self.num_clusters).flatten(), index=self.asset_names
        )
        # Multiply within-cluster weight with inter-cluster weight
        for i in range(self.num_clusters):
            cluster = clustered_assets.loc[clustered_assets == i]
            #print(cluster)
            cluster_cov = self.cov_matrix.loc[cluster.index, cluster.index]
            cluster_returns = self.data.loc[:, cluster.index]
            
            cluster_weights = pd.Series(
                naive_risk(
                    cluster_returns, cluster_cov,self.risk_method, self.risk_free_rate,self.alpha).flatten(),
                index=cluster_cov.index)
                
            weight.loc[cluster_weights.index] *= cluster_weights

        return weight
        
        
    
            
                
                
                
            

        
            
            
        
    
    
    def optimize(self): 
        """
        Compute HERC weights 
        """
        # assumes that the returns are passed in before computing the weights
        self.compute_cov()
        self.compute_cor()
        self.cov_matrix.fillna(0, inplace = True)
        self.cor_matrix.fillna(0, inplace = True)
        
        # step 1: Tree clustering 
        self.cov_matrix = pd.DataFrame(self.cov_matrix.values, columns=self.asset_names, index=self.asset_names)
        self.cor_matrix =  pd.DataFrame(self.cor_matrix.values, columns=self.asset_names, index=self.asset_names)
        self.distances = ((1 - np.round(self.cor_matrix,6)) / 2)
        np.fill_diagonal(self.distances.values, 0)
        self.distances = pd.DataFrame(self.distances, index=self.asset_names, columns=self.asset_names)
        
        flat_dist = squareform(self.distances.values, checks=False)
        self.clustering = linkage(flat_dist, method = self.linkage_method)
        
       
        
        
        
        # step 2: selection of optimal number of clusters  
        if self.num_clusters_method == 'silhouette': 
            self.num_clusters = opt_num_clusters_silhouette(self.clustering, self.distances,self.max_num_clusters)
            print('chosen number of clusters ', self.num_clusters)
        else : 
            self.num_clusters = opt_num_clusters_gap(self.clustering,self.distances, self.max_num_clusters)
            print('chosen number of clusters ', self.num_clusters)

        
        
        # step 3 : Recursive bisection for weights 
        
        self.weights =  self._HERC_recursive_bisection()
        self.weights = self.weights.loc[self.asset_names]
        self.weights = np.array(self.weights).reshape(-1,1)
        return(np.array(self.weights).reshape(-1,1))






class max_cluster_portfolio(Portfolio): 
    """
    Generic class representing a max/min Cluster portfolio. the difference will be done in the choice of the "risk_method" chosen. 
     
    
    Parameters
    ----------
    
    """
    
    def __init__(self, 
                 asset_names=None,
                 data=None,
                 is_price_data=True,
                 data_frequency='daily',
                 returns_frequency='daily',
                 risk_free_rate=0.0,
                 period_length=1,
                 init_amount_invested=1.0,
                cov_matrix=None,
                 cor_matrix = None, 
                 name='max_cluster',
                 cor_method = 'sample',
                 linkage_method = 'average', 
                 risk_method = 'CVaR',
                 num_clusters_type = 'fixed',
                 max_num_clusters = 10,
                 num_clusters_method = 'silhouette',
                 alpha = 0.01
                ):
        self.asset_names = asset_names 
        self.data = data 
        self.is_price_data = is_price_data
        self.data_frequency = data_frequency
        self.returns_frequency = returns_frequency 
        self.risk_free_rate = risk_free_rate 
        self.period_length = period_length
        self.init_amount_invested = init_amount_invested
        self.cov_matrix = cov_matrix 
        self.cor_matrix = cor_matrix
        self.name = name
        self.cor_method = cor_method
        self.linkage_method = linkage_method
        self.risk_method = risk_method
        self.num_clusters_type = num_clusters_type
        self.max_num_clusters = max_num_clusters 
        self.num_clusters_method = num_clusters_method
        self.alpha = alpha
        self.weights =  None
        
    
    
    def compute_cor(self,rowvar=0):
        
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
            
        list_methods = ['sample', 'robust_cor', 'robust_dcor', 'dcor', 'custom']
        if not(self.cor_method in list_methods):
            raise(ValueError('The chosen cov or cor method is not allowed'))
            
        
            
        if(self.cor_method =='sample'):
            self.cor_matrix = self.excess_returns.corr()*self.period_length
            
        elif(self.cor_method == 'dcor'):
            self.cor_matrix = pd.DataFrame(pairwise_dcor(excess_returns_values) *self.period_length)
            
        elif(self.cor_method == 'robust_cor'):
            self.cor_matrix = wrapped_covariance_correlation(excess_returns_values)[1]*self.period_length
            
        elif(self.cor_method == 'robust_dcor'):
            self.cor_matrix = wrapped_dcor(excess_returns_values)*self.period_length
            
        else : 
            print('wrong method cor')
            
            
        return 
        
    
        
        
    def optimize(self):
        # assumes that the returns are passed in before computing the weights
        # determine the number of clusters 
        
        # cluster the assets using hierarchical clustering 
        self.compute_cor()
        self.cor_matrix.fillna(0, inplace = True)
        self.distances = ((1 - np.round(self.cor_matrix,6)) / 2)
        self.distances_r  = rpy2.robjects.conversion.py2rpy(self.distances)
        self.cor_diss = squareform(np.round((1- self.cor_matrix)/2,6))
        self.cor_clust = linkage(self.cor_diss, method =self.linkage_method )

        if self.num_clusters_type == 'fixed': 
            self.num_clusters = self.max_num_clusters
        else: 
            if self.num_clusters_method == 'silhouette': 
                self.num_clusters = opt_num_clusters_silhouette(self.cor_clust, self.distances,self.max_num_clusters)
                print('chosen number of clusters ', self.num_clusters)
            else : 
                self.num_clusters = opt_num_clusters_gap(self.cor_clust,self.distances, self.max_num_clusters)
                print('chosen number of clusters ', self.num_clusters)
                
            
        self.cor_clusts = cut_tree(self.cor_clust, n_clusters= self.num_clusters)
        self.weights = np.zeros((1,))
        for clust_index in range(np.max(self.cor_clusts)+1): 
            #print('clust index ', clust_index)
            dat_clust = self.excess_returns.values[:,self.cor_clusts.flatten()==clust_index]
            cluster_weight_vec = np.zeros((dat_clust.shape[1],))
            if (self.risk_method == "sharpe"): 
                best_risk = np.mean(dat_clust[:,0])/np.sqrt(np.var(dat_clust[:,0]))
            elif(self.risk_method == "omega"): 
                best_risk = LPM(dat_clust[:,0], MAR=0, p=1)
            elif(self.risk_method == "sortino"): 
                best_risk = LPM(dat_clust[:,0], MAR=0, p=2)
            elif(self.risk_method == "maxDD"): 
                best_risk = MDD_Abs(dat_clust[:,0])
            elif(self.risk_method == "ADD"): 
                best_risk = ADD_Abs(dat_clust[:,0])
            elif(self.risk_method == "DaR"):
                best_risk = DaR_Abs(dat_clust[:,0])
            elif(self.risk_method == "CVaR" ):
                best_risk = CVaR_Hist(dat_clust[:,0])
            else: 
                print("wrong risk cluster portfolio method")

            best_risk_index = 0
            #get index of asset with the highest sharpe ratio of that cluster 
            for sub_clust_index in range(dat_clust.shape[1]):
                if(self.risk_method == "sharpe"): 
                    current_risk = np.mean(dat_clust[:,sub_clust_index])/np.sqrt(np.var(dat_clust[:,sub_clust_index]))
                elif( self.risk_method == "omega"):
                    current_risk = LPM(dat_clust[:,sub_clust_index], MAR=0, p=1)
                elif(self.risk_method == "sortino"):
                    current_risk = LPM(dat_clust[:,sub_clust_index], MAR=0, p=2)
                elif(self.risk_method == "maxDD"): 
                    current_risk = MDD_Abs(dat_clust[:,sub_clust_index])
                elif(self.risk_method == "ADD"): 
                    current_risk = ADD_Abs(dat_clust[:,sub_clust_index])
                elif(self.risk_method == "DaR"):
                    current_risk = DaR_Abs(dat_clust[:,sub_clust_index])
                elif(self.risk_method == "CVaR" ):
                    current_risk = CVaR_Hist(dat_clust[:,sub_clust_index])
                    
                else: 
                    print("wrong risk method cluster portfolio")
                if (self.risk_method in ["sharpe", "omega", "sortino"]):
                    if(current_risk >best_risk):
                        best_risk_index = sub_clust_index 
                        best_risk = current_risk
                if (self.risk_method in ["maxDD", "ADD", "DaR", "CVaR"]):
                    if(current_risk < best_risk):
                        best_risk_index = sub_clust_index 
                        best_risk = current_risk
                                 
            cluster_weight_vec[best_risk_index] = 1
            self.weights = np.concatenate((self.weights, cluster_weight_vec))

        
        self.weights = self.weights[1:]/np.sum(self.weights[1:])
        self.weights = np.array(self.weights).reshape(-1,1)
        
        
        
        return (np.array(self.weights).reshape(-1,1))