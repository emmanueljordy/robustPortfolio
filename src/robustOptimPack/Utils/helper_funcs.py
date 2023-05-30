import numpy as np 
import pandas as pd
import scipy.cluster.hierarchy as hr
from scipy.linalg import block_diag
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import fcluster
import dcor
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy import stats
from math import gamma, sqrt, pi
from scipy.stats import t

def compute_turnover(current_names,current_weights, prev_names, prev_weights): 
    """
    returns the turnover of the portfolio, taking into account that some assets can be added/removed between steps 
    """

    df1 = pd.DataFrame(current_weights.reshape((1,-1)), columns = current_names)
    df2 = pd.DataFrame(prev_weights.reshape((1,-1)), columns =prev_names)

    return(np.abs(pd.concat([df2, df1]).replace(np.NaN, 0).diff().tail(1)).sum(axis=1)[0])

def compute_log_returns(prices_DF): 
    """
    returns a DF containing the log returns obtained from the prices
    
    Parameters
    ----------
        prices_DF : pandas DataFrame 
                    prices
    """
    
    log_returns_DF = prices_DF.copy()
    for col_name in log_returns_DF.columns: 
        log_returns_DF[col_name] =  np.diff(np.log(prices_DF[col_name]), prepend = np.log(prices_DF[col_name])[0])
    
    log_returns_DF.replace([np.inf, -np.inf], np.nan, inplace=True)
    log_returns_DF.fillna(0, inplace = True)
    return(log_returns_DF.iloc[1:,:])

def compute_returns(prices_DF): 
    """
    returns a DF containing the log returns obtained from the prices
    
    Parameters
    ----------
        prices_DF : pandas DataFrame 
                    prices
    """
    
    returns_DF = prices_DF.pct_change()
    returns_DF.replace([np.inf, -np.inf], np.nan, inplace=True)
    returns_DF.fillna(0, inplace = True)
    
    
    return(returns_DF.iloc[1:,:])

def get_highest_trading_volume(volumes_df, K): 
    """
    returns the names of the K columns of volumes_df with the highest volumes
    
    Parameters
    ----------
         volumes_df : pandas dataframe, 
         
         K : int, 
             number of columns to return
    """
    vol_means = volumes_df.median(axis=0)
    vol_means_non_null = vol_means[vol_means>0]
    len_non_null  =  len(vol_means_non_null)
    return(volumes_df.median(axis=0).nlargest(n=min(K,len_non_null)).index.values.tolist())



# Helper functions for HRP and HERC obtained from Gauter Marti's blog 

def seriation(Z, N, cur_index):
    """Returns the order implied by a hierarchical tree (dendrogram).
    
       :param Z: A hierarchical tree (dendrogram).
       :param N: The number of points given to the clustering process.
       :param cur_index: The position in the tree for the recursive traversal.
       
       :return: The order implied by the hierarchical tree Z.
    """
    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index - N, 0])
        right = int(Z[cur_index - N, 1])# In the hierarchical clustering, if you want to go down one level you have to remove N since when to clusters are merged, the  new cluster number is defined as current_ind (i.e. current_hierarchical level) + N
        return (seriation(Z, N, left) + seriation(Z, N, right))

    
def compute_serial_matrix(dist_mat, method="ward"):
    """Returns a sorted distance matrix.
    
       :param dist_mat: A distance matrix.
       :param method: A string in ["ward", "single", "average", "complete"].
        
        output:
            - seriated_dist is the input dist_mat,
              but with re-ordered rows and columns
              according to the seriation, i.e. the
              order implied by the hierarchical tree
            - res_order is the order implied by
              the hierarhical tree
            - res_linkage is the hierarhical tree (dendrogram)
        
        compute_serial_matrix transforms a distance matrix into 
        a sorted distance matrix according to the order implied 
        by the hierarchical tree (dendrogram)
    """
    N = len(dist_mat)
    flat_dist_mat = squareform(dist_mat)
    res_linkage = linkage(flat_dist_mat, method=method)
    res_order = seriation(res_linkage, N, N + N - 2)#-2 = -1-1 because of 0 based aindexing and there are N-1 steps in chierarchical clustering
    # res order contains the index of elements that are close by in the dendogram. elements are ordered by how close they are in the left/right side of the dendogram => check this
    seriated_dist = np.zeros((N, N))
    a,b = np.triu_indices(N, k=1)# matrix is symmetric so only need to fill the upper diagonal
    seriated_dist[a,b] = dist_mat[[res_order[i] for i in a], [res_order[j] for j in b]]
    seriated_dist[b,a] = seriated_dist[a,b]
    
    return seriated_dist, res_order, res_linkage

def intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2))


def opt_num_clusters_gap(clusters,dist_mat,max_k=10):
    r"""
    copied from riskFolio-Lib
    Calculate the optimal number of clusters based on the two difference gap
    statistic :cite:`d-twogap`.
    Parameters
    ----------

    dist_mat : numpy array
        the distance matrix.
    clusters : string, optional
        The hierarchical clustering encoded as a linkage matrix, see `linkage <https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html?highlight=linkage#scipy.cluster.hierarchy.linkage>`_ for more details.
    max_k : int, optional
        Max number of clusters used by the two difference gap statistic
        to find the optimal number of clusters. The default is 10.
    Returns
    -------
    k : int
        The optimal number of clusters based on the two difference gap statistic.
    Raises
    ------
        ValueError when the value cannot be calculated.
    """
    dist_mat.rename(index={x:y for x,y in zip(dist_mat.columns,range(0,len(dist_mat.columns)))}, inplace = True)
    dist_mat.rename(columns={x:y for x,y in zip(dist_mat.columns,range(0,len(dist_mat.columns)))}, inplace = True)
    
    # cluster levels over from 1 to N-1 clusters
    cluster_lvls = pd.DataFrame(hr.cut_tree(clusters))
    num_k = cluster_lvls.columns  # save column with number of clusters
    cluster_lvls = cluster_lvls.iloc[:, ::-1]  # reverse order to start with 1 cluster
    cluster_lvls.columns = num_k  # set columns to number of cluster
    W_list = []

    # get within-cluster dissimilarity for each k
    for k in range(min(len(cluster_lvls.columns), max_k)):
        level = cluster_lvls.iloc[:, k]  # get k clusters
        D_list = []  # within-cluster distance list

        for i in range(np.max(level.unique()) + 1):
            cluster = level.loc[level == i]
            # Based on correlation distance
            cluster_dist = dist_mat.loc[cluster.index, cluster.index]  # get distance
            cluster_pdist = squareform(cluster_dist, checks=False)
            if cluster_pdist.shape[0] != 0:
                D = np.nan_to_num(cluster_pdist.mean())
                D_list.append(D)  # append to list

        W_k = np.sum(D_list)
        W_list.append(W_k)

    W_list = pd.Series(W_list)
    n = dist_mat.shape[0]
    limit_k = int(min(max_k, np.sqrt(n)))
    gaps = W_list.shift(2) + W_list - 2 * W_list.shift(1)
    gaps = gaps[0:limit_k]
    if gaps.isna().all():
        k = len(gaps)
    else:
        k = int(gaps.idxmax() + 2)

    return k


def opt_num_clusters_silhouette_r(clustering, dist_mat, max_num_clusters=10):
    """
    returns the optimal number of clusters using the silhouette plot in R
    """   
    avg_silhouette_list = []
    num_clusters_list = []
    for num_clusters in range(2,max_num_clusters+1):
        cluster_labels = statslib.cutree(clustering, num_clusters)
        if (len(set(cluster_labels))>1):
            sil_obj = clusterlib.silhouette_default_R(cluster_labels, dist_mat)
            avg_silhouette_list.append(np.mean(sil_obj[:,2]))
            num_clusters_list.append(num_clusters)
            
    return (num_clusters_list[avg_silhouette_list.index(max(avg_silhouette_list))])
    
    
    
def opt_num_clusters_silhouette(clustering, dist_mat,max_num_clusters=10): 
    """
    returns the optimal number of clusters using the silhouette plot 
    """
    avg_silhouette_list =[]
    num_clusters_list = []
    for num_clusters in range(2,max_num_clusters+1):
        cluster_labels = fcluster(clustering, num_clusters, criterion='maxclust')
        if(len(set(cluster_labels))>1):
            avg_silhouette_list.append(silhouette_score(dist_mat,cluster_labels, metric = 'precomputed'))
            num_clusters_list.append(num_clusters)
        
    return (num_clusters_list[avg_silhouette_list.index(max(avg_silhouette_list))])






# risk metrics from Riskfolio-Lib 

def CDaR_Abs(X, alpha=0.05):
    r"""
    Calculate the Conditional Drawdown at Risk (CDaR) of a returns series
    using uncumpounded cumulative returns.
    .. math::
        \text{CDaR}_{\alpha}(X) = \text{DaR}_{\alpha}(X) + \frac{1}{\alpha T}
        \sum_{j=0}^{T} \max \left [ \max_{t \in (0,j)}
        \left ( \sum_{i=0}^{t}X_{i} \right ) - \sum_{i=0}^{j}X_{i}
        - \text{DaR}_{\alpha}(X), 0 \right ]
    Where:
    :math:`\text{DaR}_{\alpha}` is the Drawdown at Risk of an uncumpound
    cumulated return series :math:`X`.
    Parameters
    ----------
    X : 1d-array
        Returns series, must have Tx1 size..
    alpha : float, optional
        Significance level of CDaR. The default is 0.05.
    Raises
    ------
    ValueError
        When the value cannot be calculated.
    Returns
    -------
    value : float
        CDaR of an uncumpounded cumulative returns series.
    """

    a = np.array(X, ndmin=2)
    if a.shape[0] == 1 and a.shape[1] > 1:
        a = a.T
    if a.shape[0] > 1 and a.shape[1] > 1:
        raise ValueError("returns must have Tx1 size")

    prices = np.insert(np.array(a), 0, 1, axis=0)
    NAV = np.cumsum(np.array(prices), axis=0)
    DD = []
    peak = -99999
    for i in NAV:
        if i > peak:
            peak = i
        DD.append(-(peak - i))
    del DD[0]
    sorted_DD = np.sort(np.array(DD), axis=0)
    index = int(np.ceil(alpha * len(sorted_DD)) - 1)
    sum_var = 0
    for i in range(0, index + 1):
        sum_var = sum_var + sorted_DD[i] - sorted_DD[index]
        
    value = -sorted_DD[index] - sum_var / (alpha * len(sorted_DD))
    value = np.array(value).item()

    return value

def CDaR_Rel(X, alpha=0.05):
    r"""
    Calculate the Conditional Drawdown at Risk (CDaR) of a returns series
    using cumpounded cumulative returns.
    .. math::
        \text{CDaR}_{\alpha}(X) = \text{DaR}_{\alpha}(X) + \frac{1}{\alpha T}
        \sum_{i=0}^{T} \max \left [ \max_{t \in (0,T)}
        \left ( \prod_{i=0}^{t}(1+X_{i}) \right )- \prod_{i=0}^{j}(1+X_{i})
        - \text{DaR}_{\alpha}(X), 0 \right ]
    Where:
    :math:`\text{DaR}_{\alpha}` is the Drawdown at Risk of a cumpound
    cumulated return series :math:`X`.
    Parameters
    ----------
    X : 1d-array
        Returns series, must have Tx1 size..
    alpha : float, optional
        Significance level of CDaR. The default is 0.05.
    Raises
    ------
    ValueError
        When the value cannot be calculated.
    Returns
    -------
    value : float
        CDaR of a cumpounded cumulative returns series.
    """

    a = np.array(X, ndmin=2)
    if a.shape[0] == 1 and a.shape[1] > 1:
        a = a.T
    if a.shape[0] > 1 and a.shape[1] > 1:
        raise ValueError("X must have Tx1 size")

    prices = 1 + np.insert(np.array(a), 0, 0, axis=0)
    NAV = np.cumprod(prices, axis=0)
    DD = []
    peak = -99999
    for i in NAV:
        if i > peak:
            peak = i
        DD.append(-(peak - i) / peak)
    del DD[0]
    sorted_DD = np.sort(np.array(DD), axis=0)
    index = int(np.ceil(alpha * len(sorted_DD)) - 1)
    sum_var = 0
    for i in range(0, index + 1):
        sum_var = sum_var + sorted_DD[i] - sorted_DD[index]
    value = -sorted_DD[index] - sum_var / (alpha * len(sorted_DD))
    value = np.array(value).item()

    return value

def CVaR_Hist(X, alpha=0.05):
    r"""
    Calculate the Conditional Value at Risk (CVaR) of a returns series.
    .. math::
        \text{CVaR}_{\alpha}(X) = \text{VaR}_{\alpha}(X) +
        \frac{1}{\alpha T} \sum_{t=1}^{T} \max(-X_{t} -
        \text{VaR}_{\alpha}(X), 0)
    Parameters
    ----------
    X : 1d-array
        Returns series, must have Tx1 size.
    alpha : float, optional
        Significance level of CVaR. The default is 0.05.
    Raises
    ------
    ValueError
        When the value cannot be calculated.
    Returns
    -------
    value : float
        CVaR of a returns series.
    """

    a = np.array(X, ndmin=2)
    if a.shape[0] == 1 and a.shape[1] > 1:
        a = a.T
    if a.shape[0] > 1 and a.shape[1] > 1:
        raise ValueError("returns must have Tx1 size")

    sorted_a = np.sort(a, axis=0)
    index = int(np.ceil(alpha * len(sorted_a)) - 1)
    sum_var = 0
    for i in range(0, index + 1):
        sum_var = sum_var + sorted_a[i] - sorted_a[index]# uses definition from QRM book using the order statistics (2.28)
        #print('sorted_a[i] ', sorted_a[i])
        #print('sorted_a[index] ', sorted_a[index])
    value = -sorted_a[index] - sum_var / (alpha * len(sorted_a))
    value = np.array(value).item()

    return value

def naive_risk(returns, cov, rm="CVaR", rf=0, alpha = 0.01):
    assets = returns.columns.tolist()
    n = len(assets)

    if rm == "equal":
        weight = np.ones((n, 1)) * 1 / n
    else:
        inv_risk = np.zeros((n, 1))
        for i in assets:
            k = assets.index(i)
            w = np.zeros((n, 1))
            w[k, 0] = 1
            w = pd.DataFrame(w, columns=["weights"], index=assets)
            if (rm == "CVaR"):
                risk = CVaR_Hist(np.matmul(returns.values,w.values), alpha)
                #print('risk is ', risk) 
            elif (rm == "CDaR"):
                risk = CDaR_Abs(np.matmul(returns.values,w.values), alpha)
                #print('risk is ', risk) 
            else : 
                ValueError("wrong name for risk measure")
            inv_risk[k, 0] = risk
        
            
        inv_risk = np.nan_to_num(1 / inv_risk )
        weight = np.nan_to_num(inv_risk * (1 / np.sum(inv_risk)))

    weight = weight.reshape(-1, 1)

    return weight


def MDD_Abs(X):
    r"""
    Calculate the Maximum Drawdown (MDD) of a returns series
    using uncompounded cumulative returns.
    .. math::
        \text{MDD}(X) = \max_{j \in (0,T)} \left [\max_{t \in (0,j)}
        \left ( \sum_{i=0}^{t}X_{i} \right ) - \sum_{i=0}^{j}X_{i}  \right ]
    Parameters
    ----------
    X : 1d-array
        Returns series, must have Tx1 size.
    Raises
    ------
    ValueError
        When the value cannot be calculated.
    Returns
    -------
    value : float
        MDD of an uncompounded cumulative returns.
    """

    a = np.array(X, ndmin=2)
    if a.shape[0] == 1 and a.shape[1] > 1:
        a = a.T
    if a.shape[0] > 1 and a.shape[1] > 1:
        raise ValueError("returns must have Tx1 size")

    prices = np.insert(np.array(a), 0, 1, axis=0)
    NAV = np.cumsum(np.array(prices), axis=0)
    value = 0
    peak = -99999
    for i in NAV:
        if i > peak:
            peak = i
        DD = peak - i
        if DD > value:
            value = DD

    value = np.array(value).item()

    return value


def ADD_Abs(X):
    r"""
    Calculate the Average Drawdown (ADD) of a returns series
    using uncompounded cumulative returns.
    .. math::
        \text{ADD}(X) = \frac{1}{T}\sum_{j=0}^{T}\left [ \max_{t \in (0,j)}
        \left ( \sum_{i=0}^{t}X_{i} \right ) - \sum_{i=0}^{j}X_{i} \right ]
    Parameters
    ----------
    X : 1d-array
        Returns series, must have Tx1 size.
    Raises
    ------
    ValueError
        When the value cannot be calculated.
    Returns
    -------
    value : float
        ADD of an uncompounded cumulative returns.
    """

    a = np.array(X, ndmin=2)
    if a.shape[0] == 1 and a.shape[1] > 1:
        a = a.T
    if a.shape[0] > 1 and a.shape[1] > 1:
        raise ValueError("returns must have Tx1 size")

    prices = np.insert(np.array(a), 0, 1, axis=0)
    NAV = np.cumsum(np.array(prices), axis=0)
    value = 0
    peak = -99999
    n = 0
    for i in NAV:
        if i > peak:
            peak = i
        DD = peak - i
        if DD > 0:
            value += DD
        n += 1
    if n == 0:
        value = 0
    else:
        value = value / (n - 1)

    value = np.array(value).item()

    return value


def DaR_Abs(X, alpha=0.05):
    r"""
    Calculate the Drawdown at Risk (DaR) of a returns series
    using uncompounded cumulative returns.
    .. math::
        \text{DaR}_{\alpha}(X) & = \max_{j \in (0,T)} \left \{ \text{DD}(X,j)
        \in \mathbb{R}: F_{\text{DD}} \left ( \text{DD}(X,j) \right )< 1-\alpha
        \right \} \\
        \text{DD}(X,j) & = \max_{t \in (0,j)} \left ( \sum_{i=0}^{t}X_{i}
        \right )- \sum_{i=0}^{j}X_{i}
    Parameters
    ----------
    X : 1d-array
        Returns series, must have Tx1 size..
    alpha : float, optional
        Significance level of DaR. The default is 0.05.
    Raises
    ------
    ValueError
        When the value cannot be calculated.
    Returns
    -------
    value : float
        DaR of an uncompounded cumulative returns series.
    """

    a = np.array(X, ndmin=2)
    if a.shape[0] == 1 and a.shape[1] > 1:
        a = a.T
    if a.shape[0] > 1 and a.shape[1] > 1:
        raise ValueError("returns must have Tx1 size")

    prices = np.insert(np.array(a), 0, 1, axis=0)
    NAV = np.cumsum(np.array(prices), axis=0)
    DD = []
    peak = -99999
    for i in NAV:
        if i > peak:
            peak = i
        DD.append(-(peak - i))
    del DD[0]
    sorted_DD = np.sort(np.array(DD), axis=0)
    index = int(np.ceil(alpha * len(sorted_DD)) - 1)
    value = -sorted_DD[index]
    value = np.array(value).item()

    return value


def LPM(X, MAR=0, p=1):
    r"""
    Calculate the First or Second Lower Partial Moment of a returns series.
    .. math::
        \text{LPM}(X, \text{MAR}, 1) &= \frac{1}{T}\sum_{t=1}^{T}
        \max(\text{MAR} - X_{t}, 0) \\
        \text{LPM}(X, \text{MAR}, 2) &= \left [ \frac{1}{T-1}\sum_{t=1}^{T}
        \max(\text{MAR} - X_{t}, 0)^{2} \right ]^{\frac{1}{2}} \\
    Where:
    :math:`\text{MAR}` is the minimum acceptable return.
    :math:`p` is the order of the :math:`\text{LPM}`.
    Parameters
    ----------
    X : 1d-array
        Returns series, must have Tx1 size.
    MAR : float, optional
        Minimum acceptable return. The default is 0.
    p : float, optional can be {1,2} 
        order of the :math:`\text{LPM}`. The default is 1.
    Raises
    ------
    ValueError
        When the value cannot be calculated.
    Returns
    -------
    value : float
        p-th Lower Partial Moment of a returns series.
    """

    a = np.array(X, ndmin=2)
    if a.shape[0] == 1 and a.shape[1] > 1:
        a = a.T
    if a.shape[0] > 1 and a.shape[1] > 1:
        raise ValueError("returns must have Tx1 size")
    if p not in [1, 2]:
        raise ValueError("p can only be 1 or 2")

    value = a- MAR

    if p == 2:
        n = value.shape[0] - 1
    else:
        n = value.shape[0]

    value = np.sum(np.power(value[np.where(value >= 0)], p)) / n
    value = np.power(value, 1 / p).item()

    return value


def DCOV_pairwise(X): 
    """
    returns the matrix of pairwise DCOV for the elements of X
    
    input :
    X : ndarray of shape (n,p)
    
    """ 
    n,p = X.shape
    DCOV_mat = np.ones((p,p))
    for i in range(p): 
        for j in range(p): 
            DCOV_mat[i,j] = np.sqrt(dcor.distance_stats_sqr(X[:,i].reshape(-1,1),X[:,j].reshape(-1,1))[0])
    
    
    return(DCOV_mat)


def robust_z_scores(X): 
    """returns the robust z_scores  of X

    Args:
        X (numpy array):  size (n x 1)
    """
    X.dropna(inplace=True)
    mad = stats.median_abs_deviation(X, scale = "normal")
    med = np.median(X)
    
    robust_z = np.abs(X - med)/mad
    
    return robust_z

def fraction_outliers(X, cutoff = 2.5):
    """returns the fraction of outliers in X

    Args:
        X (numpy array): size n x 1
    """
    
    frac = np.round(np.mean(robust_z_scores(X)> cutoff),2)
    return frac
    
def sktinv(u,nu,_lambda):
    '''
    % sktinv(u,nu,_lambda)
    % u      = Cdf value (Between 0 and 1)
    % nu     = Kurtosis Parameter (>2 to inf)
    % lambda = Assymetry Parameter (-1 to 1)
    %
    % returns z = random Variable (between -inf and inf)
    %
    % Calculates the inverse CDF for the skewed Student-t. Hansen (1994) version.
    % Author: Christian Contino
    % Date:   2 May 2021
    '''
    
    u  = np.atleast_1d(u)
    
    c = gamma((nu+1)/2)/(sqrt(pi*(nu-2))*gamma(nu/2))
    a = 4*_lambda*c*((nu-2)/(nu-1))
    b = sqrt(1 + 3 * _lambda**2 - a**2)
    
    inv1 = (1-_lambda)/b * sqrt((nu-2) / nu) * t.ppf(u / (1-_lambda), nu) - a/b
    inv2 = (1+_lambda)/b * sqrt((nu-2) / nu) * t.ppf(0.5 + (1 / (1+_lambda)) * (u - (1-_lambda)/2) , nu) - a/b
    
    limit_variable = (1-_lambda)/2
    
    z = u.copy()
    
    lt = u < limit_variable
    gt = u >= limit_variable
    
    z[lt] = inv1[lt]
    z[gt] = inv2[gt]

    return z

def sktcdf(z,nu,_lambda):
    '''
    % sktcdf(z,nu,lambda)
    % z      = Random Variable value (Between -inf and inf)
    % nu     = Kurtosis Parameter (>2 to inf)
    % lambda = Assymetry Parameter (-1 to 1)
    %
    % returns u = random Variable (between 0 and 1)
    %
    % Calculates the CDF for the skewed Student-t. Hansen (1994) version.
    % Author: Christian Contino
    % Date:   2 May 2021
    '''
    
    z  = np.atleast_1d(z)
    
    c = gamma((nu+1)/2)/(sqrt(pi*(nu-2))*gamma(nu/2))
    a = 4*_lambda*c*((nu-2)/(nu-1))
    b = sqrt(1 + 3 * _lambda**2 - a**2)
    
    limit_variable = -a/b
    lt = z < limit_variable
    gt = z >= limit_variable
    
    y_1 = (b*z+a) / (1-_lambda) * sqrt(nu/(nu-2))
    y_2 = (b*z+a) / (1+_lambda) * sqrt(nu/(nu-2))
    
    pdf1 = (1-_lambda) * t.cdf(y_1, nu)
    pdf2 = (1-_lambda)/2 + (1+_lambda) * (t.cdf(y_2, nu)-0.5)

    u = z.copy()

    u[lt] = pdf1[lt]
    u[gt] = pdf2[gt]

    return u


def generate_skt(n,nu, _lambda):
    """generates random samples from skewed t dist

    Args:
        n (_int_): number of samples to generate 
        nu (_float_): Kurtosis Parameter (>2 to inf)
        _lambda (_float_): Assymetry Parameter (-1 to 1)
    """
    
    u = np.random.uniform(size=(n,1))
    res = sktinv(u,nu,_lambda)
    
    return(res)
    
    