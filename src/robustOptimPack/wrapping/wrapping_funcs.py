
import numpy as np
from scipy import stats
import pandas as pd
from math import sqrt, tanh, ceil
import dcor 
import scipy

###############################
# Main functions
###############################

# Splits the array into individual columns and performs the wrapping with the corresponding location and scale parameters
def wrap(x, params=1):
    """
    Main function
    inputs:
        x: time series as a (n,d) array (n timepoints, d dimensions)
        params: [1,2,3] for parameter sets used in the manuscript
    
    returns: Wrapped time series, location and scale
    """
    b=1.5; c=4.0; k = 4.1517212; A = 0.7532528; B = 0.8430849

    if params == 2:
        b=1.25; c= 3.5; k = 3.856305; A = 0.6119228; B = 0.7354239
    elif params == 3:
        b=1.25; c = 3.0; k = 4.357096; A = 0.5768820; B = 0.6930791

    xW = np.zeros(x.shape)
    try:
        loc = np.zeros(x.shape[1])
        scale = np.zeros(x.shape[1])
        for i in range(x.shape[1]):
            l, s = estLocScale(x[:,i], b, c, k, A, B)
            loc[i] = l
            scale[i] = s
            xW[:,i] = perform_wrapping(x[:,i], loc[i], scale[i], b, c, k, A, B)

    except:
        loc = np.zeros(1)
        scale = np.zeros(1)
        l, s = estLocScale(x, b, c, k, A, B)
        loc = l
        scale = s
        xW = perform_wrapping(x, loc, scale, b, c, k, A, B)

    return xW, loc, scale

# Robust location and scale parameters
def estLocScale(x, b=1.5, c=4.0, k=4.1517212, A=0.7532528, B=0.8430849):
    m1 = loc1StepM(x, b, c, k, A, B)
    s1 = scale1StepM(x-m1, b, c, k, A, B)

    return m1,s1

# Performs the wrapping with the psiTanh function
def perform_wrapping(x, loc, scale, b=1.5, c=4.0, k=4.1517212, A=0.7532528, B=0.8430849):
    u = x - loc
    u = np.nan_to_num(u / scale)
    xW = psiTanh(u, b, c, k, A, B)
    xW = xW * scale + loc

    return xW

###############################
# Scale
###############################

def scale1StepM(x, b=1.5, c=4.0, k=4.1517212, A=0.7532528, B=0.8430849):
    s0 = 1.482602218505602 * np.median(abs(x))
    rho = np.nan_to_num(x / s0)
    w = rhoTanh154(rho, b, c, k, A, B)
    s1 = s0 * sqrt(sum(w) / (0.5 * len(x))) 
    
    cn = len(x) / (len(x) - 1.208) # finite sample correction

    s1 *= cn
    if (s1 == 0): 
        s1 = np.std(x)
    return s1

# Hyperbolic tangent rho function
def rhoTanh154(x, b=1.5, c=4.0, k=4.1517212, A=0.7532528, B=0.8430849):  
    x = psiTanh(x, b, c, k, A, B)
    x = pow(x,2) / 1.506506

    return x

# Psi-function of the hyperbolic tangent estimator. Default k, A, B for b= 1.5 and c=4.0
def psiTanh(x, b=1.5, c=4.0, k=4.1517212, A=0.7532528, B=0.8430849):    
    for i in range(len(x)):  
        x[i] = 0.0 if abs(x[i]) > c else x[i]
    
    for i in range(len(x)):                                                
        x[i] = sqrt(A*(k-1.0)) * tanh(0.5*sqrt((k-1.0)*pow(B,2.0)/A) * (c - abs(x[i]))) * np.sign(x[i]) if abs(x[i]) > b else x[i]

    return x

###############################
# Location
###############################

def loc1StepM(x, b=1.5, c=4.0, k=4.1517212, A=0.7532528, B=0.8430849):
    med = np.median(x)
    mad = stats.median_abs_deviation(x, scale='normal')
    if (mad == 0): 
        z = 0*x # because this mostly happen when returns are 0
    else: 
        z = (x-med)/mad
    weights = locTanh154(z, b, c, k, A, B)
    mu = sum(x*weights) / sum(weights)

    return mu



# Hyperbolic Tangent weight function to be used in location M-estimators
def locTanh154(x, b=1.5, c=4.0, k=4.1517212, A=0.7532528, B=0.8430849):
    for i in range(len(x)):
        if abs(x[i]) < b:
            x[i] = 1.0
        elif abs(x[i]) > c:
            x[i] = 0.0
        else:
            x[i] = sqrt(A*(k-1.0)) * tanh(0.5*sqrt((k-1.0)*pow(B,2.0)/A) * (c - abs(x[i]))) / abs(x[i])

    return x

# Functions for robust covariance/correlation based on  Fast Robust Correlation for High-Dimensional Data Jakob Raymaekers and Peter J. Rousseeuw

def subset_hampel( z,a,b,c):
    """
    Hampel's function is defined piecewise over the range of z
    """
    z = np.abs(np.asarray(z))
    t1 = np.less_equal(z, a)
    t2 = np.less_equal(z, b) * np.greater(z, a)
    t3 = np.less_equal(z, c) * np.greater(z, b)
    return t1, t2, t3


def psi_hampel( z,a,b,c):
        r"""
        # function copied from statsmodels documentation 
        The psi function for Hampel's estimator

        The analytic derivative of rho

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        psi : ndarray
            psi(z) = z                            for \|z\| <= a

            psi(z) = a*sign(z)                    for a < \|z\| <= b

            psi(z) = a*sign(z)*(c - \|z\|)/(c-b)    for b < \|z\| <= c

            psi(z) = 0                            for \|z\| > c
        """
        z = np.asarray(z)
        t1, t2, t3 = subset_hampel(z,a,b,c)
        s = np.sign(z)
        z = np.abs(z)
        v = s * (t1 * z +
                 t2 * a*s +
                 t3 * a*s * (c - z) / (c - b))
        return v
    
# def wrap(data ):
#     """
#     Transforms the input data to its wrapped counterpart 
    
#     Parameters
#     -----------
#             data : numpy ndarray, 
            
#                 The input data to wrap
#     """
    
#     # for each column estimate a robust location M-estimator and the MAD as scale estimator 
#     wrap_data = data.copy()
#     robust_mean= np.ones(wrap_data.shape[1])
#     robust_scale = np.ones(wrap_data.shape[1])
#     for col_index in range(wrap_data.shape[1]):
#         robust_mod_prep = sm.RLM(wrap_data[:,col_index], np.ones((data.shape[0])), M=sm.robust.norms.Hampel(1,2,4))
#         robust_mod = robust_mod_prep.fit(scale_est='mad')
#         robust_mean[col_index]  = robust_mod.params 
#         robust_scale[col_index] = robust_mod.scale
#         wrap_data[:,col_index] = robust_mean[col_index]  + robust_scale[col_index]  *(psi_hampel((wrap_data[:,col_index] - robust_mean[col_index] )/robust_scale[col_index],1,2,4 )) 
#         wrap_data[:,col_index] = np.nan_to_num(wrap_data[:,col_index])
#     return(wrap_data, robust_mean, robust_scale)


def wrap_dcov(data ):
    """
    Transforms the input data to its wrapped counterpart 
    
    Parameters
    -----------
            data : numpy ndarray, 
            
                The input data to wrap
    """
    
    # for each column estimate a robust location M-estimator and the MAD as scale estimator 
    wrap_data = data.copy()
    robust_mean= np.ones(wrap_data.shape[1])
    robust_scale = np.ones(wrap_data.shape[1])
    for col_index in range(wrap_data.shape[1]):
        #robust_mod_prep = sm.RLM(wrap_data[:,col_index], np.ones((data.shape[0])), M=sm.robust.norms.Hampel(1,2,4))
        #robust_mod = robust_mod_prep.fit(scale_est='mad')
        robust_mean[col_index]  = np.median(wrap_data[:,col_index])
        robust_scale[col_index] = stats.median_abs_deviation(wrap_data[:,col_index], scale='normal')
        if (robust_scale[col_index] == 0):
            robust_scale[col_index] = np.std(wrap_data[:,col_index])
        wrap_data[:,col_index] = robust_mean[col_index]  + robust_scale[col_index]  *(np.tanh((wrap_data[:,col_index] - robust_mean[col_index] )/robust_scale[col_index])) 
        wrap_data[:,col_index] = np.nan_to_num(wrap_data[:,col_index])
    return(wrap_data, robust_mean, robust_scale)



def wrapped_covariance_correlation(data) : 
    """
    Computes the wrapped covariance and correlation of the data 
        Parameters
    -----------
            data : numpy ndarray, 
            
                The input data to wrap
    """
    
    
    wrap_data, robust_mean, robust_scale = wrap(data.copy())
    
    p = data.shape[1]
    cov = np.ones((p, p))
    cor = np.ones((p, p))
    indices = np.triu_indices(p, 0)
    
    for i, j in zip(indices[0], indices[1]):
        #print(scipy.stats.pearsonr(wrap_data[:, i], wrap_data[:, j]))
        cor[i, j] = round(scipy.stats.pearsonr(np.nan_to_num(wrap_data[:, i]), np.nan_to_num(wrap_data[:, j]))[0],6)
        cor[i,j] = np.nan_to_num(cor[i, j])
        cor[j, i] = cor[i, j]
        cov[i,j] = cor[i, j] * robust_scale[i] *robust_scale[j]
        cov[i,j] = np.nan_to_num(cov[i, j])
        cov[j,i] = cov[i,j]
    
    np.fill_diagonal(cor,1)
    cov = pd.DataFrame(cov)
    cor = pd.DataFrame(cor)
        
    return cov, cor 


def wrapped_dcor(data) : 
    """
    Computes the wrapped covariance and correlation of the data 
        Parameters
    -----------
            data : numpy ndarray, 
            
                The input data to wrap
    """
    
    
    wrap_data, robust_mean, robust_scale = wrap_dcov(data.copy())
    
    p = data.shape[1]
    cor = np.ones((p, p))
    indices = np.triu_indices(p, 0)
    
    for i, j in zip(indices[0], indices[1]):
        try:
            cor[i, j] = round(np.sqrt(dcor.distance_correlation_sqr(wrap_data[:, i], wrap_data[:, j])),6)
        except Exception as e:
            print(e)
            cor[i, j] = 0
            
        cor[i,j] = np.nan_to_num(cor[i, j])
        cor[j, i] = cor[i, j]
    
    np.fill_diagonal(cor,1)
    cor = pd.DataFrame(cor)
        
    return  cor 

def pairwise_dcor(data): 
    
    """
    Computes the distance correlation of the data 
        Parameters
    -----------
            data : numpy ndarray, 
            
                The input data to wrap
    """
    
    p = data.shape[1]
    cor = np.ones((p, p))
    indices = np.triu_indices(p, 0)
    
    for i, j in zip(indices[0], indices[1]):
        try:
            cor[i, j] = round(np.sqrt(dcor.distance_correlation_sqr(data[:, i], data[:, j])),6)
        except Exception as e:
            print(e)
            cor[i,j] = 0
        cor[i,j] = np.nan_to_num(cor[i, j])
        cor[j, i] = cor[i, j]
    
    np.fill_diagonal(cor,1)
    cor = pd.DataFrame(cor)
        
    return  cor 
    



