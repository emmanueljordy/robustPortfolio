## robustPortfolio

This repository contains the code for an application of the article: "Portfolio optimization using cellwise robust association measures and clustering methods with application to highly volatile markets."



## Abstract of the article 

This paper introduces the minCluster portfolio, which is a portfolio optimization method combining the optimization of downside risk measures, hierarchical clustering and cellwise robustness. Using cellwise robust association measures, the minCluster portfolio is able to retrieve the underlying hierarchical structure in the data. Furthermore, it provides downside protection by using tail risk measures for portfolio optimization. We show through simulation studies and a real data example that the minCluster portfolio produces better out-of-sample results than mean-variances or other hierarchical clustering based approaches. Cellwise outlier robustness makes the minCluster method particularly suitable for stable optimization of portfolios in highly volatile markets, such as portfolios containing cryptocurrencies.


## Starting point:  

- Python version: 3.7 or later.
- numpy 
- pandas
- scipy 
- dcor 

## How to proceed: 


Each folder in src contains the implementation of a submodule necessary to run the data example.  

## References

* Menvouta, E. J., Serneels, S., & Verdonck, T. (2023). Portfolio optimization using cellwise robust association measures and clustering methods with application to highly volatile markets. The Journal of Finance and Data Science, 9, 100097.
