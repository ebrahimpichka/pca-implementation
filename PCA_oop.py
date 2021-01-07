import pandas as pd
import numpy as np



class PCA:
   
    def __init__(self,k_components):
        
        self.k_components = k_components
        self.fitted = False
    
    
    
    def fit(self,X):
        """ 
        calculates the covariance matrix of X, eigen values and eigen vectors of the covariance matrix

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        self : object
        """

        self.X = X
        
        # Normalizing data (Subtrackting the mean of each column from each element of that column)
        self.normalized_X_ = (self.X - self.X.mean()).copy()
        
        
        # Calculating Covariance Matrix
        self.covmat_ = self.normalized_X_.cov()
        
        
        # Calculating eigenVectors Matrix (Sorted by the eigenValues)
        self.eigen_values_ , self.eigen_vectors_ = np.linalg.eig(self.covmat_)
        self._sorted_indx = np.argsort(self.eigen_values_)[::-1]
        self.sorted_eigen_vectors_ = self.eigen_vectors_[self._sorted_indx]
        self.sorted_eigen_values_ = self.eigen_values_[self._sorted_indx]
        
        self.fitted = True
        
        return(self)

    
    
    
    def transform(self,X):

        """ 
        calculates the covariance matrix of X, eigen values and eigen vectors of the covariance matrix
        then calculates the K components of X (first K eigen vectors whith the greatest eigen values)

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        transformed_X : K components of X (first K eigen vectors whith the greatest eigen values)
        """
        
        if not self.fitted:
            raise Exception('No data was fitted to PCA')            
            
        else:    
            normalized_X = (X - X.mean()).copy()
            
            self.transformed_X = normalized_X.dot(self.sorted_eigen_vectors_)
            
            return(self.transformed_X.iloc[:,:self.k_components])
    
    
    def fit_transform(self,X):
        """ 
        calculates the K components of X

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        transformed_X : K components of X (first K eigen vectors whith the greatest eigen values)
        """
        self.fit(X)
        self.transformed_X = self.normalized_X.dot(self.sorted_eigen_vectors_)
        return(self.transformed_X.iloc[:,:self.k_components])
    
            