import numpy as np

class Gaussian_process():

    def __init__(self, x_data, y_data, x_pred, sigma=0.001):
        """
        Input:
            x_data (np.array): NxM matrix of x-values (N rows with a fingerprint, each of M dimensions)
            y_data (np.array): N vector of y-values 
            x_pred (np.array): NxM matrix of x-values to get predicted y values for
        """
        self.x_data = x_data
        self.y_data = y_data
        self.x_pred = x_pred
        self.sigma = sigma
        return None

    ### Defining kernel and kernel matrix
    #Kernel function
    def kernelf(self, x, xprime, l, k0=1):
        """The squared exponential as basis function/kernel"""
        return k0 * np.exp(np.sum(-(x-xprime)**2) / (2 * l**2))

    #Construct the kernel matrix
    def get_kernel_mat(self, l, k0=1):
        """Kernel matrix (covariance matrix) from x_grid and kernel function (using l-param) """
        return np.array([[self.kernelf(x,xprime,l,k0) for x in self.x_data] for xprime in self.x_data])

    #Construct C matrix (sigma has been defined when defining noice)
    def get_c_mat(self, K):
        """C matrix from sigma and K"""
        N = np.shape(K)[0] #K is symmetric, dimension is num of datapoints
        return K + self.sigma**2 * np.identity(N)

    #construct k vector
    def get_k_vec(self, x, l, k0=1):
        """K vector for given x, given x-val of data and l-param"""
        return np.array([self.kernelf(x, xprime, l, k0) for xprime in self.x_data])

    #Construct the prior average function (here just a constant)
    def get_yprior_av(self, xval, param1):
        """Defining a constant function as the prior average function"""
        return param1

    ###Functions for predictions
    def get_predictions(self, l, k0=1, cst_av=0):
        """Function to predict y-values by a gaussian process from an array of x-values
        
        Input: 
            l, sigma, k0 (floats): hyper parameters
            cst_av (float): average of data
        """
        N_data = len(self.x_data)
    
        #Elements only needed to be calculated once (C^(-1).(t-y_av_prior))
        K_mat = self.get_kernel_mat(l,k0)
        C_mat = self.get_c_mat(K_mat)
        yprior_av = self.get_yprior_av(0, cst_av)
        yprior_av_vec = np.ones(N_data) * yprior_av
        t_ypriorav_vec = self.y_data - yprior_av_vec
        #Solving inversion x=C^(-1).(y_data-y_av_prior) without inverting matrix (costly)
        C_inv_t = np.linalg.solve(C_mat, t_ypriorav_vec)
        print(f'in get_predictions(): calculated C_inv_t ...')

        #Make prediction for each x input
        y_predictions = []
        for i,x_val in enumerate(self.x_pred):
            k_vec = self.get_k_vec(x_val,l,k0)
            y_pred = yprior_av + k_vec.transpose() @ C_inv_t
            y_predictions.append(y_pred)
            
            if i % 10 == 0:
                print(f'in get_predictions(): Predicted {i} vals ...')
        
        return np.array(y_predictions)

    def get_variances(self, l, k0=1):
        
        K_mat = self.get_kernel_mat(l,k0)
        C_mat = self.get_c_mat(K_mat)
        C_mat_inv = np.linalg.inv(C_mat)
        
        #Make variance for each x input
        y_variances = []
        for x_val in self.x_pred:
            k_vec = self.get_k_vec(x_val,l,k0)
            kernel_val = self.kernelf(x_val,x_val,l,k0)
            y_variance = kernel_val - k_vec.transpose() @ C_mat_inv @ k_vec
            
            y_variances.append(y_variance)
        
        return np.array(y_variances)

