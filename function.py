import matplotlib.pyplot as plt
import numpy as np


class isHI_WDRO:
    def __init__(self, input_dim):
        self.input_dim = input_dim
    def update(self, config = {}):
        if 'eps' in config.keys():
            self.eps = config['eps']
    #The PAGE estimater
    def PAGE(self, itr, Xi_1, Xi_2, x_1, x_2, gradient):
        sample_size, __ = Xi_1.shape
        a = 2
        eps = 0.1
        pk = 1 - np.power(itr/(itr+1), 2*a)/(2 - np.power(itr/(itr+1), 2*a+1))
        N1 = min(np.ceil(2*np.power(eps**2*(itr+1), 2*a)), sample_size)
        N2 = min(np.ceil(2*self.L0**2*np.sum(np.power(Xi_2-Xi_1, 2))*(eps**2*(itr+1)**(2*a+1))), sample_size)
        p = np.random.rand(1)

        
        output = np.zeros(Xi_1.shape)
        if p <= pk:
            selected_rows = np.random.choice(sample_size, int(N1), replace=False) 
            for rows in selected_rows:
                output[rows, :] = (-x_2 + Xi_2[rows, :]*np.exp(-np.sum(np.power(Xi_2[rows, :], 2)))-Xi_2[rows, :])/N1
            return output
        else:
            selected_rows = np.random.choice(sample_size, int(N2), replace=False) 
            for rows in selected_rows:
                output[rows, :] = (-x_2 + Xi_2[rows, :]*np.exp(-np.sum(np.power(Xi_2[rows, :], 2)))-Xi_2[rows, :])/N2 - (-x_1 + Xi_1[rows, :]*np.exp(-np.sum(np.power(Xi_1[rows, :], 2)))-Xi_1[rows, :])/N2
            return output + gradient
    def fit(self, X):
        # alpha = 1, L_0 = 2
        sample_size, __ = X.shape
        L = 2
        self.L0 = L
        alpha = 1
        eta = alpha*(4-alpha*L)/4
        itr = 0
        loss = []
        sample = []
        decision = []
        unchange_loss = np.sum(np.exp(-np.sum(np.power(X, 2), 1)))/sample_size/2
        
        x_0 = np.random.rand(self.input_dim)
        temp = x_0 @ x_0/2 - np.sum(X, axis=0)@x_0/sample_size - unchange_loss
        loss.append(temp)
        decision.append(x_0)
        sample.append(X)
        x_1 = -1/2*(x_0 - np.sum(X, axis=0)/sample_size)
        decision.append(x_1)
        temp = x_1 @ x_1/2 - np.sum(X, axis=0)@x_1/sample_size - unchange_loss
        loss.append(temp)
        
        Xi_0 = X
        gd = (-x_0 + np.exp(-np.sum(np.power(Xi_0, 2), 1))[:, np.newaxis]* Xi_0-Xi_0)/sample_size
        error = np.sum(np.power((-x_0 + np.exp(-np.sum(np.power(Xi_0, 2), 1))[:, np.newaxis]* Xi_0-Xi_0)/sample_size, 2))
        if sample_size*np.power(self.eps, 2) >= error:
            Xi_1 = Xi_0 + 1/(2*sample_size)*(-x_0 + Xi_0*np.exp(-np.sum(np.power(Xi_0, 2), 1))[:, np.newaxis]-Xi_0) / eta /alpha
        else:
            Xi_1 = Xi_0 + np.power(sample_size, 1/2) * self.eps/np.power(error, 1/2) * (-x_0+Xi_0*np.exp(-np.sum(np.power(Xi_0, 2), 1))[:, np.newaxis]-Xi_0)/sample_size/2 / eta/alpha
        sample.append(Xi_1)

        
        itr = 1
        x_2 = x_0/(itr+2) + (itr+1)/(itr+2) * x_1 - (itr+1)/(itr+2) / eta * (x_1 - np.sum(Xi_1, axis = 0)/sample_size)
        
        gd = self.PAGE(itr, Xi_0, Xi_1, x_0, x_1, gd)
        error = np.sum(np.power(Xi_1 + gd - Xi_0, 2))
        if sample_size*np.power(self.eps, 2) >= error:
            Xi_2 = Xi_0/(itr+2) + (itr+1)/(itr+2) * Xi_1 + (itr+1)/(itr+2)/eta/alpha * gd
        else:
            Xi_2 = (1/(itr+2) + (itr+1)/(itr+2)/eta/alpha) * Xi_0 + ((itr+1)/(itr+2) - (itr+1)/(itr+2)/eta/alpha) * Xi_1 + (itr+1)/(itr+2)/eta/alpha * np.power(sample_size, 1/2) * self.eps/np.power(error, 1/2) * (Xi_1 + gd - Xi_0)

        count = 2
        temp = x_2 @ x_2/2 - np.sum(X, axis=0)@x_2/sample_size - unchange_loss
        loss.append(temp)
        sample.append(Xi_2)
        decision.append(x_2)
        while np.sum(np.power(x_2 - x_1, 2)) + np.sum(np.power(Xi_1-Xi_2, 2))/sample_size >= 5*1e-3:
            gd = self.PAGE(itr, Xi_1, Xi_2, x_1, x_2, gd)
            
            x_3 = x_0/(itr+2) + (itr+1)/(itr+2) * x_2 - (itr+1)/(itr+2)/eta * (x_2 - np.sum(Xi_2, axis = 0)/sample_size)
            
            error = np.sum(np.power(Xi_2 + gd - Xi_0, 2))
            
            if sample_size*np.power(self.eps, 2) >= error:
                Xi_3 = Xi_0/(itr+2) + (itr+1)/(itr+2) * Xi_2 + (itr+1)/(itr+2)/eta/alpha * gd
            else:
                Xi_3 = Xi_0/(itr+2) + ((itr+1)/(itr+2) - (itr+1)/(itr+2)/eta/alpha) * Xi_2 + (itr+1)/(itr+2) /eta/alpha * (np.power(sample_size, 1/2) * self.eps/np.power(error, 1/2) * (Xi_2 + gd - Xi_0) + Xi_0)
            Xi_1 = Xi_2
            x_1 = x_2
            Xi_2 = Xi_3
            x_2 = x_3
            itr += 1
            temp = x_3 @ x_3/2 - np.sum(X, axis=0)@x_3/sample_size - unchange_loss
            loss.append(temp)
            sample.append(Xi_3)
            decision.append(x_3)
            count += 1
          
        self.theta = x_3
        self.worst = Xi_3
        self.loss = loss
        self.decision = decision
        self.sample = sample
        return self.theta

    def worst_distribution(self):
        return self.worst
     
    def loss_values(self):
        return self.loss
    def decision_values(self):
        return self.decision
    def sample_points(self):
        return self.sample
