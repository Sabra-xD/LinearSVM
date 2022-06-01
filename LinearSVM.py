import numpy as np

class SVM:
    def __init__(self,learning_rate=0.001,lambda_param=0.01,n_iters=1000):
         self.lr = learning_rate
         self.lambda_param = lambda_param
         self.w = None
         self.n_iters = n_iters
         self.b = None
    
    def  fit(self,X,y):
        y_ = np.where(y <= 0,-1, 1)
        n_samples,n_features = X.shape

        self.w = np.zeros(n_features) #For each feature component we put a zero.
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X): #This will give me the current index and current sample
              condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
              if condition:
                  self.w -= self.lr * (2 * self.lambda_param * self.w)   # w_new = w_old - learning rate * dw
                    
              else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))  # w_new = w_old - learning rate * dw
                    self.b -= self.lr * y_[idx]
                
                
               

    
        
    def predict(self,X):
        linear_output = np.dot(X,self.w) - self.b
        return np.sign(linear_output)

