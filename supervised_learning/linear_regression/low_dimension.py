import numpy as np
import matplotlib.pyplot as plt

def predict(x, w, b):
    return w * x + b

# mean-squared error loss function
def mse(x, y, w, b):
    y_pred = predict(x, w, b)
    errors = y_pred - y
    squared_errors = errors ** 2
    loss = squared_errors.mean()
    return loss

# perform regression
def regression(x, y, w, b, alpha, epoch):
    for iter in range(epoch):
        y_pred = predict(x,w,b)
        loss = mse(x,y,w,b)
        if (iter%(epoch/10) == 0): print(f"Epoch {iter}: {loss}")
        
        error = y-y_pred
        gradient_w = (-2/len(x)) * (x*error).sum()
        gradient_b = (-2/len(x)) * error.sum()
        
        w = w - alpha * gradient_w
        b = b - alpha * gradient_b
        
    return w, b
       
# reproducibility
np.random.seed(42)
n_samples = 1000

# create uniform points from [low] to [high]
x = np.random.uniform(0, 20, size=(n_samples, 1)) 

# true values to compute
true_slope = 2.5
true_intercept = 5

# add random noise normally distributed about [mean] with [sdev]
noise = np.random.normal(0, 3.0, size=(n_samples, 1))  
y = true_slope * x + true_intercept + noise
  
# params to predict
w = 0.0
b = 0.0

# number of iterations 
epoch = 10000
# learning rate
alpha = 0.001

new_w, new_b = regression(x,y,w,b,alpha,epoch)
print(f"Predicted Slope: {new_w}, Predicted Intercept: {new_b}")
print(f"Actual Slope: {true_slope}, Actual Intercept: {true_intercept}")

plt.scatter(x,y,s=1)
plt.axline((0,new_b), slope=new_w, color='red')
plt.axline((0,true_intercept), slope=true_slope, color='green')
plt.show()