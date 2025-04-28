import numpy as np
import matplotlib.pyplot as plt

def predict(x, w, b):
    return x @ w + b

# mean-squared error loss function
def mse(x, y, w, b):
    y_pred = predict(x, w, b)
    errors = y_pred - y
    squared_errors = errors ** 2
    loss = squared_errors.mean()
    return loss

# perform regression
def regression(x, y, w, b, alpha, epoch):
    losses = []
    for iter in range(epoch):
        y_pred = predict(x,w,b)
        loss = mse(x,y,w,b)
        losses.append(loss)
                
        error = y - y_pred  
        
        gradient_w = (-2/len(x)) * (x.T @ error)
        gradient_b = ((-2/len(x)) * error.sum())

        w = w - alpha * gradient_w
        b = b - alpha * gradient_b
        
    return w, b, losses
       
# reproducibility
np.random.seed(42)

n_samples = 1000
n_features = 4

# create uniform points of shape (n_samples, n_features)
x = np.random.uniform(0, 20, size=(n_samples, n_features)) 

# true values to compute, shape (n_features, 1)
true_weights = np.array([2.5, -1, 3.7, 5.2])
true_intercept = 5.0

# add random noise normally distributed about [mean] with [sdev]
noise = np.random.normal(0, 3.0, size=(n_samples, 1))

# y = (n_samples,1) + (n_samples,1) + (1)    
y = (x @ true_weights).reshape(-1,1) + noise + true_intercept
  
w = np.zeros(n_features).reshape(-1,1)
b = 0.0

epoch = 10000
alpha = 0.0005

new_w, new_b, losses = regression(x,y,w,b,alpha,epoch)
print(f"Predicted Weights: {new_w}, Predicted Intercept: {new_b}")
print(f"Actual Weights: {true_weights}, Actual Intercept: {true_intercept}")
print(f"Final MSE: {losses[-1]}")

plt.plot(losses)
plt.show()