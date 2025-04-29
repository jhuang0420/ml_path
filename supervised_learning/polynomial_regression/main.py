import numpy as np
import matplotlib.pyplot as plt

def create_poly(x, degree):
    return np.column_stack([x**i for i in range(1, degree+1)])

def fn(x_poly, weights, bias):
    return x_poly @ weights + bias

def polynomial_regression(x, y, degree, alpha, epoch, mode=0, lambda_=0.01):
    x_poly = create_poly(x, degree)
    w = np.zeros(degree)
    b = 0.0
    losses = []
    
    for iter in range(epoch):
        y_pred = fn(x_poly,w,b)
        
        error = y-y_pred
        loss = (error ** 2).mean()
        losses.append(loss)
        
        if (iter % (epoch/10) == 0): print(f"Epoch {iter}: {loss:.4f}")
        
        gradient_w = -2/len(x) * (x_poly.T @ error)
        gradient_b = -2/len(x) * error.sum()
        
        # L1 Regularization
        if (mode == 1): gradient_w += lambda_ * np.sign(w)
          
        # L2 Regularization
        elif (mode == 2): gradient_w += lambda_ * 2 * lambda_ * w
            
        # L1 + L2 Regularization
        elif (mode == 3): gradient_w += lambda_ * 2 * lambda_ * w + lambda_ * np.sign(w)
            
        w -= alpha * gradient_w
        b -= alpha * gradient_b
        
    return w,b,losses

np.random.seed(42)

n_samples = 1000
epochs = 100000
alpha = 0.0001

true_weights = np.array([2.2, -3.1, 5.1, 0.4]) # coefficients of the fn, least significant first
true_bias = 5.0 # constant term 
poly_deg = len(true_weights)

x = np.linspace(-3, 3, n_samples) # uniform distribution of samples from [left, right]
x_poly = create_poly(x, poly_deg)

y = fn(x_poly, true_weights, true_bias) # apply fn(x)
y_noisy = y + np.random.randn(n_samples) * 2 # gaussian noise, mean 0, sdev 1 * [scale]

new_w, new_b, losses = polynomial_regression(x,y_noisy,poly_deg,alpha,epochs,mode=1,lambda_=0.02)

print(f"Predicted Weights: {new_w}, Predicted Intercept: {new_b}")
print(f"Actual Weights: {true_weights}, Actual Intercept: {true_bias}")

y_regress = fn(x_poly, new_w, new_b)

plt.scatter(x, y_noisy, s=0.15) 
plt.plot(x, y, color='red', linewidth=2) # true fn
plt.plot(x, y_regress, color='green', linewidth=2) # predicted fn
plt.show()