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
        
        # if (iter % (epoch/10) == 0): print(f"Epoch {iter}: {loss:.4f}")
        
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

def min_max_scale(x):
    return (x - np.min(x)) / (np.max(x)-np.min(x))

np.random.seed(42)

n_samples = 10000
epochs = 1000
alpha = 0.001

true_weights = np.array([2.2, -3.1, 4.1, 9.7, -2.3, 8]) # coefficients of the fn, least significant first
true_bias = 5.0 # constant term 
poly_deg = len(true_weights)

x = np.linspace(-100, 100, n_samples) # uniform distribution of samples from [left, right]
x = min_max_scale(x)
x_poly = create_poly(x, poly_deg)

y = fn(x_poly, true_weights, true_bias) # apply fn(x)
y_noisy = y + np.random.randn(n_samples) * 4 # gaussian noise, mean 0, sdev 1 * [scale]

w_l1, b_l1, losses_l1 = polynomial_regression(x,y_noisy,poly_deg,alpha,epochs,mode=1,lambda_=0.1)
w_l2, b_l2, losses_l2 = polynomial_regression(x,y_noisy,poly_deg,alpha,epochs,mode=2,lambda_=0.1)
w_l3, b_l3, losses_l3 = polynomial_regression(x,y_noisy,poly_deg,alpha,epochs,mode=3,lambda_=0.1)

def plot_losses(losses_l1,losses_l2,losses_l3):
    plt.plot(losses_l1, color="red", label="L1")
    plt.plot(losses_l2, color="green", label="L2")
    plt.plot(losses_l3, color="blue", label="Elastic")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (log scale)")
    plt.yscale('log')
    plt.legend()
    plt.show()

def plot_all_models(x, y_actual, y_noisy, preds_dict):
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y_noisy, s=0.15, alpha=0.5, label='Noisy Data')  # scatter of noisy points
    plt.plot(x, y_actual, color='black', linewidth=2, label='True Function')  # true function
    for label, y_pred in preds_dict.items():
        plt.plot(x, y_pred, linewidth=2, label=label)  # each regression line
    plt.legend()
    plt.title("Polynomial Regression: True Function vs L1, L2, Elastic Net")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    
print(f"Predicted Weights (l1): {w_l1}, Predicted Intercept: {b_l1}")
print(f"Predicted Weights (l2): {w_l2}, Predicted Intercept: {b_l2}")
print(f"Predicted Weights (Elastic): {w_l3}, Predicted Intercept: {b_l3}")
print(f"Actual Weights: {true_weights}, Actual Intercept: {true_bias}")
print(f"L1 losses: {losses_l1[-1]}")
print(f"L2 losses: {losses_l2[-1]}")
print(f"Elastic losses: {losses_l3[-1]}")

y_regress_l1 = fn(x_poly, w_l1, b_l1)
y_regress_l2 = fn(x_poly, w_l2, b_l2)
y_regress_l3 = fn(x_poly, w_l3, b_l3)

predictions = {
    "L1 (Lasso)": y_regress_l1,
    "L2 (Ridge)": y_regress_l2,
    "Elastic Net": y_regress_l3
}

plot_losses(losses_l1, losses_l2, losses_l3)
plot_all_models(x, y, y_noisy, predictions)