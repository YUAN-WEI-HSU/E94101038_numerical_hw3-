import numpy as np
from scipy.interpolate import lagrange
from math import factorial, cos, sin

# 給定的數據點
x_values = np.array([0.698, 0.733, 0.768, 0.803])
y_values = np.array([cos(x) for x in x_values])  # 真值來自 cos(x)
x_target = 0.750  # 欲估算的 x

# 計算誤差界限
def error_bound(x_vals, x_target, degree):
    """ 計算 Lagrange 插值誤差界限 """
    # 計算 (n+1) 階導數的最大值 (cos(x) 的導數最大絕對值為 1)
    max_derivative = 1  # 因為 cos(x) 的高階導數最大值為 ±1

    # 計算 |Π (x - xi)|
    product_term = np.prod([abs(x_target - x) for x in x_vals[:degree+1]])

    # 計算誤差界限
    error = (max_derivative / factorial(degree+1)) * product_term
    return error

# Lagrange 插值計算
def lagrange_interpolation(x_vals, y_vals, x_target, degree):
    """ 使用 Lagrange 插值計算 x_target 的近似值 """
    # 選取最接近 x_target 的 (degree+1) 個點
    idx = np.argsort(np.abs(x_vals - x_target))[:degree+1]
    x_subset = x_vals[idx]
    y_subset = y_vals[idx]

    # 建立 Lagrange 多項式
    poly = lagrange(x_subset, y_subset)

    # 計算近似值與誤差界限
    approx_value = poly(x_target)
    error = error_bound(x_subset, x_target, degree)

    return approx_value, error

# 計算不同階數的插值結果及誤差界限
true_value = cos(x_target)  # 真實的 cos(0.750)

for deg in range(1, 4):  # 一次到四次插值
    approx_value, error = lagrange_interpolation(x_values, y_values, x_target, deg)
    absolute_error = abs(true_value - approx_value)  # 真實誤差
    print(f"{deg} 次插值結果: P_{deg}({x_target}) ≈ {approx_value:.6f}, 誤差界限 ≤ {error:.6e}, 真實誤差 = {absolute_error:.6e}")
