import numpy as np
from scipy.interpolate import CubicHermiteSpline

# 原始資料
T = np.array([0, 3, 5, 8, 13])    # 時間 (秒)
D = np.array([0, 200, 375, 620, 990])  # 距離 (英尺)
V = np.array([75, 77, 80, 74, 72])   # 速度 (英尺/秒)

# 創建Hermite插值函數（同時使用位置和速度資料）
hermite_spline = CubicHermiteSpline(T, D, V)

# 問題a: 預測t=10秒時的位置和速度
t_eval = 10
position_10 = hermite_spline(t_eval)
speed_10 = hermite_spline.derivative()(t_eval)

# 問題b: 檢查是否超速 (55 mi/h = 80.6667 ft/s)
speed_limit = 55 * 5280 / 3600
t_fine = np.linspace(0, 13, 1000)
speeds = hermite_spline.derivative()(t_fine)

# 尋找超速時刻（考慮數值精度）
exceeds = np.where(speeds > speed_limit + 1e-6)[0]  # 加入小量避免浮點誤差
if len(exceeds) > 0:
    first_exceed_idx = exceeds[0]
    first_exceed_time = t_fine[first_exceed_idx]
else:
    first_exceed_time = None

# 問題c: 計算最高速度
max_speed = np.max(speeds)
max_speed_time = t_fine[np.argmax(speeds)]


# 印出結果
print("=== 問題a ===")
print(f"在 t=10秒 時:")
print(f"  預測位置 = {position_10:.1f} 英尺")
print(f"  預測速度 = {speed_10:.2f} ft/s ({speed_10*3600/5280:.1f} mi/h)")

print("\n=== 問題b ===")
if first_exceed_time:
    print(f"汽車在 {first_exceed_time:.2f} 秒時首次超速")
    print(f"  超速時速度 = {speeds[first_exceed_idx]:.2f} ft/s ({speeds[first_exceed_idx]*3600/5280:.1f} mi/h)")
else:
    print("汽車未超過限速")

print("\n=== 問題c ===")
print(f"預測最高速度 = {max_speed:.2f} ft/s ({max_speed*3600/5280:.1f} mi/h)")
print(f"出現在 {max_speed_time:.2f} 秒")