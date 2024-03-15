import numpy as np
import sympy as sp

# 四參數坐標轉換

def calculate_affine_params(original_pts:list, converted_pts:list):
    # 四參數坐標轉換
    # 轉換公式：
    # x' = S(x*cos(theta) - y*sin(theta)) + Tx
    # y' = S(x*sin(theta) - y*cos(theta)) + Ty

    # 其中，
    # x, y: 原始坐標
    # X, Y: 轉換後坐標
    # Tx, Ty: 平移參數
    # S: 尺度參數
    # theta: 旋轉參數
    
    original_pts = np.array(original_pts)
    converted_pts = np.array(converted_pts)

    #確認點數量是否有2個以上
    if original_pts.shape[0] < 3:
        return False, print("Coordinate points number less than three.")
    
    #確認轉換前後點數量是否一致
    if original_pts.shape[0] != converted_pts.shape[0]:
        return False, print("Coordinate points number are incosistent.")
    
    #定義觀測方程式 & 未知參數
    Tx, Ty, S, theta, x, y, X, Y = sp.symbols('Tx Ty S theta x y X Y')

    fx = S*(x * sp.cos(theta) - y * sp.sin(theta)) + Tx - X
    fy = S*(x * sp.sin(theta) + y * sp.cos(theta)) + Ty - Y
    variable = [Tx, Ty, S, theta]


    #將已知觀測量x, y, X, Y代入觀測方程式
    n = original_pts.shape[0]
    F = sp.Matrix([])

    for i in range(0, n):
        f1 = fx.subs({x: original_pts[i,0], y: original_pts[i,1], X: converted_pts[i,0], Y: converted_pts[i,1]})
        f2 = fy.subs({x: original_pts[i,0], y: original_pts[i,1], X: converted_pts[i,0], Y: converted_pts[i,1]})
        F = F.col_join(sp.Matrix([f1])).col_join(sp.Matrix([f2]))
    
    B0 = F.jacobian(variable)   #B矩陣
    X0 = np.array([0.0, 0.0, 1.0, 0.0]) #給定未知參數初始值

    #定義權矩陣 (假設為等權)
    W = np.diag(np.ones(2*n,))

    count = 0
    deltaX = np.ones((4,1), dtype = 'float64')
    
    #間接觀測平差迭代計算
    while np.linalg.norm(deltaX) > 0.000001:
        B = B0.subs({Tx: X0[0], Ty: X0[1], S: X0[2], theta: X0[3]})
        B = np.array(B, dtype = 'float64')
        f = -F.subs({Tx: X0[0], Ty: X0[1], S: X0[2], theta: X0[3]})
        f = np.array(f, dtype = 'float64')
    
        N = np.transpose(B).dot(W).dot(B)
        t = np.transpose(B).dot(W).dot(f)
        deltaX = np.linalg.inv(N).dot(t)

        deltaX = np.squeeze(deltaX)
        
        X0 += deltaX
        count += 1

        if count >= 10: #停損
            break
    print("迭代次数:", count)

    final_X = np.around(X0, 3)
    
    return final_X



def coord_affine_trans(four_param:np.array, original_pts:list):
    # 使用解算的四參數進行坐標轉換

    original_pts = np.array(original_pts)

    n = original_pts.shape[0]
    converted_pts = np.zeros((n,2))

    #定義觀測方程式 & 未知參數
    Tx, Ty, S, theta, x, y = sp.symbols('Tx Ty S theta x y')
    fx = S*(x * sp.cos(theta) - y * sp.sin(theta)) + Tx
    fy = S*(x * sp.sin(theta) + y * sp.cos(theta)) + Ty

    for i in range(0, n):
        subs_dict = {Tx: four_param[0], Ty: four_param[1], S: four_param[2], theta: four_param[3], 
                                               x: original_pts[i,0], y: original_pts[i,1]}
        converted_pts[i,0] = fx.subs(subs_dict)
        converted_pts[i,1] = fy.subs(subs_dict)
    
    converted_pts = np.around(converted_pts, 3).tolist()
    
    return converted_pts


#解算四參數
original_pts = [[5297.08, -5277.02], [5288.72, -259.99], [109.53, -278.9], [121.56, -5292.9]] 
converted_pts = [[106.057, 105.967], [-106.056, 106.039], [-105.953, -106.041], [105.948, -105.963]]
original_pts = np.array(original_pts)
converted_pts = np.array(converted_pts)
params = calculate_affine_params(original_pts, converted_pts)

print(params)

#計算未知坐標
ori_pts = [[1010.660, -3025.660], [657.820, -2856.550], [1264.430, -738.400],
           [4328.460, -2684.230], [2511.780, -4541.090]] #原始坐標

converted_pts = coord_affine_trans(params, ori_pts)  #轉換後坐標
print(converted_pts)

