import numpy as np
import sympy as sp

# 七參數坐標轉換


def rotation_matrix(wx, wy, wz):
# 定義三個旋轉矩陣相乘
    Rx = sp.Matrix([[1, 0, 0],
                   [0, sp.cos(wx), sp.sin(wx)],
                   [0, -sp.sin(wx), sp.cos(wx)]])
    Ry = sp.Matrix([[sp.cos(wy), 0, -sp.sin(wy)],
                   [0, 1, 0],
                   [sp.sin(wy), 0, sp.cos(wy)]])
    Rz = sp.Matrix([[sp.cos(wz), sp.sin(wz), 0],
                   [-sp.sin(wz), sp.cos(wz), 0],
                   [0, 0, 1]])
    R = Rx*Ry*Rz
    return R.T


def calculate_seven_params(original_pts:list, converted_pts:list):
    # 七參數坐標轉換
    # Bursa轉換公式：
    # |X|                             |x|  |Tx|
    # |Y| = (1+m)R_1(wx)R_2(wy)R_3(wz)|y|+ |Ty|
    # |Z|                             |z|  |Tz|
    # 其中，
    # x, y, z: 原始坐標
    # X, Y, Z: 轉換後坐標
    # Tx, Ty, Tz: 平移參數
    # m: 尺度參數
    # wx, wy, wz: 旋轉參數
    
    original_pts = np.array(original_pts)
    converted_pts = np.array(converted_pts)

    #確認點數量是否有3個以上
    if original_pts.shape[0] < 4:
        return False, print("Coordinate points number less than three.")
    
    #確認轉換前後點數量是否一致
    if original_pts.shape[0] != converted_pts.shape[0]:
        return False, print("Coordinate points number are incosistent.")
    
    #定義觀測方程式 & 未知參數
    Tx, Ty, Tz, m, wx, wy, wz, x, y, z, X, Y, Z = sp.symbols('Tx Ty Tz m wx wy wz x y z X Y Z')
    R = rotation_matrix(wx, wy, wz)
    fx = (1 + m) * (R[0,0]*x + R[0,1]*y + R[0,2]*z) + Tx - X
    fy = (1 + m) * (R[1,0]*x + R[1,1]*y + R[1,2]*z) + Ty - Y
    fz = (1 + m) * (R[2,0]*x + R[2,1]*y + R[2,2]*z) + Tz - Z
    variable = [Tx, Ty, Tz, m, wx, wy, wz]

    #將已知觀測量x, y, X, Y代入觀測方程式
    n = original_pts.shape[0]
    F = sp.Matrix([])

    for i in range(0, n):
        f1 = fx.subs({x: original_pts[i,0], y: original_pts[i,1], z: original_pts[i,2], 
                      X: converted_pts[i,0], Y: converted_pts[i,1], Z: converted_pts[i,2]})
        f2 = fy.subs({x: original_pts[i,0], y: original_pts[i,1], z: original_pts[i,2], 
                      X: converted_pts[i,0], Y: converted_pts[i,1], Z: converted_pts[i,2]})
        f3 = fz.subs({x: original_pts[i,0], y: original_pts[i,1], z: original_pts[i,2], 
                      X: converted_pts[i,0], Y: converted_pts[i,1], Z: converted_pts[i,2]})
        
        F = F.col_join(sp.Matrix([f1])).col_join(sp.Matrix([f2])).col_join(sp.Matrix([f3]))

    B0 = F.jacobian(variable)   
    X0 = np.array([-730.160, -346.212, -472.186, 0.9999, 0.0, 0.0, 0.0])

    #定義權矩陣 (假設為等權)
    W = np.diag(np.ones(3*n,))

    count = 0
    deltaX = np.ones((7,1), dtype = 'float64')
    
    #間接觀測平差迭代計算
    while np.linalg.norm(deltaX) > 0.000001:
        B = B0.subs({Tx: X0[0], Ty: X0[1], Tz: X0[2], m: X0[3], wx: X0[4], wy: X0[5], wz: X0[6]})
        B = np.array(B, dtype = 'float64')
        f = -1*F.subs({Tx: X0[0], Ty: X0[1], Tz: X0[2], m: X0[3], wx: X0[4], wy: X0[5], wz: X0[6]})
        f = np.array(f, dtype = 'float64')
    
        N = np.transpose(B).dot(W).dot(B)
        t = np.transpose(B).dot(W).dot(f)
        deltaX = np.linalg.inv(N).dot(t)

        deltaX = np.squeeze(deltaX)
        
        X0 += deltaX
        count += 1

        if count >= 10:
            break
    print("迭代次数:", count)

    final_X = np.around(X0, 3)
    
    return final_X


# 使用解算的七參數進行坐標轉換
def coord_affine_trans(seven_param:np.array, original_pts:list):
    
    original_pts = np.array(original_pts)
    
    n = original_pts.shape[0]

    converted_pts = np.zeros((n,3))

    #定義觀測方程式 & 未知參數
    Tx, Ty, Tz, m, wx, wy, wz, x, y, z = sp.symbols('Tx Ty Tz m wx wy wz x y z')
    R = rotation_matrix(wx, wy, wz)
    fx = (1 + m) * (R[0,0]*x + R[0,1]*y + R[0,2]*z) + Tx
    fy = (1 + m) * (R[1,0]*x + R[1,1]*y + R[1,2]*z) + Ty 
    fz = (1 + m) * (R[2,0]*x + R[2,1]*y + R[2,2]*z) + Tz 

    for i in range(0, n):
        subs_dict = {Tx: seven_param[0], Ty: seven_param[1], Tz: seven_param[2], m: seven_param[3], 
                     wx: seven_param[4], wy: seven_param[5], wz: seven_param[6],
                     x: original_pts[i,0], y: original_pts[i,1], z: original_pts[i,2]}
        
        converted_pts[i,0] = fx.subs(subs_dict)
        converted_pts[i,1] = fy.subs(subs_dict)
        converted_pts[i,2] = fz.subs(subs_dict)

    converted_pts = np.around(converted_pts, 3).tolist()
    
    return converted_pts




#解算七參數 (已TWD67 -> TWD97為例)
original_pts =   [[274040.686, 2586564.018, 2156.937], [305411.430, 2636012.058, 614.486],
                  [311681.117, 2654585.256, 109.973], [291817.600, 2558402.938, 7.338]]
converted_pts = [[274869.216, 2586357.074, 2155.462], [306241.392, 2635805.371, 613.682], 
                 [312510.857, 2654379.319, 110.322], [292645.120, 2558196.844, 7.299]]

original_pts = np.array(original_pts)
converted_pts = np.array(converted_pts)
params = calculate_seven_params(original_pts, converted_pts)
print(params)


#計算未知坐標
ori_pts = [[293127.450,	  2585586.658,	 1333.863],
           [309649.284,	  2435022.868,	  482.735],
           [268491.766,	  2532886.335,	 1190.435],
           [239910.854,	  2481617.154,	  780.773],
           [249368.628,	  2541913.927,	 2931.209]] #原始坐標
converted_pts = coord_affine_trans(params, ori_pts)  #轉換後坐標
print(converted_pts)



    




