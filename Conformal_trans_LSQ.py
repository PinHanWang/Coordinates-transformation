import numpy as np


# 四參數坐標轉換
# 轉換公式：
# x' = S(x*cos(theta) - y*sin(theta)) + Tx
# y' = S(x*sin(theta) - y*cos(theta)) + Ty

# 其中，
# x, y: 原始坐標
# x', y': 轉換後坐標
# Tx, Ty: 平移參數
# S: 尺度參數
# theta: 旋轉參數
def calculate_affine_params(original_pts:list, converted_pts:list):

    original_pts = np.array(original_pts)
    converted_pts = np.array(converted_pts)

    #確認點數量是否有2個以上
    if original_pts.shape[0] < 3:
        return False, print("Coordinate points number less than three.")
    
    #確認轉換前後點數量是否一致
    if original_pts.shape[0] != converted_pts.shape[0]:
        return False, print("Coordinate points number are incosistent.")
    


    
    n = original_pts.shape[0]   #坐標點數量
    B = np.zeros((n*2, 4))      #B矩陣
    L = np.zeros((n*2, 1))      #L矩陣
    X = np.zeros((4, 1))        #未知參數矩陣
        
    #觀測量代入B & L矩陣
    for i in range(0, n):
        B[2 * i, 0] = 1.0
        B[2 * i, 1] = 0.0
        B[2 * i, 2] = original_pts[i,0]
        B[2 * i, 3] = -original_pts[i,1]
        B[2 * i + 1, 0] = 0.0
        B[2 * i + 1, 1] = 1.0
        B[2 * i + 1, 2] = original_pts[i,1]
        B[2 * i + 1, 3] = original_pts[i,0]

        L[2 * i] = converted_pts[i,0]
        L[2 * i + 1] = converted_pts[i,1]

    #最小二乘平差
    X = np.linalg.inv(np.transpose(B).dot(B)).dot(np.transpose(B).dot(L))

    #平差後四參數：
    final_X = np.zeros((4,1))
    final_X[0] = X[0]
    final_X[1] = X[1]
    final_X[2] = np.arctan2(X[3], X[2])
    final_X[3] = np.sqrt(X[2]**2 + X[3]**2)

    return final_X

# 使用解算的四參數進行坐標轉換
def coord_affine_trans(four_param:np.array, original_pts:list):
    
    original_pts = np.array(original_pts)
    
    n = original_pts.shape[0]
    Tx = four_param[0][0]
    Ty = four_param[1][0]
    theta = four_param[2][0]
    S = four_param[3][0]

    converted_pts = np.zeros((n,2))
    for i in range(0, n):
        converted_pts[i,0] = Tx + S*np.cos(theta)*original_pts[i,0] - S*np.sin(theta)*original_pts[i,1]
        converted_pts[i,1]  = Ty + S*np.sin(theta)*original_pts[i,0] + S*np.cos(theta)*original_pts[i,1]  
    
    converted_pts = converted_pts.tolist()
    
    return converted_pts



#實際計算四參數
old_pts = [[5297.08, -5277.02], [5288.72, -259.99], [109.53, -278.9], [121.56, -5292.9]]    #原始坐標
new_pts = [[106.057, 105.967], [-106.056, 106.039], [-105.953, -106.041], [105.948, -105.963]]  #轉換後坐標
param = calculate_affine_params(old_pts, new_pts)   #四參數


#計算未知坐標
ori_pts = [[1010.660, -3025.660], [657.820, -2856.550], [1264.430, -738.400], [4328.460, -2684.230], [2511.780, -4541.090]] #原始坐標
converted_pts = coord_affine_trans(param, ori_pts)  #轉換後坐標
print(converted_pts)