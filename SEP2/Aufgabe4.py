import numpy as np
import matplotlib.pyplot as plt

#a
b = np.array([[21, 67, 44]], dtype=np.float64).T

L = np.array([
    [0,0,0],
    [1,0,0],
    [0,1,0]
], dtype=np.float64)

D = np.array([
    [15,0,0],
    [0,3,0],
    [0,0,6]
], dtype=np.float64)

R = np.array([
    [0,0,1],
    [0,0,7],
    [0,0,0]
], dtype=np.float64)

#b
def F_gauss_seidel(x, b, L, D, R):
    return -np.linalg.inv(D + L) @ R @ x + np.linalg.inv(D + L) @ b
x = np.array([[0,0,0]]).T
for i in range(6):
    x1 = F_gauss_seidel(x, b, L, D, R)
    print(str(i) + " = ")
    print(x1)
    x = x1

# 0 = 
# [[ 1.4       ]
#  [21.86666667]
#  [ 3.68888889]]
# 1 = 
# [[ 1.15407407]
#  [13.34123457]
#  [ 5.10979424]]
# 2 = 
# [[ 1.05934705]
#  [10.05736443]
#  [ 5.65710593]]
# 3 = 
# [[1.0228596 ]
#  [8.7924663 ]
#  [5.86792228]]
# 4 = 
# [[1.00880518]
#  [8.30524628]
#  [5.94912562]]
# 5 = 
# [[1.00339163]
#  [8.11757634]
#  [5.98040394]]

#c das iterative lösungsverfahren ist bei grossen Matrizen performanter
# und kann allgemein umgesetzt werden. Beim direkten lösungsverfahren kommt
# man schnell an eine Grenze (wegen sehr hohen Zahlen) bei grösseren Matrizen