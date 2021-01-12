import numpy as np

A = np.array([[15, 0, 1], [1,3,7], [0, 1, 6]])
b=np.array([[21, 67, 44]]).T
print(A)
print(b)
# [[15  0  1]
#  [ 1  3  7]
#  [ 0  1  6]]
# [[21]
#  [67]
#  [44]]

#a
print("\na)")
D=np.diag(np.diag(A))
R=np.triu(A)-D
L=np.tril(A)-D

print("L:")
print(L)
print("D:")
print(D)
print("R:")
print(R)
# L:
# [[0 0 0]
#  [1 0 0]
#  [0 1 0]]
# D:
# [[15  0  0]
#  [ 0  3  0]
#  [ 0  0  6]]
# R:
# [[0 0 1]
#  [0 0 7]
#  [0 0 0]]

#b
print("\nb)")
x = np.array([[0,0,0]]).T
# [[0]
#  [0]
#  [0]]
def F_gauss_seidel(x, L, D, R):
    return -np.linalg.inv(D + L) @ R @ x + np.linalg.inv(D + L) @ b

for i in np.arange(1,7):
    x = F_gauss_seidel(x, L, D, R)
    print(x)
# [[ 1.4       ]
#  [21.86666667]
#  [ 3.68888889]]
# [[ 1.15407407]
#  [13.34123457]
#  [ 5.10979424]]
# [[ 1.05934705]
#  [10.05736443]
#  [ 5.65710593]]
# [[1.0228596 ]
#  [8.7924663 ]
#  [5.86792228]]
# [[1.00880518]
#  [8.30524628]
#  [5.94912562]]
# [[1.00339163]
#  [8.11757634]
#  [5.98040394]]

#c
print("\nc)")
#Bei grossen Gleichungsystemen ist der numerische Aufwand für die exakte Lösung zu gross.
#Iterative Verfahren sind dann effizienter