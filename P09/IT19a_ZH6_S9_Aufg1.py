import numpy  as np

A = np.array([
    [1., 0 ,2],
    [0, 1, 0],
    [10**-4, 0, 10**-4]
])

b = np.array([
    [1],
    [1],
    [0]
])

x = 0

b1 = np.array([
    [1],
    [1]
    [x]
])

#1a
Ainverse = np.linalg.inv(A)
condA = np.linalg.norm(A, np.inf) * np.linalg.norm(Ainverse, np.inf)
print("Aufgabe 1a")
print(f"A = \n{A}")
print(f"A_inverse = \n{Ainverse}")
print(f"||A|| = {np.linalg.norm(A, np.inf)}")
print(f"||A^-1|| = {np.linalg.norm(Ainverse, np.inf)}")
print(f"cond(A)={condA}")

#1b
print("\nAufgabe 1b")
x = np.linalg.solve(A, b)
print(f"x = \n{x}")