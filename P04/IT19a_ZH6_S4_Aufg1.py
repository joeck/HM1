# A
x = 0
def F(z):
    return (230 * z**4 + 18 * z**3 + 9 * z**2 - 9)/221
before = 0.1
count = 0

while (abs(x-before) > 10**-6 and count < 50):
    before = x
    print("x" + str(count) + ": " + str(x))
    x = F(x)
    count += 1

print("x" + str(count) + ": " + str(x))
print("count: " + str(count))

# Die nullstelle in [0,1] wird nicht gefunden, da sie abstossend ist

# B
# x = -0.04065928897359839
# [a,b] = [-0.1, 0]
a = -0.1
b = 0
#print(a <= F(-0.1) <= b)
#print(a <= F(0) <= b)

def F1(z):
    return (920 * z**3 + 54 * z**2 + 18 * z)/221

aplha = abs(F1(0.5))
print(aplha)

#apriori
x0 = F(0)
x1 = F(x0)
l = abs(0-aplha)
r = 1/(1-aplha) * abs(x1 - x0)
apriori = l <= r

print("apriori: " + str(apriori))
print(F(0.5))
print(F1(0.5))