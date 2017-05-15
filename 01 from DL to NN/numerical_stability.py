x = 1e9; N = int(1e6);
delta = 1e-6; xi = x
for i in range(N):
    x += delta
x -= xi
print(x)
