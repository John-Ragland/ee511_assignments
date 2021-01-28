import math

w = [0, 0, 0]
x = [[1, 1, 1, 1, 1, 1], [-1, -1, 2, 0, 1, 2], [5, 2, 5, -2, 0, 1]]
y = [-1, -1, -1, 1, 1, 1]
# x = [[1,1],[3,-2],[-3,2]]
# y = [1, 0]
rate = 0.1
iterations = 3
n = len(x[0])
for t in range(iterations):
    print('$$t = %d$$' % t)
    print('$$w = [%.2f, %.2f, %.2f]$$' % (w[0], w[1], w[2]))
    p = [0, 0, 0, 0, 0, 0]
    for j in range(0, n):
        p1 = math.exp(w[0] + (w[1] * x[1][j]) + (w[2] * x[2][j]))
        p2 = p1 / (p1 + 1)
        p[j] = p2
        print('$$\\text{{exp}}(%.2f+%.2f*%d+%.2f*%d) = %.2f \\rightarrow P(Y^%d=1| x^%d, w) = %.2f$$' \
            % (w[0], w[1], x[1][j], w[2], x[2][j], p1, j, j, p2))

    print('Loop over i, sum over j')

    grad = [0, 0, 0]
    for i in range(0, 3):
        for j in range(0, n):
            res = x[i][j] * (y[j] - p[j])
            grad[i] += res
            print('$$i=%d, j=%d: x_%d^%d(y^%d-P(Y^%d=1|x^%d, w)) = %d(%d-%.2f) = %.2f$$' \
                % (i, j, i, j, j, j, j, x[i][j], y[j], p[j], res))

    print('grad = $$[%.2f, %.2f, %.2f]$$' % (grad[0], grad[1], grad[2]))
    w[0] += rate * grad[0]
    w[1] += rate * grad[1]
    w[2] += rate * grad[2]
    print('$$η = %.2f \\rightarrow w^{t+1} = [%.2f, %.2f, %.2f]$$' % (rate, w[0], w[1], w[2]))
    print()