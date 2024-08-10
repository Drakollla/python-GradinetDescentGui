import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# Определение функции и её градиента
x1, x2 = sp.symbols('x1 x2')
f = 6*x1**2 - 4*x2*x1 + 3*x2**2 + 4*sp.sqrt(5)*(2*x2 + x1) + 22
grad_f = [sp.diff(f, x1), sp.diff(f, x2)]

# Преобразование функции и градиента в лямбда-функции для использования в NumPy
f_lambda = sp.lambdify((x1, x2), f, 'numpy')
grad_f_lambda = sp.lambdify((x1, x2), grad_f, 'numpy')

# Параметры градиентного спуска
epsilon = 0.01
alpha_no_halving = 0.01
alpha_halving = 10
beta = 0.5
gamma = 0.8
max_iterations = 1000
x0 = np.array([-2, 1], dtype=float)

def gradient_descent(f, grad_f, x0, epsilon, alpha, beta, gamma, max_iterations, step_halving=False):
    x = np.array(x0, dtype=float)
    iterations = 0
    values = []
    points = []
    alphas = [alpha]

    def modgrad(g):
        return np.sqrt(g[0]**2 + g[1]**2)

    while True:
        grad = np.array(grad_f(*x), dtype=float)
        new_x = np.array([0, 0], dtype=float)
        _alpha = alpha

        if step_halving:
            while True:
                new_x = x - _alpha * grad
                if f(*x) - f(*new_x) >= _alpha * beta * modgrad(grad)**2:
                    break
                _alpha *= gamma
                alphas.append(_alpha)
        else:
            new_x = x - alpha * grad

        values.append(f(*x))
        points.append(x.copy())

        if modgrad(grad) < epsilon or iterations >= max_iterations:
            break

        x = new_x
        iterations += 1

    return points, values, iterations, alphas

# Запуск градиентного спуска без дробления шага
points_no_halving, values_no_halving, iterations_no_halving, alphas_no_halving = gradient_descent(f_lambda, grad_f_lambda, x0, epsilon, alpha_no_halving, beta, gamma, max_iterations, step_halving=False)

# Запуск градиентного спуска с дроблением шага
points_halving, values_halving, iterations_halving, alphas_halving = gradient_descent(f_lambda, grad_f_lambda, x0, epsilon, alpha_halving, beta, gamma, max_iterations, step_halving=True)

# Визуализация результатов
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot([point[0] for point in points_no_halving], [point[1] for point in points_no_halving], label='Без дробления шага')
plt.scatter([point[0] for point in points_no_halving], [point[1] for point in points_no_halving], color='red')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title(f'Градиентный спуск без дробления шага\nИтерации: {iterations_no_halving}')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot([point[0] for point in points_halving], [point[1] for point in points_halving], label='С дроблением шага')
plt.scatter([point[0] for point in points_halving], [point[1] for point in points_halving], color='red')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title(f'Градиентный спуск с дроблением шага\nИтерации: {iterations_halving}')
plt.legend()

plt.tight_layout()
plt.show()
