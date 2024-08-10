import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from tkinter import Tk, Label, Entry, Button, StringVar, messagebox

def gradient_descent(f, grad_f, x0, learning_rate, epsilon, max_iterations, step_halving=False):
    x = np.array(x0, dtype=float)
    iterations = 0
    values = []
    points = []

    while True:
        grad = np.array(grad_f(*x), dtype=float)
        new_x = x - learning_rate * grad
        values.append(f(*x))
        points.append(x.copy())

        if step_halving:
            learning_rate /= 2

        if np.linalg.norm(f(*new_x) - f(*x)) < epsilon or iterations >= max_iterations:
            break

        x = new_x
        iterations += 1

    return points, values, iterations

def run_gradient_descent():
    function_str = function_var.get()
    epsilon = float(epsilon_var.get())
    x0 = float(x0_var.get())  # Начальная точка по x
    y0 = float(y0_var.get())  # Начальная точка по y
    learning_rate = float(learning_rate_var.get())
    max_iterations = int(max_iterations_var.get())

    # Определение функции и её градиента
    x, y = sp.symbols('x y')
    try:
        f = sp.lambdify((x, y), function_str, 'numpy')
        grad_f = sp.lambdify((x, y), [sp.diff(function_str, x), sp.diff(function_str, y)], 'numpy')
    except Exception as e:
        messagebox.showerror("Ошибка", f"Неверный формат функции: {e}")
        return

    # Градиентный спуск без дробления шага
    points_no_halving, values_no_halving, iterations_no_halving = gradient_descent(f, grad_f, [x0, y0], learning_rate, epsilon, max_iterations, step_halving=False)

    # Градиентный спуск с дроблением шага
    points_halving, values_halving, iterations_halving = gradient_descent(f, grad_f, [x0, y0], learning_rate, epsilon, max_iterations, step_halving=True)

    # Визуализация результатов
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot([point[0] for point in points_no_halving], [point[1] for point in points_no_halving], label='Без дробления шага')
    plt.scatter([point[0] for point in points_no_halving], [point[1] for point in points_no_halving], color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Градиентный спуск без дробления шага\nИтерации: {iterations_no_halving}')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot([point[0] for point in points_halving], [point[1] for point in points_halving], label='С дроблением шага')
    plt.scatter([point[0] for point in points_halving], [point[1] for point in points_halving], color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Градиентный спуск с дроблением шага\nИтерации: {iterations_halving}')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Создание основного окна
root = Tk()
root.title("Градиентный спуск")

# Создание переменных для хранения ввода пользователя
function_var = StringVar()
epsilon_var = StringVar()
x0_var = StringVar()  # Переменная для хранения начальной точки по x
y0_var = StringVar()  # Переменная для хранения начальной точки по y
learning_rate_var = StringVar()
max_iterations_var = StringVar()

# Создание и размещение виджетов
Label(root, text="Функция (например, '((y+1)*x)**2 + 4'):").grid(row=0, column=0)
Entry(root, textvariable=function_var).grid(row=0, column=1)

Label(root, text="Эпсилон:").grid(row=1, column=0)
Entry(root, textvariable=epsilon_var).grid(row=1, column=1)

Label(root, text="Начальная точка x:").grid(row=2, column=0)
Entry(root, textvariable=x0_var).grid(row=2, column=1)  # Текстовое поле для ввода начальной точки по x

Label(root, text="Начальная точка y:").grid(row=3, column=0)
Entry(root, textvariable=y0_var).grid(row=3, column=1)  # Текстовое поле для ввода начальной точки по y

Label(root, text="Скорость обучения:").grid(row=4, column=0)
Entry(root, textvariable=learning_rate_var).grid(row=4, column=1)

Label(root, text="Максимальное число итераций:").grid(row=5, column=0)
Entry(root, textvariable=max_iterations_var).grid(row=5, column=1)

Button(root, text="Запустить", command=run_gradient_descent).grid(row=6, column=0, columnspan=2)

# Запуск основного цикла
root.mainloop()