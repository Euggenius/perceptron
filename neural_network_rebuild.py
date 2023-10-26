import numpy as np
import tkinter as tk
import os
from math import e
from PIL import Image
from datetime import datetime
from copy import deepcopy


def activation_func(x):
    return 1 / (1 + (e ** -x))


def deriv_activation_func(x):
    fx = activation_func(x)
    return fx * (1 - fx)


class Perceptron:
    def __init__(self, layers_parameters):
        self.lp = layers_parameters
        self.weights = []
        self.biases = []
        self.z_values = [[] for _ in range(len(self.lp))]
        self.a_values = [[] for _ in range(len(self.lp))]

        # создание случайных весов и смещений
        for i in range(1, len(self.lp)):
            self.weights.append(np.random.uniform(-1, 1, size=(self.lp[i - 1], self.lp[i])))
            self.biases.append(np.random.uniform(-1, 1, size=(self.lp[i])))

    # прямое распространение
    def forward_propagation(self, input_data):
        self.z_values[0] = input_data
        self.a_values[0] = input_data
        for i in range(len(self.lp) - 1):
            self.z_values[i + 1] = np.dot(self.a_values[i], self.weights[i]) + self.biases[i]
            self.a_values[i + 1] = np.vectorize(activation_func)(self.z_values[i + 1])
        return self.a_values[-1]

    # обратное распространение + обучение
    def back_propagation(self, output, correct_output, learn_rate=0.1):
        mse_loss = np.vectorize(np.square)(correct_output - output)
        deriv_loss = (output - correct_output) * 2
        
        for i in range(len(self.lp) - 2, -1, -1):
            # обновление смещений
            self.biases[i] -= learn_rate * np.vectorize(deriv_activation_func)(self.z_values[i + 1]) * deriv_loss
            # обновление весов
            self.weights[i] -= np.tile(learn_rate * deriv_loss * np.vectorize(deriv_activation_func)(self.z_values[i + 1]), (self.lp[i], 1)) * np.rot90(np.array([self.a_values[i]]), 3)
            # обновление производных нейронов от функции потерь
            if i > 0:
                deriv_loss = np.sum(self.weights[i] * np.tile(np.vectorize(deriv_activation_func)(self.z_values[i + 1]) * deriv_loss, (self.lp[i], 1)), axis=1)

        return sum(mse_loss)


mnist_classifier = Perceptron([784, 250, 100, 10])

images_train_list = os.listdir('/Users/evgenijbojko/Downloads/mnist_train')
images_test_list = os.listdir('/Users/evgenijbojko/Downloads/mnist_test')
learning_data, test_data = [], []

for image_name in images_train_list[:30000]:
    sample = Image.open(f'/Users/evgenijbojko/Downloads/mnist_train/{image_name}')
    pixels = np.array(sample.getdata()) / 255

    answer = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    answer[int(image_name[10])] = 1

    learning_data.append({'input': pixels, 'output': answer})
for image_name in images_test_list:
    sample = Image.open(f'/Users/evgenijbojko/Downloads/mnist_test/{image_name}')
    pixels = np.array(sample.getdata()) / 255

    test_data.append({'input': pixels, 'output': int(image_name[10])})

learning_start_time = datetime.now()

print('start learning')
print(datetime.now())
print()
print('-' * 50)

for epoch in range(3):
    average_mse_loss = 0
    for true_answer in learning_data:
        pred_answer = np.array(mnist_classifier.forward_propagation(true_answer['input']))
        average_mse_loss += mnist_classifier.back_propagation(pred_answer, true_answer['output'])

    if (epoch + 1) % 1 == 0:
        print(f'epoch {epoch + 1} finished')
        print(datetime.now())
        print()
        print(average_mse_loss / len(learning_data))
        print(datetime.now())
        print()
        print('-' * 50)

print('learning finished')
print(f'learning time: {datetime.now() - learning_start_time}')
print()

correct_answers = 0

for test_example in test_data:
    answer = list(mnist_classifier.forward_propagation(test_example['input']))
    if answer.index(max(answer)) == test_example['output']:
        correct_answers += 1

print(f'correct answers: {correct_answers / len(test_data) * 100}%')
print()
print('-' * 50)

# размеры холста и пикселей
canvas_width = 560
canvas_height = 560
pixel_size = 20
grid_size = canvas_width // pixel_size

# размер окна с цифрой
number_width = 20
number_height = 2

# создание холста
root = tk.Tk()
root.title("Рисовалка")
root.geometry(f"{canvas_width + number_width + 120}x{canvas_height}")

canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg='black')
canvas.pack(side=tk.LEFT)

# создание массива для хранения рисунка
pixels = [[0 for _ in range(grid_size)] for _ in range(grid_size)]

eraser = False


# функция для вычисления интенсивности пикселей в зависимости от расстояния
def compute_intensity(distance):
    max_intensity = 255  # максимальная интенсивность, ограничивающая яркость

    intensity = max_intensity * (0.25 ** distance)

    return int(intensity)


def draw(event):
    x = event.x // pixel_size
    y = event.y // pixel_size

    # рисуем пиксели вокруг курсора с учетом интенсивности
    for i in range(-1, 2):
        for j in range(-1, 2):
            if 0 <= x + i < grid_size and 0 <= y + j < grid_size:
                intensity = compute_intensity(abs(i) + abs(j))

                if not eraser:
                    pixels[y + j][x + i] = min(255, pixels[y + j][x + i] + intensity)
                else:
                    pixels[y + j][x + i] = max(0, pixels[y + j][x + i] - int(intensity * 1.25))

                color = '#{:02x}{:02x}{:02x}'.format(pixels[y + j][x + i], pixels[y + j][x + i], pixels[y + j][x + i])
                canvas.create_rectangle((x + i) * pixel_size, (y + j) * pixel_size, (x + i + 1) * pixel_size,
                                        (y + j + 1) * pixel_size, fill=color)

    # переводим картинку во входные данные
    image = []
    for row in pixels:
        for pixel in row:
            image.append(pixel / 255)

    # обновляем значения чисел в окнах
    if image == [0.0] * 784:
        for i in range(10):
            number_labels[i].config(text=str(0.0))
    else:
        for i, number_label_value in enumerate(mnist_classifier.forward_propagation(np.array(image))):
            if number_label_value >= 0.0001:
                number_labels[i].config(text=str(round(number_label_value, 4)))
            else:
                number_labels[i].config(text=str(0.0))

    update_button_colors()


canvas.bind("<B1-Motion>", draw)


def clear_canvas():
    canvas.delete('all')
    for i in range(grid_size):
        for j in range(grid_size):
            pixels[i][j] = 0

    for i, number_label_value in enumerate([0.0] * 10):
        number_labels[i].config(text=str(round(number_label_value, 5)))

    update_button_colors()


def toggle_eraser():
    global eraser
    eraser = not eraser
    eraser_button.config(text="Карандаш" if eraser else "Ластик")


eraser_button = tk.Button(root, text="Карандаш" if eraser else "Ластик", command=toggle_eraser)
eraser_button.pack(pady=10)
clear_button = tk.Button(root, text="Очистить поле", command=clear_canvas)
clear_button.pack(pady=5)

numbers_frame = tk.Frame(root)
numbers_frame.pack(side=tk.RIGHT, padx=10, pady=10)
number_labels = []

for number in [0.0] * 10:
    number_label = tk.Label(numbers_frame, text=str(number), width=number_width, height=number_height,
                            font=("Arial", 12), relief=tk.RAISED, borderwidth=2)
    number_label.pack(pady=5)
    number_labels.append(number_label)


def get_gradient_color(value):
    # вычисляем значения каналов RGB
    red = max(100, int(255 * (1 - value)))
    green = 255
    blue = max(100, int(255 * (1 - value)))

    # возвращаем цвет в формате RGB
    return '#{:02x}{:02x}{:02x}'.format(red, green, blue)


def update_button_colors():
    for label in number_labels:
        label.config(bg=get_gradient_color(float(label.cget("text"))))


update_button_colors()
root.mainloop()
