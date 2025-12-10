# Лабораторная работа №5 по дисциплине МРЗвИС
# Вариант 10: Реализовать модель сети Джордана-Элмана с экспоненциально-линейной функцией активации (ELU).
# Выполнил студенты группы 221701 БГУИР Абушкевич Алексей Александрович и Юркевич Марианна Сергеевна
# Файл с реализацией сети Джордана-Элмана
# Дата 28.11.2025

import numpy as np
from activation_func import ELU

class JordanElmanNetwork:
    def __init__(self, input_size: int, hidden_size: int, context_size: int, 
                 jordan_size: int, output_size: int, elu_alpha: float = 1.0):
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.jordan_size = jordan_size
        self.output_size = output_size
        
        self.activation = ELU(alpha=elu_alpha)
        
        # Инициализация весов (как в задании: uniform -0.01, 0.01)
        # W_in: Вход -> Скрытый
        self.W_in = np.random.uniform(-0.1, 0.1, (hidden_size, input_size))
        # W_context: Контекст (Элман) -> Скрытый
        self.W_context = np.random.uniform(-0.1, 0.1, (hidden_size, context_size))
        # W_jordan: Выходы (Джордан) -> Скрытый
        self.W_jordan = np.random.uniform(-0.1, 0.1, (hidden_size, jordan_size * output_size))
        # W_out: Скрытый -> Выход
        self.W_out = np.random.uniform(-0.1, 0.1, (output_size, hidden_size))
        
        # Состояния
        self.context = np.zeros((context_size, 1))
        # Эффектор (история выходов) для Джордана
        self.prev_outputs = np.zeros((jordan_size * output_size, 1))

    def reset_context(self):
        self.context = np.zeros((self.context_size, 1))
        self.prev_outputs = np.zeros((self.jordan_size * self.output_size, 1))

    def forward_step(self, x, context, prev_outputs):
        """Один шаг прямого распространения"""
        # x shape: (input_size, 1)
        # context shape: (context_size, 1)
        # prev_outputs shape: (jordan_size * output_size, 1)
        
        # Суммируем входы
        hidden_input = (np.dot(self.W_in, x) + 
                        np.dot(self.W_context, context) + 
                        np.dot(self.W_jordan, prev_outputs))
        
        # Активация скрытого слоя
        hidden_state = self.activation.activate(hidden_input)
        
        # Выходной слой (линейная комбинация скрытых)
        # В C++ примере выход = hidden * W_. Здесь добавим активацию, если нужно, 
        # но обычно для регрессии на выходе активация либо линейная, либо такая же.
        # В исходном Python коде был ELU на выходе. Оставим ELU для совместимости.
        out_input = np.dot(self.W_out, hidden_state)
        output = self.activation.activate(out_input) 
        
        return hidden_input, hidden_state, out_input, output

    def train(self, inputs: list, targets: list, epochs: int, learning_rate: float, max_error: float, reset_ctx: bool):
        n_samples = len(inputs)
        
        for epoch in range(epochs):
            if reset_ctx:
                self.reset_context()
            
            # Хранилища для BPTT (как в C++ inputs, hidden_states, h_inputs)
            store_inputs = []
            store_h_inputs = []     # Вход в скрытый слой (до активации)
            store_h_states = []     # Выход скрытого слоя (после активации)
            store_outputs = []      # Выходы сети
            store_out_inputs = []   # Вход в выходной слой (до активации)
            store_contexts = []     # Значения контекста на каждом шаге
            store_jordan = []       # Значения входов Джордана на каждом шаге
            
            total_error = 0.0
            
            # --- ПРЯМОЙ ПРОХОД (по всей эпохе) ---
            curr_context = self.context.copy()
            curr_prev_outs = self.prev_outputs.copy()
            
            for i in range(n_samples):
                x = np.array(inputs[i]).reshape(-1, 1)
                t = np.array(targets[i]).reshape(-1, 1)
                
                # Сохраняем то, что пришло на вход нейронам (для градиентов)
                store_inputs.append(x)
                store_contexts.append(curr_context)
                store_jordan.append(curr_prev_outs)
                
                h_in, h_out, o_in, out = self.forward_step(x, curr_context, curr_prev_outs)
                
                store_h_inputs.append(h_in)
                store_h_states.append(h_out)
                store_out_inputs.append(o_in)
                store_outputs.append(out)
                
                # Обновление контекста (Elman): берем часть скрытого слоя
                # В C++: hidden.head(context_size)
                curr_context = h_out[:self.context_size].copy()
                
                # Обновление памяти Джордана (сдвиг и добавление нового выхода)
                # prev_outputs сдвигается "вниз", новый выход добавляется в начало
                new_jordan = np.roll(curr_prev_outs, self.output_size)
                new_jordan[:self.output_size] = out # или t, если Teacher Forcing (здесь используем out)
                curr_prev_outs = new_jordan
                
                # Ошибка MSE на текущем шаге
                err = np.sum((out - t) ** 2)
                total_error += err

            mse = total_error / n_samples
            if epoch % 100 == 0:
                print(f"Эпоха {epoch}, MSE: {mse:.6f}")
            
            if mse <= max_error:
                print(f"Обучение завершено на эпохе {epoch}. MSE: {mse:.6f}")
                break

            # --- ОБРАТНЫЙ ПРОХОД (BPTT) ---
            # Инициализация градиентов
            dW_in = np.zeros_like(self.W_in)
            dW_ctx = np.zeros_like(self.W_context)
            dW_jor = np.zeros_like(self.W_jordan)
            dW_out = np.zeros_like(self.W_out)
            
            # Ошибка, приходящая из будущего (для скрытого слоя)
            d_hidden_next = np.zeros((self.hidden_size, 1))
            
            for i in reversed(range(n_samples)):
                t = np.array(targets[i]).reshape(-1, 1)
                out = store_outputs[i]
                
                # 1. Градиент выхода
                # dE/dOut = 2 * (Out - Target)
                diff = 2 * (out - t) 
                
                # Производная функции активации выхода
                d_act_out = self.activation.activate_derivative(store_out_inputs[i])
                delta_out = diff * d_act_out # (output_size, 1)
                
                # Градиент для W_out
                dW_out += np.dot(delta_out, store_h_states[i].T)
                
                # 2. Градиент скрытого слоя
                # Ошибка от выхода текущего шага + ошибка от следующего шага (по времени)
                error_from_out = np.dot(self.W_out.T, delta_out)
                
                # Полная ошибка на скрытом слое (текущая + из будущего через контекст)
                total_hidden_error = error_from_out + d_hidden_next
                
                # Производная скрытого слоя
                d_act_hidden = self.activation.activate_derivative(store_h_inputs[i])
                delta_hidden = total_hidden_error * d_act_hidden # (hidden_size, 1)
                
                # 3. Накопление градиентов весов
                dW_in += np.dot(delta_hidden, store_inputs[i].T)
                dW_ctx += np.dot(delta_hidden, store_contexts[i].T)
                dW_jor += np.dot(delta_hidden, store_jordan[i].T)
                
                # 4. Вычисление ошибки для передачи назад во времени (t-1)
                
                # Путь через контекст (Elman): H(t) -> Context(t) -> H(t-1)
                # Поскольку Context(t) это просто копия H(t-1) (первые context_size элементов)
                # Градиент: W_context.T * delta_hidden
                grad_wrt_context = np.dot(self.W_context.T, delta_hidden)
                
                # Подготавливаем d_hidden_next для следующей итерации (которая t-1)
                # Сначала берем вклад от контекста.
                # Т.к. контекст - это часть скрытого, то ошибка прибавляется к соответствующим нейронам
                d_hidden_next = np.zeros((self.hidden_size, 1))
                d_hidden_next[:self.context_size] += grad_wrt_context
                
                # Путь через Джордана: H(t) -> Jordan(t) -> Out(t-1) -> H(t-1)
                # Jordan(t) содержит Out(t-1). Значит delta_hidden влияет на Out(t-1).
                # grad_wrt_jordan = W_jordan.T * delta_hidden
                grad_wrt_jordan_input = np.dot(self.W_jordan.T, delta_hidden)
                
                # Jordan input[0] соответствует Out(t-1).
                # Если history > 1, то input[1] это Out(t-2) и т.д.
                # Нас интересует влияние на Out(t-1), так как он зависит от H(t-1).
                if i > 0:
                    # Вклад в ошибку выхода предыдущего шага
                    delta_out_prev = grad_wrt_jordan_input[:self.output_size]
                    
                    # Эта ошибка должна пройти через активацию выхода (t-1) и веса W_out
                    # dOut(t-1)/dNet(t-1)
                    d_act_out_prev = self.activation.activate_derivative(store_out_inputs[i-1])
                    
                    # dNet(t-1)/dH(t-1) = W_out
                    # Полный путь: delta_out_prev * f'(net_out_prev) * W_out
                    term = (delta_out_prev * d_act_out_prev) # element-wise
                    d_hidden_next += np.dot(self.W_out.T, term)

            # Обновление весов
            self.W_in -= learning_rate * dW_in
            self.W_context -= learning_rate * dW_ctx
            self.W_jordan -= learning_rate * dW_jor
            self.W_out -= learning_rate * dW_out

    def predict(self, initial_window: list, steps: int):
        self.reset_context()
        predictions = []
        
        # Сначала "прогреваем" сеть на начальном окне, чтобы заполнить контекст
        # В данном случае initial_window - это просто входные данные перед прогнозом
        curr_context = self.context.copy()
        curr_prev_outs = self.prev_outputs.copy()
        
        # Подготовка: прогон начального окна, чтобы сформировать контекст
        # Здесь мы предполагаем, что initial_window подается шаг за шагом
        # Если initial_window - это последовательность значений X
        
        # Для простоты, так как размер окна на входе сети = WIN_SIZE (из main),
        # то initial_window ожидается как [x_1, x_2, ..., x_win].
        # Но архитектура принимает 1 вектор input_size.
        # Если input_size == WIN_SIZE, то мы делаем один проход.
        
        x = np.array(initial_window).reshape(-1, 1)
        
        # Если мы хотим прогнозировать рекурсивно, нам нужно подавать выходы как новые входы
        # Для этого нужно знать логику формирования входа.
        # В этой лабе обычно input - это скользящее окно.
        
        # Логика прогноза:
        current_input_window = list(initial_window) # список длины input_size
        
        for _ in range(steps):
            x_in = np.array(current_input_window).reshape(-1, 1)
            
            _, h_out, _, out_val = self.forward_step(x_in, curr_context, curr_prev_outs)
            
            pred_scalar = out_val[0, 0]
            predictions.append(pred_scalar)
            
            # Обновляем состояния
            curr_context = h_out[:self.context_size].copy()
            
            new_jordan = np.roll(curr_prev_outs, self.output_size)
            new_jordan[:self.output_size] = out_val
            curr_prev_outs = new_jordan
            
            # Обновляем входное окно (сдвигаем и добавляем предсказанное значение)
            current_input_window.pop(0)
            current_input_window.append(pred_scalar)
            
        return predictions

    def evaluate(self, inputs: list, targets: list):
        self.reset_context()
        total_error = 0
        n = len(inputs)
        
        curr_context = self.context.copy()
        curr_prev_outs = self.prev_outputs.copy()
        
        for i in range(n):
            x = np.array(inputs[i]).reshape(-1, 1)
            t = np.array(targets[i]).reshape(-1, 1)
            
            _, h_out, _, out = self.forward_step(x, curr_context, curr_prev_outs)
            
            curr_context = h_out[:self.context_size].copy()
            new_jordan = np.roll(curr_prev_outs, self.output_size)
            new_jordan[:self.output_size] = out
            curr_prev_outs = new_jordan
            
            total_error += np.sum((out - t) ** 2)
            
        return total_error # Возвращаем сумму квадратов, как в C++ (или MSE)