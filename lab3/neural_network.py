# Лабораторная работа №5 по дисциплине МРЗвИС
# Вариант 10: Реализовать модель сети Джордана-Элмана с экспоненциально-линейной функцией активации (ELU).
# Выполнил студенты группы 221701 БГУИР Абушкевич Алексей Александрович и Юркевич Марианна Сергеевна
# Файл с реализацией сети Джордана-Элмана
# Дата 28.11.2025

import numpy as np
from activation_func import activate_elu, activate_elu_derivative

class JordanElmanNetwork:
    def __init__(self, seq, input_size, hidden_size, context_size, 
                 effector_size, alpha, max_errors, max_iters, predict_len,
                 reset_context, effector_activation_type='linear', hidden_alpha=1.0, verbose=True):
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.effector_size = effector_size
        
        self.alpha = alpha
        self.max_errors = max_errors
        self.max_iters = max_iters
        self.predict_len = predict_len
        self.reset_context = reset_context
        
        self.effector_act_type = effector_activation_type
        self.effector_act_alpha = hidden_alpha
        self.hidden_alpha = hidden_alpha
        self.verbose = verbose

        self.sequence = np.array(seq)
        self.expected = self.sequence[input_size:]

        assert 1 <= context_size <= hidden_size, "Context size must be <= Hidden size"
        assert effector_size >= 1, "Effector size must be >= 1"
        
        self.W_in = np.random.uniform(-0.1, 0.1, (input_size, hidden_size))
        self.W_jordan = np.random.uniform(-0.1, 0.1, (hidden_size, 1))
        self.W_context = np.random.uniform(-0.1, 0.1, (context_size, hidden_size))
        self.W_out = np.random.uniform(-0.1, 0.1, (effector_size, hidden_size))
        
        self.context = np.zeros(context_size)

    def _effector_activation(self, x):
        if self.effector_act_type == 'linear':
            return x
        return activate_elu(x, self.effector_act_alpha)

    def _d_effector_activation(self, x):
        if self.effector_act_type == 'linear':
            return 1.0
        return activate_elu_derivative(x, self.effector_act_alpha)

    def train(self):
        iteration = 0
        error = self.max_errors + 1.0

        error_history = [] 

        while iteration < self.max_iters and error > self.max_errors:
            dW = np.zeros_like(self.W_in)
            dW_ = np.zeros_like(self.W_jordan)
            dW_C = np.zeros_like(self.W_context)
            dW_O = np.zeros_like(self.W_out)

            inputs_hist = []
            hidden_states = []
            h_inputs = [] 
            outputs = []

            if self.reset_context:
                self.context.fill(0)
            
            prev_outputs = np.zeros(self.effector_size)
            total_sq_error = 0

            for i in range(len(self.expected)):
                inp = self.sequence[i : i + self.input_size]
                inputs_hist.append(inp)

                h_in = (inp @ self.W_in) + (self.context @ self.W_context) + (prev_outputs @ self.W_out)
                h_inputs.append(h_in)

                curr_hidden = activate_elu(h_in, self.hidden_alpha)
                hidden_states.append(curr_hidden)

                out_val = (curr_hidden @ self.W_jordan).item()
                outputs.append(out_val)

                self.context = curr_hidden[:self.context_size]

                prev_outputs = np.roll(prev_outputs, 1)
                prev_outputs[0] = self._effector_activation(out_val)

                diff = out_val - self.expected[i]
                total_sq_error += diff * diff

            d_hidden_next = np.zeros(self.hidden_size)

            for i in range(len(self.expected) - 1, -1, -1):
                diff = outputs[i] - self.expected[i]
                
                dW_ += diff * hidden_states[i].reshape(-1, 1)

                d_output_hidden = diff * self.W_jordan.T.flatten()
                
                total_hidden_error = d_output_hidden + d_hidden_next
                
                d_act = activate_elu_derivative(h_inputs[i], self.hidden_alpha)
                d_h_input = total_hidden_error * d_act 

                dW += np.outer(inputs_hist[i], d_h_input)

                if i == 0:
                    prev_context = np.zeros(self.context_size)
                else:
                    prev_context = hidden_states[i-1][:self.context_size]
                
                dW_C += np.outer(prev_context, d_h_input)

                prev_outputs_for_grad = np.zeros(self.effector_size)
                if i > 0:
                    for e in range(self.effector_size):
                        if (i - 1 - e) >= 0:
                            prev_outputs_for_grad[e] = self._effector_activation(outputs[i - 1 - e])
                
                dW_O += np.outer(prev_outputs_for_grad, d_h_input)

                d_hidden_next.fill(0)

                grad_wrt_context = d_h_input @ self.W_context.T
                d_hidden_next[:self.context_size] = grad_wrt_context

                if i > 0:
                    grad_wrt_prev_outputs = d_h_input @ self.W_out.T
                    grad_through_act = grad_wrt_prev_outputs[0] * self._d_effector_activation(outputs[i-1])
                    
                    d_hidden_next += grad_through_act * self.W_jordan.T.flatten()

            self.W_in   -= self.alpha * dW
            self.W_jordan  -= self.alpha * dW_
            self.W_context -= self.alpha * dW_C
            self.W_out -= self.alpha * dW_O

            error = total_sq_error / len(self.expected)

            error_history.append(error)
            iteration += 1

            if self.verbose and (iteration % 1000 == 0 or iteration == 1):
                print(f"Итерация {iteration}, Ошибка: {error:.10f}")

        print(f"Обучение прошло за {iteration} итерации, ошибка = {error:.10f}")
        return iteration, error_history

    def predict(self):
        res = []
        self.context.fill(0)
        prev_outputs = np.zeros(self.effector_size)
        
        for i in range(len(self.expected)):
            current_input = self.sequence[i : i + self.input_size]
            
            h_in = (current_input @ self.W_in) + (self.context @ self.W_context) + (prev_outputs @ self.W_out)
            
            curr_hidden = activate_elu(h_in, self.hidden_alpha)
            
            out_val = (curr_hidden @ self.W_jordan).item()
            
            self.context = curr_hidden[:self.context_size]
            prev_outputs = np.roll(prev_outputs, 1)
            prev_outputs[0] = self._effector_activation(out_val)

        current_input = self.sequence[-self.input_size:].copy()

        for _ in range(self.predict_len):
            h_in = (current_input @ self.W_in) + (self.context @ self.W_context) + (prev_outputs @ self.W_out)
            
            curr_hidden = activate_elu(h_in, self.hidden_alpha)
            
            out_val = (curr_hidden @ self.W_jordan).item()
            res.append(out_val)

            current_input = np.roll(current_input, -1)
            current_input[-1] = out_val

            self.context = curr_hidden[:self.context_size]
            prev_outputs = np.roll(prev_outputs, 1)
            prev_outputs[0] = self._effector_activation(out_val)
            
        return res