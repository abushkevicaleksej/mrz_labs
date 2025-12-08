from typing import List, Dict
import numpy as np
from fclayer import FCLayer
from activation_func import ELU, Linear

MAX_ERROR = 1e-6

class JordanElmanNetwork:
    
    def __init__(self, window_size: int, hidden_size: int, output_size: int, 
                 context_reset: bool = False, elu_alpha: float = 1.0):
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.context_reset = context_reset
        
        self.context_elman = np.zeros((hidden_size, 1))
        self.context_jordan = np.zeros((output_size, 1))
        
        input_total_size = window_size + hidden_size + output_size
        
        self.hidden_layer = FCLayer(input_total_size, hidden_size, ELU(alpha=elu_alpha))
        self.output_layer = FCLayer(hidden_size, output_size, Linear())
        
    def reset_context(self):
        self.context_elman = np.zeros((self.hidden_size, 1))
        self.context_jordan = np.zeros((self.output_size, 1))
    
    def forward_step(self, x: np.ndarray, ctx_elman: np.ndarray, ctx_jordan: np.ndarray):
        combined_input = np.vstack([
            x.reshape(-1, 1),
            ctx_elman,
            ctx_jordan
        ])
        
        hidden_out, hidden_z = self.hidden_layer.forward(combined_input)
        net_out, net_z = self.output_layer.forward(hidden_out)
        
        return combined_input, hidden_out, hidden_z, net_out, net_z

    def forward(self, x: np.ndarray) -> np.ndarray:
        _, h_out, _, out, _ = self.forward_step(x, self.context_elman, self.context_jordan)
        self.context_elman = h_out.copy()
        self.context_jordan = out.copy()
        return out.flatten()
    
    def train(self, sequence: List[float], epochs: int, learning_rate: float = 0.01) -> None:
        n = len(sequence)
        
        for epoch in range(epochs):
            if self.context_reset:
                self.reset_context()
                
            history = []
            curr_elman = np.zeros((self.hidden_size, 1))
            curr_jordan = np.zeros((self.output_size, 1))
            total_loss = 0
            
            for i in range(self.window_size, n - self.output_size + 1):
                input_window = np.array(sequence[i - self.window_size:i])
                target = np.array(sequence[i:i + self.output_size]).reshape(-1, 1)
                
                combined_inp, h_out, h_z, out, out_z = self.forward_step(input_window, curr_elman, curr_jordan)
                
                state = {
                    'input_window': combined_inp,
                    'hidden_out': h_out,
                    'hidden_z': h_z,
                    'output_out': out,
                    'output_z': out_z,
                    'target': target,
                }
                history.append(state)
                
                curr_elman = h_out
                curr_jordan = out
                
                err = out - target
                total_loss += 0.5 * np.sum(err ** 2)

            
            w_hidden_acc = np.zeros_like(self.hidden_layer.weights)
            w_output_acc = np.zeros_like(self.output_layer.weights)
            
            t_hidden_acc = np.zeros_like(self.hidden_layer.threshold)
            t_output_acc = np.zeros_like(self.output_layer.threshold)
            
            delta_from_future_elman = np.zeros((self.hidden_size, 1))
            delta_from_future_jordan = np.zeros((self.output_size, 1))
            
            for t in range(len(history) - 1, -1, -1):
                state = history[t]
                
                output_error = state['output_out'] - state['target']
                total_output_grad = output_error + delta_from_future_jordan
                
                dx_hidden_curr, dw_out, dt_out = self.output_layer.backward(
                    total_output_grad, 
                    state['hidden_out'], 
                    state['output_z']
                )
                
                w_output_acc += dw_out
                t_output_acc += dt_out
                
                total_hidden_grad = dx_hidden_curr + delta_from_future_elman
                
                dx_combined, dw_hidden, dt_hidden = self.hidden_layer.backward(
                    total_hidden_grad,
                    state['input_window'],
                    state['hidden_z']
                )
                
                w_hidden_acc += dw_hidden
                t_hidden_acc += dt_hidden
                
                idx_elman_start = self.window_size
                idx_jordan_start = self.window_size + self.hidden_size
                
                delta_from_future_elman = dx_combined[idx_elman_start : idx_jordan_start]
                delta_from_future_jordan = dx_combined[idx_jordan_start :]
            
            
            self.hidden_layer.update_weights(w_hidden_acc, t_hidden_acc, learning_rate)
            self.output_layer.update_weights(w_output_acc, t_output_acc, learning_rate)
            
            # self.loss_history.append(total_loss)

            if epoch % 1000 == 0:
                print(f"Эпоха {epoch:5d}, Суммарная ошибка: {total_loss:.8f}")

            if total_loss <= MAX_ERROR:
                print(f"\nОбучение остановлено на эпохе {epoch}, точность достигнута.")
                break
    
    def predict(self, initial_window: List[float], steps: int) -> List[float]:
        if len(initial_window) != self.window_size:
            raise ValueError(f"Начальное окно должно иметь размер {self.window_size}")
        self.reset_context()
        predictions = []
        current_window = list(initial_window)
        for step in range(steps):
            window_array = np.array(current_window[-self.window_size:])
            prediction = self.forward(window_array)
            predictions.extend(prediction)
            if self.output_size == 1:
                current_window.append(prediction[0])
            else:
                current_window.append(prediction[-1])
        return predictions
    
    def evaluate(self, sequence: List[float]) -> float:
        self.reset_context()
        n = len(sequence)
        total_loss = 0
        for i in range(self.window_size, n - self.output_size + 1):
            input_window = np.array(sequence[i - self.window_size:i])
            target = np.array(sequence[i:i + self.output_size])
            prediction = self.forward(input_window)
            error = prediction - target.flatten()
            loss = 0.5 * np.sum(error ** 2)
            total_loss += loss
        return total_loss