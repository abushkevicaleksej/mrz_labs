import numpy as np
import argparse
import time
import math
import sys

# ==========================================
# PART 1: Sequences (без изменений)
# ==========================================

# class Sequences:
#     SEQ_SIZE = 15

#     @staticmethod
#     def fibonacci(n=SEQ_SIZE):
#         fib = np.zeros(n)
#         if n > 0: fib[0] = 0
#         if n > 1: fib[1] = 1
#         for i in range(2, n):
#             fib[i] = fib[i-1] + fib[i-2]
#         return fib

#     @staticmethod
#     def geometric(n=SEQ_SIZE):
#         return np.array([1.0 / (2.0**(i + 1)) for i in range(n)])

#     @staticmethod
#     def periodic1(n=SEQ_SIZE):
#         seq = np.zeros(n)
#         pattern = [0, -1, 0, 1]
#         for i in range(n):
#             seq[i] = pattern[i % 4]
#         return seq

#     @staticmethod
#     def periodic2(n=SEQ_SIZE):
#         seq = np.zeros(n)
#         for i in range(n):
#             seq[i] = 1.0 if (i % 3 == 1) else 0.0
#         return seq

#     @staticmethod
#     def reciprocal(n=SEQ_SIZE):
#         return np.array([1.0 / (i + 2) for i in range(n)])

#     @staticmethod
#     def natural(n=SEQ_SIZE):
#         return np.arange(1, n + 1, dtype=float)

#     @staticmethod
#     def squares(n=SEQ_SIZE):
#         return np.array([(i + 1)**2 for i in range(n)], dtype=float)

#     @staticmethod
#     def fibonacci_like(n=SEQ_SIZE):
#         seq = np.zeros(n)
#         if n > 0: seq[0] = 1
#         if n > 1: seq[1] = 2
#         for i in range(2, n):
#             seq[i] = seq[i-1] + seq[i-2]
#         return seq

#     @staticmethod
#     def factorial(n=SEQ_SIZE):
#         seq = np.zeros(n)
#         if n > 0: seq[0] = 1
#         for i in range(1, n):
#             seq[i] = seq[i-1] * (i + 1)
#         return seq

#     @classmethod
#     def get_all(cls):
#         return {
#             "fib": cls.fibonacci(),
#             "geom": cls.geometric(),
#             "recip": cls.reciprocal(),
#             "per1": cls.periodic1(),
#             "per2": cls.periodic2(),
#             "nat": cls.natural(),
#             "sqr": cls.squares(),
#             "fib-like": cls.fibonacci_like(),
#             "fact": cls.factorial()
#         }

# ==========================================
# PART 2: Elman-Jordan RNN with ELU
# ==========================================

# def elu(x, alpha=1.0):
#     # Vectorized version of: val if val >= 0 else alpha * (np.exp(val) - 1)
#     return np.where(x >= 0, x, alpha * (np.exp(x) - 1))

# def d_elu(x, alpha=1.0):
#     # Vectorized version of: 1 if val > 0 else alpha * np.exp(val)
#     return np.where(x > 0, 1.0, alpha * np.exp(x))

# class ElmanJordanRNN:
#     def __init__(self, seq, input_size, hidden_size, context_size, 
#                  effector_size, alpha, max_errors, max_iters, predict_len,
#                  reset_context, effector_activation_type='linear', hidden_alpha=1.0, verbose=True):
        
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.context_size = context_size
#         self.effector_size = effector_size
        
#         self.alpha = alpha
#         self.max_errors = max_errors
#         self.max_iters = max_iters
#         self.predict_len = predict_len
#         self.reset_context = reset_context
        
#         self.effector_act_type = effector_activation_type
#         self.effector_act_alpha = hidden_alpha # Use same alpha for consistency if needed
#         self.hidden_alpha = hidden_alpha
#         self.verbose = verbose

#         self.sequence = np.array(seq)
#         self.expected = self.sequence[input_size:]
        
#         assert 1 <= context_size <= hidden_size, "Context size must be <= Hidden size"
#         assert effector_size >= 1, "Effector size must be >= 1"

#         # Initialize Weights [-1, 1]
#         self.W = np.random.uniform(-1, 1, (input_size, hidden_size))      
#         self.W_ = np.random.uniform(-1, 1, (hidden_size, 1))              
#         self.W_C = np.random.uniform(-1, 1, (context_size, hidden_size))  
#         self.W_O = np.random.uniform(-1, 1, (effector_size, hidden_size)) 

#         self.context = np.zeros(context_size)

#     def _effector_activation(self, x):
#         if self.effector_act_type == 'linear':
#             return x
#         # Если не linear, используем ELU (раньше было LeakyReLU)
#         return elu(x, self.effector_act_alpha)

#     def _d_effector_activation(self, x):
#         if self.effector_act_type == 'linear':
#             return 1.0
#         return d_elu(x, self.effector_act_alpha)

#     def train(self):
#         iteration = 0
#         error = self.max_errors + 1.0

#         while iteration < self.max_iters and error > self.max_errors:
#             # Gradients
#             dW = np.zeros_like(self.W)
#             dW_ = np.zeros_like(self.W_)
#             dW_C = np.zeros_like(self.W_C)
#             dW_O = np.zeros_like(self.W_O)

#             inputs_hist = []
#             hidden_states = []
#             h_inputs = [] 
#             outputs = []

#             if self.reset_context:
#                 self.context.fill(0)
            
#             prev_outputs = np.zeros(self.effector_size)
#             total_sq_error = 0

#             # --- Forward Pass ---
#             for i in range(len(self.expected)):
#                 inp = self.sequence[i : i + self.input_size]
#                 inputs_hist.append(inp)

#                 h_in = (inp @ self.W) + (self.context @ self.W_C) + (prev_outputs @ self.W_O)
#                 h_inputs.append(h_in)

#                 # USE ELU HERE
#                 curr_hidden = elu(h_in, self.hidden_alpha)
#                 hidden_states.append(curr_hidden)

#                 out_val = (curr_hidden @ self.W_).item()
#                 outputs.append(out_val)

#                 self.context = curr_hidden[:self.context_size]

#                 prev_outputs = np.roll(prev_outputs, 1)
#                 prev_outputs[0] = self._effector_activation(out_val)

#                 diff = out_val - self.expected[i]
#                 total_sq_error += diff * diff

#             # --- Backward Pass (BPTT) ---
#             d_hidden_next = np.zeros(self.hidden_size)

#             for i in range(len(self.expected) - 1, -1, -1):
#                 diff = outputs[i] - self.expected[i]
                
#                 dW_ += diff * hidden_states[i].reshape(-1, 1)

#                 # FIX: Flatten to avoid (1, H) vs (H,) shape mismatch
#                 d_output_hidden = diff * self.W_.T.flatten()
                
#                 total_hidden_error = d_output_hidden + d_hidden_next
                
#                 # USE d_ELU HERE
#                 d_act = d_elu(h_inputs[i], self.hidden_alpha)
#                 d_h_input = total_hidden_error * d_act 

#                 dW += np.outer(inputs_hist[i], d_h_input)

#                 if i == 0:
#                     prev_context = np.zeros(self.context_size)
#                 else:
#                     prev_context = hidden_states[i-1][:self.context_size]
                
#                 dW_C += np.outer(prev_context, d_h_input)

#                 prev_outputs_for_grad = np.zeros(self.effector_size)
#                 if i > 0:
#                     for e in range(self.effector_size):
#                         if (i - 1 - e) >= 0:
#                             prev_outputs_for_grad[e] = self._effector_activation(outputs[i - 1 - e])
                
#                 dW_O += np.outer(prev_outputs_for_grad, d_h_input)

#                 d_hidden_next.fill(0)

#                 grad_wrt_context = d_h_input @ self.W_C.T
#                 d_hidden_next[:self.context_size] = grad_wrt_context

#                 if i > 0:
#                     grad_wrt_prev_outputs = d_h_input @ self.W_O.T
#                     grad_through_act = grad_wrt_prev_outputs[0] * self._d_effector_activation(outputs[i-1])
                    
#                     # FIX: Flatten here as well for safety, though d_output_hidden fix above handles most cases
#                     d_hidden_next += grad_through_act * self.W_.T.flatten()

#             self.W   -= self.alpha * dW
#             self.W_  -= self.alpha * dW_
#             self.W_C -= self.alpha * dW_C
#             self.W_O -= self.alpha * dW_O

#             error = total_sq_error / len(self.expected)
#             iteration += 1

#             if self.verbose and (iteration % 1000 == 0 or iteration == 1):
#                 print(f"Iteration {iteration}, Error: {error:.10f}")

#         print(f"Training finished after {iteration} iterations, final error = {error:.10f}")
#         return iteration

#     def predict(self):
#         res = []
#         self.context.fill(0)
#         prev_outputs = np.zeros(self.effector_size)
        
#         for i in range(len(self.expected)):
#             current_input = self.sequence[i : i + self.input_size]
            
#             h_in = (current_input @ self.W) + (self.context @ self.W_C) + (prev_outputs @ self.W_O)
            
#             # USE ELU HERE
#             curr_hidden = elu(h_in, self.hidden_alpha)
            
#             out_val = (curr_hidden @ self.W_).item()
            
#             self.context = curr_hidden[:self.context_size]
#             prev_outputs = np.roll(prev_outputs, 1)
#             prev_outputs[0] = self._effector_activation(out_val)

#         current_input = self.sequence[-self.input_size:].copy()

#         for _ in range(self.predict_len):
#             h_in = (current_input @ self.W) + (self.context @ self.W_C) + (prev_outputs @ self.W_O)
            
#             # USE ELU HERE
#             curr_hidden = elu(h_in, self.hidden_alpha)
            
#             out_val = (curr_hidden @ self.W_).item()
#             res.append(out_val)

#             current_input = np.roll(current_input, -1)
#             current_input[-1] = out_val

#             self.context = curr_hidden[:self.context_size]
#             prev_outputs = np.roll(prev_outputs, 1)
#             prev_outputs[0] = self._effector_activation(out_val)
            
#         return res

# # ==========================================
# # PART 3: Main / Utilities
# # ==========================================

# def zscore_normalize(seq):
#     arr = np.array(seq)
#     mean = np.mean(arr)
#     std = np.std(arr)
#     if std == 0: std = 1.0
#     norm = (arr - mean) / std
#     return norm, mean, std

# def zscore_denormalize(norm, mean, std):
#     return np.array(norm) * std + mean

def main():
    parser = argparse.ArgumentParser(description="Elman-Jordan RNN for Time Series Prediction (ELU Activation)")
    
    parser.add_argument("--seq", type=str, default="fact", help="Sequence to train on", 
                        choices=Sequences.get_all().keys())
    parser.add_argument("--seq-len", type=int, default=10, help="Length of the training sequence")
    parser.add_argument("--win-len", type=int, default=2, help="Length of the window")
    parser.add_argument("--con-len", type=int, default=4, help="Number of context neurons")
    parser.add_argument("--eff-len", type=int, default=2, help="Number of effector neurons")
    parser.add_argument("--hidden", type=int, default=4, help="Number of hidden neurons")
    parser.add_argument("--alpha", type=float, default=0.0001, help="Learning rate")
    
    # CHANGED DEFAULT TO 1.0 FOR ELU
    parser.add_argument("--alpha-hidden", type=float, default=1.0, help="ELU alpha for hidden (Default: 1.0)")
    
    parser.add_argument("--iters", type=int, default=500000000, help="Maximum training iterations")
    parser.add_argument("--no-reset-ctx", action="store_true", help="Do NOT reset context each epoch")
    parser.add_argument("--max-error", type=float, default=1e-4, help="Maximum training error")
    parser.add_argument("--log", action="store_true", help="Use log transform")
    parser.add_argument("--zscore", action="store_true", help="Enable Z-score normalization")
    parser.add_argument("--graphs", action="store_true", help="Run experiments")

    args = parser.parse_args()

    if args.graphs:
        print("Graph generation logic skipped.")

    sequences_map = Sequences.get_all()
    full_sequence = sequences_map[args.seq]
    
    input_raw = full_sequence[:args.seq_len]
    
    processed_input = []
    input_mean = 0.0
    input_std = 1.0

    if args.log:
        print("INFO: Using LOG TRANSFORM preprocessing.")
        processed_input = np.log(input_raw + 1.0)
    elif args.zscore:
        print("INFO: Using Z-SCORE NORMALIZATION preprocessing.")
        processed_input, input_mean, input_std = zscore_normalize(input_raw)
    else:
        print("INFO: Using NO normalization.")
        processed_input = np.array(input_raw)

    rnn = ElmanJordanRNN(
        seq=processed_input,
        input_size=args.win_len,
        hidden_size=args.hidden,
        context_size=args.con_len,
        effector_size=args.eff_len,

        alpha=args.alpha,
        max_errors=args.max_error,
        max_iters=args.iters,
        predict_len=3, 
        reset_context=not args.no_reset_ctx,
        effector_activation_type='linear',
        hidden_alpha=args.alpha_hidden # Now acts as ELU alpha
    )

    start_time = time.time()
    rnn.train()
    end_time = time.time()

    print(f"Train duration: {int(end_time - start_time)} seconds")

    predicted_processed = rnn.predict()
    
    predicted = []
    if args.log:
        predicted = np.exp(predicted_processed) - 1.0
    elif args.zscore:
        predicted = zscore_denormalize(predicted_processed, input_mean, input_std)
    else:
        predicted = predicted_processed

    print(f"Input size: {len(input_raw)}, predicted size: {len(predicted)}")
    
    predict_len_actual = len(predicted)
    val_start = args.seq_len
    validation_target = full_sequence[val_start : val_start + predict_len_actual]

    for i in range(min(len(predicted), len(validation_target))):
        expected = validation_target[i]
        got = predicted[i]
        diff = got - expected
        status = "(equal)" if abs(diff) < 1e-6 else ""
        print(f"{i}. {expected} -> {got} diff: {diff} {status}")

if __name__ == "__main__":
    main()