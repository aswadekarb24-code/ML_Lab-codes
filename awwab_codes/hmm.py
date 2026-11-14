import numpy as np

class SimpleHMM:
    def __init__(self, n_states, n_observations):
        self.n_states = n_states
        self.n_observations = n_observations
        # Random initialization of parameters (A: transitions, B: emissions, pi: priors)
        self.A = np.ones((n_states, n_states)) / n_states
        self.B = np.ones((n_states, n_observations)) / n_observations
        self.pi = np.ones(n_states) / n_states
        self.loss_history = [] # Dummy attribute for template compatibility

    # ==========================================
    # 1. FORWARD ALGORITHM (Evaluation Problem)
    # ==========================================
    def forward(self, observations):
        T = len(observations)
        alpha = np.zeros((T, self.n_states))
        
        # Initialization
        alpha[0] = self.pi * self.B[:, observations[0]]
        
        # Induction
        for t in range(1, T):
            for j in range(self.n_states):
                alpha[t, j] = np.sum(alpha[t-1] * self.A[:, j]) * self.B[j, observations[t]]
                
        return alpha, np.sum(alpha[-1]) # Probability of observation sequence P(O|lambda)

    # ==========================================
    # 2. BACKWARD ALGORITHM (Smoothing / Evaluation)
    # ==========================================
    def backward(self, observations):
        T = len(observations)
        beta = np.zeros((T, self.n_states))
        
        # Initialization
        beta[-1] = 1.0
        
        # Induction (moving backwards)
        for t in range(T - 2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = np.sum(self.A[i, :] * self.B[:, observations[t+1]] * beta[t+1])
                
        return beta

    # ==========================================
    # 3. VITERBI ALGORITHM (Decoding Problem)
    # ==========================================
    def viterbi(self, observations):
        T = len(observations)
        viterbi_table = np.zeros((T, self.n_states))
        backpointer = np.zeros((T, self.n_states), dtype=int)
        
        # Initialization
        viterbi_table[0] = self.pi * self.B[:, observations[0]]
        
        # Recursion
        for t in range(1, T):
            for j in range(self.n_states):
                probs = viterbi_table[t-1] * self.A[:, j] * self.B[j, observations[t]]
                backpointer[t, j] = np.argmax(probs)
                viterbi_table[t, j] = np.max(probs)
                
        # Termination & Backtracking
        best_last_state = np.argmax(viterbi_table[-1])
        best_path = [best_last_state]
        
        for t in range(T - 1, 0, -1):
            best_last_state = backpointer[t, best_last_state]
            best_path.insert(0, best_last_state)
            
        return best_path

    # ==========================================
    # 4. BAUM-WELCH ALGORITHM / EM (Learning)
    # ==========================================
    def fit(self, observations_list, epochs=10):
        """Trains HMM parameters given a list of observation sequences"""
        self.loss_history = []
        
        for epoch in range(epochs):
            total_log_prob = 0
            
            # Accumulators for parameter re-estimation
            xi_sum = np.zeros((self.n_states, self.n_states))
            gamma_sum = np.zeros((self.n_states, self.n_observations))
            pi_sum = np.zeros(self.n_states)
            total_sequences = len(observations_list)
            
            for obs in observations_list:
                T = len(obs)
                alpha, p_obs = self.forward(obs)
                beta = self.backward(obs)
                total_log_prob += np.log(p_obs + 1e-15)
                
                # Compute gamma (state probabilities) and xi (transition probabilities)
                gamma = (alpha * beta) / (np.sum(alpha * beta, axis=1, keepdims=True) + 1e-15)
                pi_sum += gamma[0]
                
                for t in range(T - 1):
                    denominator = np.sum(alpha[t] * np.dot(self.A, self.B[:, obs[t+1]] * beta[t+1])) + 1e-15
                    for i in range(self.n_states):
                        for j in range(self.n_states):
                            xi_sum[i, j] += (alpha[t, i] * self.A[i, j] * self.B[j, obs[t+1]] * beta[t+1, j]) / denominator
                            
                for i in range(self.n_states):
                    for k in range(self.n_observations):
                        mask = (np.array(obs) == k)
                        gamma_sum[i, k] += np.sum(gamma[mask, i])
                        
            # Maximization Step (Re-estimation)
            self.pi = pi_sum / total_sequences
            self.A = xi_sum / (np.sum(xi_sum, axis=1, keepdims=True) + 1e-15)
            self.B = gamma_sum / (np.sum(gamma_sum, axis=1, keepdims=True) + 1e-15)
            
            self.loss_history.append(-total_log_prob / total_sequences)
            
        return self

    def predict(self, X):
        """Wrapper to fit master template evaluation (returns Viterbi path prediction per sequence)"""
        # If X is 2D array of sequences
        if X.ndim == 2:
            return np.array([self.viterbi(seq) for seq in X])
        return np.array(self.viterbi(X))