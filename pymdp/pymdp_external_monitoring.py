import numpy as np  
  
def infer_states_info_monitoring(  
    self, observation, distr_obs=False,     # original infer_states() arguments  
    num_iter=10, grad_descent=True, tau=0.25, last_timestep=False,   # _run_mmp_testing() arguments  
    monitoring=False      # include intermediate printouts  
):  
    """  
    Completely standalone version of infer_states() that returns intermediate variables from MMP optimization.  
    Only depends on numpy - all other functionality is implemented inline.  
      
    Returns  
    -------  
    qs : obj_array  
        Posterior beliefs over hidden states  
    xn : list  
        Intermediate beliefs during variational iterations (MMP only)  
    vn : list    
        Prediction errors at each iteration (MMP only)  
    """  
      
    import numpy as np  
    from itertools import chain  
      
    # ===== MATHEMATICAL HELPER FUNCTIONS =====  
      
    def softmax(dist):  
        """EXACT pymdp implementation"""  
        output = dist - dist.max(axis=0)  
        output = np.exp(output)  
        output = output / np.sum(output, axis=0)  
        return output  
      
    def spm_log_single(arr):  
        """EXACT pymdp implementation"""  
        EPS_VAL = 1e-16  
        return np.log(arr + EPS_VAL)  
      
    def spm_norm(A):  
        """Normalize columns of matrix A to sum to 1"""  
        return A / (np.sum(A, axis=0) + 1e-16)  
      
    def spm_dot(X, x, dims_to_omit=None):  
        """EXACT pymdp implementation with proper tensor contraction"""  
        # Handle object array case  
        if hasattr(x, '__len__') and not isinstance(x, str):  
            dims = list(range(X.ndim - len(x), len(x) + X.ndim - len(x)))  
        else:  
            dims = [1]  
            x = [x]  
          
        if dims_to_omit is not None:  
            arg_list = [X, list(range(X.ndim))] + list(chain(*([x[xdim_i],[dims[xdim_i]]] for xdim_i in range(len(x)) if xdim_i not in dims_to_omit))) + [dims_to_omit]  
        else:  
            arg_list = [X, list(range(X.ndim))] + list(chain(*([x[xdim_i],[dims[xdim_i]]] for xdim_i in range(len(x))))) + [[0]]  
          
        Y = np.einsum(*arg_list)  
          
        # Handle scalar case  
        if np.prod(Y.shape) <= 1.0:  
            Y = Y.item()  
            Y = np.array([Y]).astype("float64")  
          
        return Y  
      
    def norm_dist(A):  
        """EXACT pymdp implementation"""  
        if A.ndim == 3:  
            new_dist = np.zeros_like(A)  
            for c in range(A.shape[2]):  
                new_dist[:, :, c] = np.divide(A[:, :, c], A[:, :, c].sum(axis=0))  
            return new_dist  
        else:  
            return np.divide(A, A.sum(axis=0))  
      
    def obj_array(shape):  
        """Create object array"""  
        if isinstance(shape, int):  
            return np.empty(shape, dtype=object)  
        elif isinstance(shape, (list, tuple)):  
            return np.empty(shape, dtype=object)  
        else:  
            return np.empty(1, dtype=object)  
      
    def obj_array_uniform(num_states):  
        """FIXED: Create uniform belief object array matching pymdp exactly"""  
        arr = obj_array(len(num_states))  
        for i, shape in enumerate(num_states):  
            arr[i] = norm_dist(np.ones(shape))  
        return arr  
      
    def obj_array_zeros(num_states):  
        """Create zeros object array"""  
        arr = obj_array(len(num_states))  
        for i, shape in enumerate(num_states):  
            arr[i] = np.zeros(shape)  
        return arr  
      
    def get_model_dimensions(A, B):  
        """Extract model dimensions from A and B matrices"""  
        if A is not None:  
            num_modalities = len(A)  
            num_obs = [A[m].shape[0] for m in range(num_modalities)]  
        else:  
            num_modalities = 0  
            num_obs = []  
          
        num_factors = len(B)  
        num_states = [B[f].shape[0] for f in range(num_factors)]  
          
        return num_obs, num_states, num_modalities, num_factors  
      
    def process_observation_seq(obs_seq, num_modalities, num_obs):  
        """Process observation sequence into object array format"""  
        obs_processed = obj_array(len(obs_seq))  
        for t, obs_t in enumerate(obs_seq):  
            obs_t_array = obj_array(num_modalities)  
            for m in range(num_modalities):  
                if hasattr(obs_t[m], '__len__'):  
                    obs_t_array[m] = obs_t[m]  
                else:  
                    one_hot = np.zeros(num_obs[m])  
                    one_hot[obs_t[m]] = 1.0  
                    obs_t_array[m] = one_hot  
            obs_processed[t] = obs_t_array  
        return obs_processed  
      
    def get_joint_likelihood_seq(A, obs_seq, num_states, num_modalities):  
        """FIXED: Compute joint likelihood across all factors"""  
        T = len(obs_seq)  
        lh_seq = obj_array(T)  
          
        for t in range(T):  
            lh_joint = np.ones(num_states)  # Use full num_states shape  
            for m in range(num_modalities):  
                modality_lh = A[m].T @ obs_seq[t][m]  
                lh_joint = lh_joint * modality_lh  
            lh_seq[t] = lh_joint  
              
        return lh_seq  
      
    def to_obj_array(arr):  
        """Convert to object array if needed"""  
        if not hasattr(arr, '__len__'):  
            return np.array([arr])  
        return arr  
      
    def get_expected_states(qs, B, actions):  
        """Compute expected states given actions"""  
        num_factors = len(qs)  
        expected_states = obj_array(num_factors)  
          
        for f in range(num_factors):  
            action_f = int(actions[f])  
            expected_states[f] = B[f][:, :, action_f].T @ qs[f]  
              
        return expected_states  
      
    def sample(probabilities):  
        """Sample from categorical distribution"""  
        return np.random.choice(len(probabilities), p=probabilities)  
      
    # ===== MAIN INFERENCE LOGIC =====  
      
    # Handle VANILLA inference  
    if self.inference_algo == "VANILLA":  
        if self.action is not None:  
            empirical_prior = get_expected_states(  
                self.qs, self.B, self.action.reshape(1, -1)  
            )  
        else:  
            empirical_prior = self.D  
              
        # Simple Bayesian update for VANILLA mode  
        qs = obj_array(len(self.num_states))  
        for f in range(len(self.num_states)):  
            if hasattr(empirical_prior, '__len__'):  
                prior_f = empirical_prior[f]  
            else:  
                prior_f = empirical_prior  
                  
            # Compute likelihood  
            likelihood = np.ones(self.num_states[f])  
            for m in range(len(self.A)):  
                obs_idx = observation[m] if not distr_obs else observation[m]  
                if hasattr(obs_idx, '__len__'):  
                    likelihood *= self.A[m][:, obs_idx.argmax() if hasattr(obs_idx, 'argmax') else obs_idx[0]]  
                else:  
                    likelihood *= self.A[m][:, obs_idx]  
              
            # Posterior = likelihood * prior, normalized  
            qs[f] = softmax(np.log(likelihood + 1e-16) + np.log(prior_f + 1e-16))  
          
        xn = None  
        vn = None  
          
    elif self.inference_algo == "MMP":  
        # Update observation history  
        self.prev_obs.append(observation)  
        if len(self.prev_obs) > self.inference_horizon:  
            latest_obs = self.prev_obs[-self.inference_horizon:]  
            latest_actions = self.prev_actions[-(self.inference_horizon-1):]  
        else:  
            latest_obs = self.prev_obs  
            latest_actions = self.prev_actions  
  
        # Initialize prev_actions properly (FIXED)  
        prev_actions = latest_actions  
        if prev_actions is not None:  
            prev_actions = np.stack(prev_actions, 0)  
          
        # Get model dimensions  
        num_obs, num_states, num_modalities, num_factors = get_model_dimensions(self.A, self.B)  
  
        # Process observations  
        prev_obs = process_observation_seq(latest_obs, num_modalities, num_obs)  
          
        # Get joint likelihood  
        lh_seq = get_joint_likelihood_seq(self.A, prev_obs, num_states, num_modalities)  
          
        # Initialize results storage  
        qs_seq_pi = obj_array(len(self.policies))  
        xn_seq_pi = obj_array(len(self.policies))  
        vn_seq_pi = obj_array(len(self.policies))  
        F = np.zeros(len(self.policies))  
          
        # Process each policy  
        for p_idx, policy in enumerate(self.policies):  
            # Window dimensions  
            past_len = len(lh_seq)  
            future_len = policy.shape[0]  
              
            if last_timestep:  
                infer_len = past_len + future_len - 1  
            else:  
                infer_len = past_len + future_len  
              
            future_cutoff = past_len + future_len - 2  
              
            # Initialize beliefs for this policy  
            qs_seq = obj_array(infer_len)  
            for t in range(infer_len):  
                qs_seq[t] = obj_array_uniform(num_states)  
              
            # Last message  
            qs_T = obj_array_zeros(num_states)  
              
            # FIXED: Initialize prior correctly using obj_array_uniform  
            prior = obj_array_uniform(num_states)  
              
            # Transposed transition matrices  
            trans_B = obj_array(num_factors)  
            for f in range(num_factors):  
                trans_B[f] = spm_norm(np.swapaxes(self.B[f], 0, 1))  
              
            # Stack policy for easier indexing  
            policy_stacked = np.vstack([np.zeros((1, num_factors)), policy])  
              
            # Initialize intermediate variables  
            xn = []  
            vn = []  
              
            # Variational iterations  
            for itr in range(num_iter):  
                xn_itr_all_factors = obj_array(num_factors)  
                vn_itr_all_factors = obj_array(num_factors)  
                  
                for f in range(num_factors):  
                    xn_itr_all_factors[f] = np.zeros((num_states[f], infer_len))  
                    vn_itr_all_factors[f] = np.zeros((num_states[f], infer_len))  
                  
                # Forward-backward pass  
                for t in range(infer_len):  
                    for f in range(num_factors):  
                        # Likelihood term  
                        if t < past_len:  
                            lnA = spm_log_single(spm_dot(lh_seq[t], qs_seq[t], [f]))  
                        else:  
                            lnA = np.zeros(num_states[f])  
                          
                        # Past message  
                        if t == 0:  
                            lnB_past = spm_log_single(prior[f])  # This now works correctly  
                        else:  
                            past_msg = self.B[f][:, :, int(policy_stacked[t, f])].dot(qs_seq[t - 1][f])  
                            lnB_past = spm_log_single(past_msg)  
                          
                        # Future message  
                        if t >= future_cutoff:  
                            lnB_future = spm_log_single(qs_T[f])  
                        else:  
                            future_msg = trans_B[f][:, :, int(policy_stacked[t, f])].dot(qs_seq[t + 1][f])  
                            lnB_future = spm_log_single(future_msg)  
                          
                        # Belief update  
                        if grad_descent:  
                            sx = qs_seq[t][f]  
                            lnqs = spm_log_single(sx)  
                            coeff = 1 if (t >= future_cutoff) else 2  
                            err = (coeff * lnA + lnB_past + lnB_future) - coeff * lnqs  
                            vn_tmp = err - err.mean()  
                            lnqs = lnqs + tau * vn_tmp  
                            qs_seq[t][f] = softmax(lnqs)  
                              
                            # Free energy accumulation  
                            if (t == 0) or (t == (infer_len - 1)):  
                                F[p_idx] += sx.dot(0.5 * err)  
                            else:  
                                F[p_idx] += sx.dot(0.5 * (err - (num_factors - 1) * lnA / num_factors))  
                              
                            # Store intermediate values  
                            xn_itr_all_factors[f][:, t] = np.copy(qs_seq[t][f])  
                            vn_itr_all_factors[f][:, t] = np.copy(vn_tmp)  
                        else:  
                            qs_seq[t][f] = softmax(lnA + lnB_past + lnB_future)  
                  
                # Store iteration results  
                xn.append(xn_itr_all_factors)  
                vn.append(vn_itr_all_factors)  
              
            # Store policy results  
            qs_seq_pi[p_idx] = qs_seq  
            xn_seq_pi[p_idx] = xn  
            vn_seq_pi[p_idx] = vn  
          
        # Set final beliefs and free energy  
        qs = qs_seq_pi  
        self.F = F  
          
        # Flatten xn and vn for return format  
        xn_all = []  
        vn_all = []  
        for p_idx in range(len(self.policies)):  
            xn_all.extend(xn_seq_pi[p_idx])  
            vn_all.extend(vn_seq_pi[p_idx])  
          
        xn = xn_all  
        vn = vn_all  
  
    # Update agent state  
    if hasattr(self, "qs_hist"):  
        self.qs_hist.append(qs)  
  
    self.qs = qs  
  
    return qs, xn, vn
