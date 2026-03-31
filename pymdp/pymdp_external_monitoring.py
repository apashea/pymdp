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
        else:  
            return np.empty(len(shape), dtype=object)  
      
    def obj_array_zeros(shape_list):  
        """Create object array with zeros"""  
        arr = obj_array(len(shape_list))  
        for i, shape in enumerate(shape_list):  
            arr[i] = np.zeros(shape)  
        return arr  
      
    def obj_array_uniform(shape_list):  
        """Create object array with uniform distributions"""  
        arr = obj_array(len(shape_list))  
        for i, shape in enumerate(shape_list):  
            arr[i] = norm_dist(np.ones(shape))  
        return arr  
      
    def get_model_dimensions(A, B):  
        """Extract model dimensions"""  
        num_modalities = len(A)  
        num_factors = len(B)  
        num_obs = [A[m].shape[0] for m in range(num_modalities)]  
        num_states = [B[f].shape[0] for f in range(num_factors)]  
        return num_obs, num_states, num_modalities, num_factors  
      
    def process_observation_seq(obs_seq, num_modalities, num_obs):  
        """Process observation sequence"""  
        processed_seq = obj_array(len(obs_seq))  
        for t, obs_t in enumerate(obs_seq):  
            if not isinstance(obs_t, (list, tuple, np.ndarray)):  
                obs_t = [obs_t]  
            obs_array = obj_array(num_modalities)  
            for m in range(num_modalities):  
                if isinstance(obs_t[m], (int, np.integer)):  
                    obs_onehot = np.zeros(num_obs[m])  
                    obs_onehot[obs_t[m]] = 1.0  
                    obs_array[m] = obs_onehot  
                else:  
                    obs_array[m] = obs_t[m]  
            processed_seq[t] = obs_array  
        return processed_seq  
      
    def get_joint_likelihood_seq(A, obs_seq, qs_seq):  
        """Compute joint likelihood sequence"""  
        num_obs, num_states, num_modalities, num_factors = get_model_dimensions(A, None)  
        infer_len = len(obs_seq)  
          
        lh_seq = obj_array(infer_len)  
        for t in range(infer_len):  
            lh_t = obj_array(num_modalities)  
            for m in range(num_modalities):  
                # Compute likelihood for this modality  
                likelihood_m = A[m].T @ obs_seq[t][m]  
                lh_t[m] = likelihood_m  
            lh_seq[t] = lh_t  
          
        return lh_seq  
      
    # ===== MAIN INFERENCE LOGIC =====  
      
    if self.inference_algo == "VANILLA":  
        # VANILLA inference  
        if self.action is not None:  
            empirical_prior = self.B[0][:, :, int(self.action[0])].dot(self.qs[0])  
        else:  
            empirical_prior = self.D[0]  
          
        # Compute posterior using Bayes' rule  
        likelihood = self.A[0].T @ observation  
        if not distr_obs:  
            obs_onehot = np.zeros(len(self.A[0]))  
            obs_onehot[observation] = 1.0  
            likelihood = self.A[0].T @ obs_onehot  
          
        posterior = likelihood * empirical_prior  
        qs = obj_array(1)  
        qs[0] = norm_dist(posterior)  
          
        xn = None  
        vn = None  
          
    elif self.inference_algo == "MMP":  
        # MMP inference  
        self.prev_obs.append(observation)  
        if len(self.prev_obs) > self.inference_horizon:  
            latest_obs = self.prev_obs[-self.inference_horizon:]  
            latest_actions = self.prev_actions[-(self.inference_horizon-1):]  
        else:  
            latest_obs = self.prev_obs  
            latest_actions = self.prev_actions  
  
        # Initialize prev_actions properly  
        prev_actions = latest_actions  
        if prev_actions is not None:  
            prev_actions = np.stack(prev_actions, 0)  
          
        # Get model dimensions  
        num_obs, num_states, num_modalities, num_factors = get_model_dimensions(self.A, self.B)  
  
        # Process observations  
        prev_obs = process_observation_seq(latest_obs, num_modalities, num_obs)  
          
        # Initialize prior  
        prior = obj_array_uniform(num_states)  
          
        # Initialize storage for results  
        qs_seq_pi = obj_array(len(self.policies))  
        xn_seq_pi = obj_array(len(self.policies))  
        vn_seq_pi = obj_array(len(self.policies))  
        F = np.zeros(len(self.policies))  
          
        # Process each policy  
        for p_idx, policy in enumerate(self.policies):  
            # Combine prev_actions with current policy  
            if prev_actions is not None:  
                full_policy = np.vstack((prev_actions, policy))  
            else:  
                full_policy = policy  
              
            infer_len = len(prev_obs) + len(policy)  
              
            # Initialize belief sequence  
            qs_seq = obj_array(infer_len)  
            for t in range(infer_len):  
                qs_seq[t] = obj_array_uniform(num_states)  
              
            # Get joint likelihood sequence  
            lh_seq = get_joint_likelihood_seq(self.A, prev_obs, qs_seq)  
              
            # Variational iterations  
            xn = []  
            vn = []  
              
            for itr in range(num_iter):  
                # CRITICAL FIX: Reset F_policy for each iteration  
                F_policy = 0.0  
                  
                # Initialize storage for intermediate values  
                shape_list = [[num_states[f], infer_len] for f in range(num_factors)]  
                xn_itr_all_factors = obj_array_zeros(shape_list)  
                vn_itr_all_factors = obj_array_zeros(shape_list)  
                  
                for t in range(infer_len):  
                    for f in range(num_factors):  
                        # Likelihood term  
                        if t < len(prev_obs):  
                            lnA = spm_log_single(spm_dot(lh_seq[t], qs_seq[t], [f]))  
                        else:  
                            lnA = np.zeros(num_states[f])  
                          
                        # Past message  
                        if t == 0:  
                            lnB_past = spm_log_single(prior[f])  
                        else:  
                            past_msg = self.B[f][:, :, int(full_policy[t - 1, f])].dot(qs_seq[t - 1][f])  
                            lnB_past = spm_log_single(past_msg)  
                          
                        # Future message  
                        if t >= infer_len - 2:  
                            lnB_future = np.zeros(num_states[f])  
                        else:  
                            future_msg = self.B[f][:, :, int(full_policy[t, f])].dot(qs_seq[t + 1][f])  
                            lnB_future = spm_log_single(future_msg)  
                          
                        # Inference  
                        if grad_descent:  
                            sx = qs_seq[t][f]  
                            lnqs = spm_log_single(sx)  
                            coeff = 1 if (t >= infer_len - 2) else 2  
                            err = (coeff * lnA + lnB_past + lnB_future) - coeff * lnqs  
                            vn_tmp = err - err.mean()  
                            lnqs = lnqs + tau * vn_tmp  
                            qs_seq[t][f] = softmax(lnqs)  
                              
                            # CRITICAL FIX: Accumulate F_policy within iteration only  
                            if (t == 0) or (t == (infer_len - 1)):  
                                F_policy += sx.dot(0.5 * err)  
                            else:  
                                F_policy += sx.dot(0.5 * (err - (num_factors - 1) * lnA / num_factors))  
                              
                            # Store intermediate values  
                            xn_itr_all_factors[f][:, t] = np.copy(qs_seq[t][f])  
                            vn_itr_all_factors[f][:, t] = np.copy(vn_tmp)  
                        else:  
                            qs_seq[t][f] = softmax(lnA + lnB_past + lnB_future)  
                  
                # Store iteration results  
                xn.append(xn_itr_all_factors)  
                vn.append(vn_itr_all_factors)  
              
            # Store final F for this policy after all iterations  
            F[p_idx] = F_policy  
              
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
