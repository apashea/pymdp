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
      
    # ===== MATHEMATICAL HELPER FUNCTIONS =====  
      
    def softmax(dist):  
        """EXACT pymdp implementation<cite repo="apashea/pymdp" path="pymdp/maths.py" start="326-334" />"""  
        output = dist - dist.max(axis=0)  
        output = np.exp(output)  
        output = output / np.sum(output, axis=0)  
        return output  
      
    def spm_log_single(arr):  
        """EXACT pymdp implementation<cite repo="apashea/pymdp" path="pymdp/maths.py" start="263-267" />"""  
        EPS_VAL = 1e-16  
        return np.log(arr + EPS_VAL)  
      
    def spm_norm(A):  
        """Normalize columns of matrix A to sum to 1"""  
        return A / (np.sum(A, axis=0) + 1e-16)  
      
    def spm_dot(X, x, dims_to_omit=None):  
        """EXACT pymdp implementation simplified for standalone use<cite repo="apashea/pymdp" path="pymdp/maths.py" start="18-54" />"""  
        if dims_to_omit is None:  
            # Simple case: direct matrix multiplication  
            if hasattr(x, '__len__') and not isinstance(x, str):  
                # Object array case - element-wise multiplication  
                result = X.copy()  
                for i in range(len(x)):  
                    result = result * x[i]  
                return result  
            else:  
                return X @ x  
        else:  
            # Complex case with dims_to_omit - simplified version  
            result = X.copy()  
            for i in range(len(x)):  
                if i not in dims_to_omit:  
                    result = result * x[i]  
            return result  
      
    # ===== OBJECT ARRAY UTILITIES =====  
      
    def obj_array(shape):  
        """Create numpy object array"""  
        if isinstance(shape, int):  
            return np.empty(shape, dtype=object)  
        elif isinstance(shape, (list, tuple)):  
            return np.empty(shape, dtype=object)  
        else:  
            return np.empty(1, dtype=object)  
      
    def obj_array_uniform(num_states):  
        """Create uniform belief object array"""  
        out = obj_array(len(num_states))  
        for f, n_s in enumerate(num_states):  
            out[f] = np.ones(n_s) / n_s  
        return out  
      
    def obj_array_zeros(num_states):  
        """Create zero object array"""  
        out = obj_array(len(num_states))  
        for f, n_s in enumerate(num_states):  
            out[f] = np.zeros(n_s)  
        return out  
      
    def obj_array_zeros_list(shape_list):  
        """Create nested object array with zeros"""  
        out = obj_array(len(shape_list))  
        for i, shape in enumerate(shape_list):  
            out[i] = np.zeros(shape)  
        return out  
      
    def norm_dist(A):  
        """Normalize distribution to sum to 1"""  
        if isinstance(A, np.ndarray) and A.dtype == object:  
            out = obj_array(len(A))  
            for i in range(len(A)):  
                out[i] = A[i] / (np.sum(A[i]) + 1e-16)  
            return out  
        else:  
            return A / (np.sum(A) + 1e-16)  
      
    def norm_dist_obj_arr(A):  
        """Normalize object array of distributions"""  
        out = obj_array(len(A))  
        for i in range(len(A)):  
            out[i] = A[i] / (np.sum(A[i]) + 1e-16)  
        return out  
      
    # ===== MODEL DIMENSION UTILITIES =====  
      
    def get_model_dimensions(A, B):  
        """Extract model dimensions from A and B matrices"""  
        if A is not None:  
            num_modalities = len(A)  
            num_obs = [A[m].shape[0] for m in range(num_modalities)]  
        else:  
            num_modalities = 0  
            num_obs = []  
          
        if B is not None:  
            num_factors = len(B)  
            num_states = [B[f].shape[0] for f in range(num_factors)]  
        else:  
            num_factors = 0  
            num_states = []  
          
        return num_obs, num_states, num_modalities, num_factors  
      
    def process_observation_seq(obs_seq, num_modalities, num_obs):  
        """Process observation sequence into distributional format"""  
        processed_seq = obj_array(len(obs_seq))  
        for t, obs_t in enumerate(obs_seq):  
            obs_array = obj_array(num_modalities)  
            for m in range(num_modalities):  
                if isinstance(obs_t[m], (int, np.integer)):  
                    # Convert integer to one-hot  
                    obs_m = np.zeros(num_obs[m])  
                    obs_m[obs_t[m]] = 1.0  
                    obs_array[m] = obs_m  
                else:  
                    obs_array[m] = obs_t[m]  
            processed_seq[t] = obs_array  
        return processed_seq  
      
    def to_obj_array(arr):  
        """Convert to object array if needed"""  
        if isinstance(arr, (list, tuple)):  
            out = obj_array(len(arr))  
            for i, val in enumerate(arr):  
                out[i] = val  
            return out  
        return arr  
      
    def get_joint_likelihood_seq(A, obs_seq, num_states):  
        """Compute joint likelihood sequence for observations"""  
        num_modalities = len(A)  
        lh_seq = obj_array(len(obs_seq))  
          
        for t, obs_t in enumerate(obs_seq):  
            # Initialize with ones for all factors  
            lh_joint = np.ones(num_states)  
              
            # Multiply likelihoods across modalities  
            for m in range(num_modalities):  
                # Get likelihood for this modality  
                modality_lh = A[m].T @ obs_t[m]  
                lh_joint = lh_joint * modality_lh  
              
            lh_seq[t] = lh_joint  
              
        return lh_seq  
      
    def calc_free_energy(qs, prior, num_factors, likelihood=None):  
        """Calculate variational free energy"""  
        F = 0.0  
        for f in range(num_factors):  
            ln_qs = spm_log_single(qs[f])  
            ln_prior = spm_log_single(prior[f])  
              
            if likelihood is not None:  
                ln_likelihood = spm_log_single(likelihood)  
                F += np.sum(qs[f] * (ln_qs - ln_likelihood))  
            else:  
                F += np.sum(qs[f] * (ln_qs - ln_prior))  
        return F  
      
    def get_expected_states(qs, B, action):  
        """Compute expected states given action"""  
        num_factors = len(qs)  
        expected_states = obj_array(num_factors)  
          
        for f in range(num_factors):  
            if len(action.shape) == 1:  
                action_f = int(action[f])  
            else:  
                action_f = int(action[0, f])  
              
            expected_states[f] = B[f][:, :, action_f].dot(qs[f])  
          
        return expected_states  
      
    def sample(p):  
        """Sample from categorical distribution"""  
        return np.random.choice(len(p), p=p)  
      
    # ===== MAIN FUNCTION LOGIC =====  
      
    observation = tuple(observation) if not distr_obs else observation  
  
    if not hasattr(self, "qs"):  
        self.reset()  
  
    if self.inference_algo == "VANILLA":  
        if self.action is not None:  
            empirical_prior = get_expected_states(  
                self.qs, self.B, self.action.reshape(1, -1)  
            )  
        else:  
            empirical_prior = self.D  
          
        # VANILLA inference: compute posterior directly  
        qs = obj_array(len(self.num_states))  
        for f in range(len(self.num_states)):  
            # Get likelihood for this factor  
            likelihood = np.zeros(self.num_states[f])  
            for m in range(len(self.A)):  
                if isinstance(observation[m], (int, np.integer)):  
                    obs_m = np.zeros(self.A[m].shape[0])  
                    obs_m[observation[m]] = 1.0  
                else:  
                    obs_m = observation[m]  
                  
                modality_likelihood = self.A[m][:, observation[m]] if isinstance(observation[m], (int, np.integer)) else self.A[m] @ obs_m  
                likelihood = likelihood * modality_likelihood if f == 0 else likelihood  
              
            # Combine with prior  
            if hasattr(empirical_prior, '__len__'):  
                posterior = likelihood * empirical_prior[f]  
            else:  
                posterior = likelihood * empirical_prior  
              
            # Normalize  
            qs[f] = posterior / (np.sum(posterior) + 1e-16)  
          
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
  
        # Initialize prev_actions properly  
        prev_actions = latest_actions  
        if prev_actions is not None:  
            prev_actions = np.stack(prev_actions, 0)  
  
        # Get model dimensions  
        num_obs, num_states, num_modalities, num_factors = get_model_dimensions(self.A, self.B)  
  
        # Process observations  
        prev_obs = process_observation_seq(latest_obs, num_modalities, num_obs)  
          
        # Get joint likelihood sequence  
        lh_seq = get_joint_likelihood_seq(self.A, prev_obs, num_states)  
          
        # Initialize storage for results  
        num_policies = len(self.policies)  
        qs_seq_pi = obj_array(num_policies)  
        xn_seq_pi = obj_array(num_policies)  
        vn_seq_pi = obj_array(num_policies)  
        F = np.zeros(num_policies)  
          
        # Process each policy  
        for p_idx, policy in enumerate(self.policies):  
            infer_len = len(prev_obs) + policy.shape[0]  
            past_len = len(prev_obs)  
            future_len = policy.shape[0]  
            future_cutoff = past_len + future_len - 2  
  
            # Initialize beliefs  
            qs_seq = obj_array(infer_len)  
            for t in range(infer_len):  
                qs_seq[t] = obj_array_uniform(num_states)  
  
            # Last message (for future boundary condition)  
            qs_T = obj_array_zeros(num_states)  
  
            # Set prior for this policy  
            if self.latest_belief is None:  
                prior = obj_array_uniform(num_states)  
            elif self.edge_handling_params['policy_sep_prior']:  
                prior = self.latest_belief[p_idx]  
            else:  
                prior = self.latest_belief  
  
            # Transposed transition matrices  
            trans_B = obj_array(num_factors)  
            for f in range(num_factors):  
                trans_B[f] = spm_norm(np.swapaxes(self.B[f], 0, 1))  
  
            # Stack previous actions with policy  
            if prev_actions is not None:  
                policy_stacked = np.vstack((prev_actions, policy))  
            else:  
                policy_stacked = policy  
  
            # Storage for intermediate variables  
            xn = []  
            vn = []  
            shape_list = [[num_states[f], infer_len] for f in range(num_factors)]  
  
            # Main variational iterations  
            for itr in range(num_iter):  
                xn_itr_all_factors = obj_array_zeros_list(shape_list)  
                vn_itr_all_factors = obj_array_zeros_list(shape_list)  
                F_policy = 0.0  
  
                for t in range(infer_len):  
                    for f in range(num_factors):  
                        # Likelihood term  
                        if t < past_len:  
                            lnA = spm_log_single(spm_dot(lh_seq[t], qs_seq[t], [f]))  
                        else:  
                            lnA = np.zeros(num_states[f])  
  
                        # Past message  
                        if t == 0:  
                            lnB_past = spm_log_single(prior[f])  
                        else:  
                            past_msg = self.B[f][:, :, int(policy_stacked[t - 1, f])].dot(qs_seq[t - 1][f])  
                            lnB_past = spm_log_single(past_msg)  
  
                        # Future message  
                        if t >= future_cutoff:  
                            lnB_future = qs_T[f]  
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
  
            # Store policy results  
            qs_seq_pi[p_idx] = qs_seq  
            xn_seq_pi[p_idx] = xn  
            vn_seq_pi[p_idx] = vn  
            F[p_idx] = F_policy  
  
        # Set final beliefs and free energy  
        qs = qs_seq_pi  
        self.F = F  
          
        # Flatten xn and vn for return format (matching reference implementation)  
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
