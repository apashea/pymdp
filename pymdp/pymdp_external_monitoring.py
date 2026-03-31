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
        """EXACT pymdp implementation<cite repo="apashea/pymdp" path="pymdp/maths.py" start="18-54" />"""  
        # Construct dims to perform dot product on  
        if hasattr(x, '__len__') and not isinstance(x, str):  
            # Object array case  
            dims = list(range(X.ndim - len(x), len(x) + X.ndim - len(x)))  
        else:  
            dims = [1]  
            x = [x]  # Convert to list for uniform handling  
  
        if dims_to_omit is not None:  
            arg_list = [X, list(range(X.ndim))] + list(chain(*([x[xdim_i],[dims[xdim_i]]] for xdim_i in range(len(x)) if xdim_i not in dims_to_omit))) + [dims_to_omit]  
        else:  
            arg_list = [X, list(range(X.ndim))] + list(chain(*([x[xdim_i],[dims[xdim_i]]] for xdim_i in range(len(x))))) + [[0]]  
  
        Y = np.einsum(*arg_list)  
  
        # check to see if `Y` is a scalar  
        if np.prod(Y.shape) <= 1.0:  
            Y = Y.item()  
            Y = np.array([Y]).astype("float64")  
  
        return Y  
      
    def norm_dist(dist):  
        """EXACT pymdp implementation<cite repo="apashea/pymdp" path="pymdp/utils.py" start="217-225" />"""  
        if dist.ndim == 3:  
            new_dist = np.zeros_like(dist)  
            for c in range(dist.shape[2]):  
                new_dist[:, :, c] = np.divide(dist[:, :, c], dist[:, :, c].sum(axis=0))  
            return new_dist  
        else:  
            return np.divide(dist, dist.sum(axis=0))  
      
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
      
    def norm_dist_obj_arr(obj_array):  
        """Normalize object array of distributions"""  
        for i in range(len(obj_array)):  
            obj_array[i] = norm_dist(obj_array[i])  
        return obj_array  
      
    def get_model_dimensions(A, B):  
        """Extract model dimensions from A and B matrices"""  
        num_modalities = len(A)  
        num_obs = [A[m].shape[0] for m in range(num_modalities)]  
          
        num_factors = len(B)  
        num_states = [B[f].shape[0] for f in range(num_factors)]  
          
        return num_obs, num_modalities, num_states, num_factors  
      
    def process_observation_seq(obs_seq, num_modalities, num_obs):  
        """Process observation sequence into one-hot format"""  
        processed_seq = obj_array(len(obs_seq))  
        for t, obs_t in enumerate(obs_seq):  
            obs_array = obj_array(num_modalities)  
            for m in range(num_modalities):  
                obs_m = np.zeros(num_obs[m])  
                obs_m[int(obs_t[m])] = 1.0  
                obs_array[m] = obs_m  
            processed_seq[t] = obs_array  
        return processed_seq  
      
    def get_joint_likelihood_seq(A, obs_seq, num_states):  
        """Compute joint likelihood sequence for observations"""  
        num_modalities = len(A)  
        lh_seq = obj_array(len(obs_seq))  
          
        for t, obs_t in enumerate(obs_seq):  
            lh_joint = np.ones(num_states)  
            for m in range(num_modalities):  
                modality_lh = A[m].T @ obs_t[m]  
                lh_joint = lh_joint * modality_lh  
            lh_seq[t] = lh_joint  
          
        return lh_seq  
      
    def get_expected_states(qs, B, action):  
        """Compute expected states given beliefs and action"""  
        num_factors = len(qs)  
        expected_states = obj_array(num_factors)  
          
        for f in range(num_factors):  
            if action.ndim == 1:  
                action_f = int(action[f])  
            else:  
                action_f = int(action[0, f])  
            expected_states[f] = B[f][:, :, action_f] @ qs[f]  
          
        return expected_states  
      
    def sample(p):  
        """Sample from categorical distribution"""  
        return np.random.choice(len(p), p=p)  
      
    # ===== MAIN INFERENCE LOGIC =====  
      
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
          
        # VANILLA inference: compute posterior using Bayes' rule  
        qs = obj_array_uniform([self.A[m].shape[1] for m in range(len(self.A))])  
          
        for f in range(len(qs)):  
            # Start with prior  
            if self.action is not None:  
                qs[f] = empirical_prior[f]  
            else:  
                qs[f] = self.D[f]  
              
            # Multiply by likelihood for each modality  
            for m in range(len(self.A)):  
                likelihood_m = self.A[m][:, int(observation[m])]  
                qs[f] = qs[f] * likelihood_m  
              
            # Normalize  
            qs[f] = qs[f] / np.sum(qs[f])  
          
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
        num_obs, num_modalities, num_states, num_factors = get_model_dimensions(self.A, self.B)  
  
        # Process observations  
        prev_obs = process_observation_seq(latest_obs, num_modalities, num_obs)  
          
        # Get joint likelihood sequence  
        lh_seq = get_joint_likelihood_seq(self.A, prev_obs, num_states)  
  
        # Initialize variables for MMP  
        past_len = len(prev_obs)  
        infer_len = past_len + self.policy_len  
        future_cutoff = past_len - 1  
          
        # Initialize belief sequences  
        qs_seq_pi = obj_array(len(self.policies))  
        xn_seq_pi = obj_array(len(self.policies))  
        vn_seq_pi = obj_array(len(self.policies))  
        F = np.zeros(len(self.policies))  
          
        # Get prior  
        if hasattr(self, 'latest_belief'):  
            prior = self.latest_belief  
        else:  
            prior = self.D  
  
        # Process each policy  
        for p_idx, policy in enumerate(self.policies):  
            # Initialize belief sequence for this policy  
            qs_seq = obj_array(infer_len)  
            for t in range(infer_len):  
                qs_seq[t] = obj_array_uniform(num_states)  
              
            # Set prior at t=0  
            qs_seq[0] = prior  
              
            # Initialize intermediate variables  
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
                            past_msg = self.B[f][:, :, int(policy[t - 1, f])] @ qs_seq[t - 1][f]  
                            lnB_past = spm_log_single(past_msg)  
                          
                        # Future message  
                        if t >= future_cutoff:  
                            lnB_future = spm_log_single(qs_seq[infer_len - 1][f])  
                        else:  
                            future_msg = self.B[f][:, :, int(policy[t, f])] @ qs_seq[t + 1][f]  
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
