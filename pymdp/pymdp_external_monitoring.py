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
      
    def softmax(x):  
        """Numerically stable softmax implementation"""  
        x_max = np.max(x)  
        exp_x = np.exp(x - x_max)  
        return exp_x / np.sum(exp_x)  
      
    def spm_log_single(x):  
        """Numerically stable log function"""  
        return np.log(x + 1e-16)  
      
    def spm_norm(A):  
        """Normalize columns of matrix A to sum to 1"""  
        return A / (np.sum(A, axis=0) + 1e-16)  
      
    def spm_dot(A, B, factors=None):  
        """Special matrix multiplication for pymdp object arrays"""  
        if factors is None:  
            return A @ B  
        else:  
            result = 1.0  
            for f in factors:  
                result = result * A[f]  
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
            num_states = [B[f].shape[1] for f in range(num_factors)]  
        else:  
            num_factors = 0  
            num_states = []  
              
        return num_obs, num_states, num_modalities, num_factors  
      
    def process_observation_seq(obs_seq, num_modalities, num_obs):  
        """Process observation sequence into proper format"""  
        processed = []  
        for obs_t in obs_seq:  
            if isinstance(obs_t, (list, tuple)):  
                obs_array = obj_array(num_modalities)  
                for m in range(num_modalities):  
                    obs_array[m] = np.zeros(num_obs[m])  
                    obs_array[m][obs_t[m]] = 1.0  
                processed.append(obs_array)  
            else:  
                processed.append(obs_t)  
        return processed  
      
    def to_obj_array(arr):  
        """Convert array to object array format"""  
        if isinstance(arr, (list, tuple)):  
            out = obj_array(len(arr))  
            for i, item in enumerate(arr):  
                out[i] = item  
            return out  
        return arr  
      
    # ===== LIKELIHOOD AND FREE ENERGY FUNCTIONS =====  
      
    def get_joint_likelihood_seq(A, obs_seq, num_states):  
        """Compute joint likelihood sequence for observations"""  
        num_modalities = len(A)  
        lh_seq = obj_array(len(obs_seq))  
          
        for t, obs_t in enumerate(obs_seq):  
            # Initialize joint likelihood  
            lh_joint = np.ones(num_states[0])  
              
            # Multiply likelihoods across modalities  
            for m in range(num_modalities):  
                if isinstance(obs_t[m], np.ndarray):  
                    # Distributional observation  
                    likelihood_m = obs_t[m]  
                else:  
                    # One-hot observation  
                    likelihood_m = np.zeros(A[m].shape[0])  
                    likelihood_m[obs_t[m]] = 1.0  
                  
                # Get likelihood for this modality  
                modality_lh = A[m].T @ likelihood_m  
                lh_joint = lh_joint * modality_lh  
              
            lh_seq[t] = lh_joint  
              
        return lh_seq  
      
    def calc_free_energy(qs, prior, num_factors, likelihood=None):  
        """Calculate variational free energy"""  
        F = 0.0  
        for f in range(num_factors):  
            ln_qs = spm_log_single(qs[f])  
            ln_prior = spm_log_single(prior[f])  
            F += np.dot(qs[f], ln_qs - ln_prior)  
              
        if likelihood is not None:  
            F -= np.dot(qs[0], likelihood)  
              
        return F  
      
    # ===== CONTROL FUNCTIONS =====  
      
    def get_expected_states(qs, B, action):  
        """Compute expected states given current beliefs and action"""  
        num_factors = len(B)  
        expected_states = obj_array(num_factors)  
          
        for f in range(num_factors):  
            action_idx = int(action[0, f]) if action.ndim > 1 else int(action[f])  
            expected_states[f] = B[f][:, :, action_idx] @ qs[f]  
              
        return expected_states  
      
    def sample(probabilities):  
        """Sample from categorical distribution"""  
        return np.random.choice(len(probabilities), p=probabilities)  
      
    # ===== MAIN FUNCTION LOGIC =====  
      
    observation = tuple(observation) if not distr_obs else observation  
  
    if not hasattr(self, "qs"):  
        self.reset()  
  
    # Initialize return variables for VANILLA mode  
    xn = None  
    vn = None  
  
    if self.inference_algo == "VANILLA":  
        # VANILLA mode - implement inference.update_posterior_states() logic inline  
        if self.action is not None:  
            empirical_prior = get_expected_states(  
                self.qs, self.B, self.action.reshape(1, -1)  
            )[0]  
        else:  
            empirical_prior = self.D  
          
        # Compute posterior using Bayes' rule  
        num_modalities = len(self.A)  
        num_states = [self.A[m].shape[1] for m in range(num_modalities)]  
          
        # Initialize posterior beliefs  
        qs = obj_array(len(num_states))  
        for f in range(len(num_states)):  
            qs[f] = np.copy(empirical_prior[f])  
          
        # Update with observation likelihood  
        for m in range(num_modalities):  
            obs_idx = observation[m]  
            likelihood_m = self.A[m][:, obs_idx]  
            for f in range(len(num_states)):  
                qs[f] = qs[f] * likelihood_m  
          
        # Normalize  
        for f in range(len(num_states)):  
            qs[f] = qs[f] / (np.sum(qs[f]) + 1e-16)  
              
    elif self.inference_algo == "MMP":  
        # MMP mode - implement inference._update_posterior_states_full_test() logic inline  
          
        # Update observation history  
        self.prev_obs.append(observation)  
        if len(self.prev_obs) > self.inference_horizon:  
            latest_obs = self.prev_obs[-self.inference_horizon:]  
            latest_actions = self.prev_actions[-(self.inference_horizon-1):]  
        else:  
            latest_obs = self.prev_obs  
            latest_actions = self.prev_actions  
  
        # Get model dimensions  
        num_obs, num_states, num_modalities, num_factors = get_model_dimensions(self.A, self.B)  
  
        # Process observations  
        prev_obs = process_observation_seq(latest_obs, num_modalities, num_obs)  
          
        # Compute likelihood sequence  
        lh_seq = get_joint_likelihood_seq(self.A, prev_obs, num_states)  
  
        # Stack previous actions if they exist  
        if prev_actions is not None:  
            prev_actions = np.stack(prev_actions, 0)  
  
        # Initialize result arrays for all policies  
        qs_seq_pi = obj_array(len(self.policies))  
        xn_seq_pi = obj_array(len(self.policies))  
        vn_seq_pi = obj_array(len(self.policies))  
        F = np.zeros(len(self.policies))  
  
        # Process each policy  
        for p_idx, policy in enumerate(self.policies):  
              
            # === Inline _run_mmp_testing() logic ===  
              
            # Window parameters  
            past_len = len(lh_seq)  
            future_len = policy.shape[0]  
  
            if last_timestep:  
                infer_len = past_len + future_len - 1  
            else:  
                infer_len = past_len + future_len  
              
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
            xn = []  # beliefs across iterations  
            vn = []  # prediction errors across iterations  
  
            shape_list = [[num_states[f], infer_len] for f in range(num_factors)]  
              
            # Main variational iterations  
            for itr in range(num_iter):  
                xn_itr_all_factors = obj_array_zeros_list(shape_list)  
                vn_itr_all_factors = obj_array_zeros_list(shape_list)  
  
                F_policy = 0.0  # free energy for this policy  
                  
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
