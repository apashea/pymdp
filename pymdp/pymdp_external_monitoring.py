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
      
    # ===== EXACT PYMDP UTILS FUNCTIONS =====  
      
    def obj_array(shape):  
        """EXACT pymdp implementation"""  
        return np.empty(shape, dtype=object)  
      
    def is_obj_array(arr):  
        """EXACT pymdp implementation [1](#52-0) """  
        return arr.dtype == "object"  
  
    def to_obj_array(arr):  
        """EXACT pymdp implementation [2](#52-1) """  
        if is_obj_array(arr):  
            return arr  
        obj_array_out = obj_array(1)  
        obj_array_out[0] = arr.squeeze()  
        return obj_array_out  
  
    def onehot(value, num_values):  
        """EXACT pymdp implementation [3](#52-2) """  
        arr = np.zeros(num_values)  
        arr[value] = 1.0  
        return arr  
  
    def process_observation(obs, num_modalities, num_observations):  
        """EXACT pymdp implementation [4](#52-3) """  
        if isinstance(obs, np.ndarray) and not is_obj_array(obs):  
            assert num_modalities == 1, "If `obs` is a 1D numpy array, `num_modalities` must be equal to 1"  
            assert len(np.where(obs)[0]) == 1, "If `obs` is a 1D numpy array, it must be a one hot vector (e.g. np.array([0.0, 1.0, 0.0, ....]))"  
  
        if isinstance(obs, (int, np.integer)):  
            obs = onehot(obs, num_observations[0])  
  
        if isinstance(obs, tuple) or isinstance(obs,list):  
            obs_arr_arr = obj_array(num_modalities)  
            for m in range(num_modalities):  
                obs_arr_arr[m] = onehot(obs[m], num_observations[m])  
            obs = obs_arr_arr  
  
        return obs  
  
    def obj_array_uniform(shape_list):  
        """EXACT pymdp implementation [5](#52-4) """  
        arr = obj_array(len(shape_list))  
        for i, shape in enumerate(shape_list):  
            arr[i] = norm_dist(np.ones(shape))  
        return arr  
  
    def norm_dist(dist):  
        """EXACT pymdp implementation [6](#52-5) """  
        if dist.ndim == 3:  
            new_dist = np.zeros_like(dist)  
            for c in range(dist.shape[2]):  
                new_dist[:, :, c] = np.divide(dist[:, :, c], dist[:, :, c].sum(axis=0))  
            return new_dist  
        else:  
            return np.divide(dist, dist.sum(axis=0))  
  
    def norm_dist_obj_arr(obj_arr):  
        """EXACT pymdp implementation [7](#52-6) """  
        normed_obj_array = obj_array(len(obj_arr))  
        for i, arr in enumerate(obj_arr):  
            normed_obj_array[i] = norm_dist(arr)  
        return normed_obj_array  
  
    # ===== EXACT PYMDP MATHS FUNCTIONS =====  
      
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
        """EXACT pymdp implementation [8](#52-7) """  
        # Handle object array case  
        if is_obj_array(x):  
            dims = list((np.arange(0, len(x)) + X.ndim - len(x)).astype(int))  
        else:  
            dims = [1]  
            x = to_obj_array(x)  
  
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
  
    def spm_cross(A, B):  
        """EXACT pymdp implementation"""  
        if is_obj_array(A) and is_obj_array(B):  
            C = obj_array(len(A))  
            for i in range(len(A)):  
                C[i] = np.outer(A[i], B[i])  
        else:  
            C = np.outer(A, B)  
        return C  
  
    # ===== HELPER FUNCTIONS =====  
      
    def get_model_dimensions(A, B):  
        """Extract model dimensions"""  
        num_modalities = len(A)  
        num_factors = len(B)  
        num_obs = [A[m].shape[0] for m in range(num_modalities)]  
        num_states = [B[f].shape[0] for f in range(num_factors)]  
        return num_obs, num_states, num_modalities, num_factors  
  
    def process_observation_seq(obs_seq, num_modalities, num_observations):  
        """Process observation sequence"""  
        if not is_obj_array(obs_seq):  
            obs_seq = to_obj_array(obs_seq)  
          
        processed_seq = obj_array(len(obs_seq))  
        for t in range(len(obs_seq)):  
            processed_seq[t] = process_observation(obs_seq[t], num_modalities, num_observations)  
          
        return processed_seq  
  
    def get_joint_likelihood_seq(A, obs_seq, qs_seq, B):  
        """Compute joint likelihood sequence"""  
        num_obs, num_states, num_modalities, num_factors = get_model_dimensions(A, B)  
          
        # Process observations  
        obs_seq = process_observation_seq(obs_seq, num_modalities, num_obs)  
          
        # Compute likelihood for each timestep  
        lh_seq = obj_array(len(obs_seq))  
        for t in range(len(obs_seq)):  
            # Initialize joint likelihood  
            lh_joint = np.ones(num_states[0])  
              
            # Multiply likelihoods across modalities  
            for m in range(num_modalities):  
                modality_lh = A[m].T @ obs_seq[t][m]  
                lh_joint = lh_joint * modality_lh  
              
            lh_seq[t] = lh_joint  
          
        return lh_seq  
  
    # ===== MAIN INFERENCE LOGIC =====  
      
    if not hasattr(self, "qs"):  
        self.reset()  
  
    if self.inference_algo == "VANILLA":  
        # VANILLA inference - exact replication  
        if self.action is not None:  
            empirical_prior = self.B[:, :, int(self.action.reshape(1, -1)[0])].dot(self.qs[0])  
        else:  
            empirical_prior = self.D  
          
        # Compute likelihood  
        if not distr_obs:  
            num_modalities = len(self.A)  
            num_observations = [self.A[m].shape[0] for m in range(num_modalities)]  
            obs_processed = process_observation(observation, num_modalities, num_observations)  
            obs = to_obj_array(obs_processed)  
        else:  
            obs = observation  
          
        # Compute posterior  
        likelihood = obj_array(len(obs))  
        for m in range(len(obs)):  
            likelihood[m] = self.A[m][:, :, int(self.action.reshape(1, -1)[0])].dot(obs[m])  
          
        qs = norm_dist_obj_arr(likelihood * empirical_prior)  
        xn = None  
        vn = None  
          
    elif self.inference_algo == "MMP":  
        # MMP inference - exact replication  
        self.prev_obs.append(observation)  
        if len(self.prev_obs) > self.inference_horizon:  
            latest_obs = self.prev_obs[-self.inference_horizon:]  
            latest_actions = self.prev_actions[-(self.inference_horizon-1):]  
        else:  
            latest_obs = self.prev_obs  
            latest_actions = self.prev_actions  
  
        # Extract actions for this policy evaluation  
        if latest_actions is not None:  
            prev_actions = np.stack(latest_actions, 0)  
        else:  
            prev_actions = None  
  
        # Initialize storage for results  
        qs_seq_pi = obj_array(len(self.policies))  
        xn_seq_pi = obj_array(len(self.policies))  
        vn_seq_pi = obj_array(len(self.policies))  
        F = np.zeros(len(self.policies))  
  
        # Process each policy  
        for p_idx, policy in enumerate(self.policies):  
            # Initialize beliefs  
            num_factors = len(self.num_states)  
            qs_seq = obj_array(len(latest_obs))  
            for t in range(len(latest_obs)):  
                qs_seq[t] = obj_array_uniform(self.num_states)  
              
            # Initialize prior  
            prior = obj_array_uniform(self.num_states)  
              
            # Get joint likelihood sequence  
            lh_seq = get_joint_likelihood_seq(self.A, latest_obs, qs_seq, self.B)  
              
            # Variational iterations  
            xn = []  
            vn = []  
            F_policy = 0.0  
              
            for itr in range(num_iter):  
                xn_itr = obj_array(len(latest_obs))  
                vn_itr = obj_array(len(latest_obs))  
                  
                for t in range(len(latest_obs)):  
                    xn_itr[t] = obj_array(num_factors)  
                    vn_itr[t] = obj_array(num_factors)  
                      
                    for f in range(num_factors):  
                        if grad_descent:  
                            # Gradient descent update  
                            sx = qs_seq[t][f]  
                            lnqs = spm_log_single(sx)  
                            coeff = 1 if (t >= 1) else 2  
                              
                            # Likelihood term  
                            if t < len(latest_obs):  
                                lnA = spm_log_single(spm_dot(lh_seq[t], qs_seq[t], [f]))  
                            else:  
                                lnA = np.zeros(self.num_states[f])  
                              
                            # Past message  
                            if t == 0:  
                                lnB_past = spm_log_single(prior[f])  
                            else:  
                                past_msg = self.B[f][:, :, int(policy[t - 1, f])].dot(qs_seq[t - 1][f])  
                                lnB_past = spm_log_single(past_msg)  
                              
                            # Future message  
                            if t < len(latest_obs) - 1:  
                                future_msg = self.B[f][:, :, int(policy[t + 1, f])].dot(qs_seq[t + 1][f])  
                                lnB_future = spm_log_single(future_msg)  
                            else:  
                                lnB_future = np.zeros(self.num_states[f])  
                              
                            # Error and update  
                            err = (coeff * lnA + lnB_past + lnB_future) - coeff * lnqs  
                            vn_tmp = err - err.mean()  
                            lnqs = lnqs + tau * vn_tmp  
                            qs_seq[t][f] = softmax(lnqs)  
                              
                            # Store intermediate values  
                            xn_itr[t][f] = qs_seq[t][f]  
                            vn_itr[t][f] = vn_tmp  
                              
                            # Free energy accumulation  
                            if (t == 0) or (t == (len(latest_obs)-1)):  
                                F_policy += sx.dot(0.5 * err)  
                            else:  
                                F_policy += sx.dot(0.5 * (err - (num_factors - 1) * lnA / num_factors))  
                  
                xn.append(xn_itr)  
                vn.append(vn_itr)  
              
            # Store policy results  
            qs_seq_pi[p_idx] = qs_seq  
            xn_seq_pi[p_idx] = xn  
            vn_seq_pi[p_idx] = vn  
            F[p_idx] = F_policy  
  
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
