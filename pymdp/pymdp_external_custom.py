# Andrew Pashea 2025



def custom_test_print_function(optional=None):
  if optional:
    print(optional)
  else:
    print("Function functional.")

import itertools
import numpy as np
from pymdp.maths import softmax, softmax_obj_arr, spm_dot, spm_wnorm, spm_MDP_G, spm_log_single, spm_log_obj_array
from pymdp.control import get_expected_obs, calc_expected_utility, calc_states_info_gain, calc_pA_info_gain, calc_pB_info_gain
from pymdp import utils
import copy

def update_posterior_policies_full_info(
    qs_seq_pi,
    A,
    B,
    C,
    policies,
    use_utility=True,
    use_states_info_gain=True,
    use_param_info_gain=False,
    prior=None,
    pA=None,
    pB=None,
    F = None,
    E = None,
    gamma=16.0
    ):
    """
    Update posterior beliefs about policies by computing expected free energy of each policy and integrating that
    with the variational free energy of policies ``F`` and prior over policies ``E``. This is intended to be used in conjunction
    with the ``update_posterior_states_full`` method of ``inference.py``, since the full posterior over future timesteps, under all policies, is
    assumed to be provided in the input array ``qs_seq_pi``.

    """

    num_obs, num_states, num_modalities, num_factors = utils.get_model_dimensions(A, B)
    horizon = len(qs_seq_pi[0])
    num_policies = len(qs_seq_pi)

    qo_seq = utils.obj_array(horizon)
    for t in range(horizon):
        qo_seq[t] = utils.obj_array_zeros(num_obs)

    # initialise expected observations
    qo_seq_pi = utils.obj_array(num_policies)

    # initialize (negative) expected free energies for all policies
    G = np.zeros(num_policies)
    expected_utility = np.zeros(num_policies)
    info_gain = np.zeros(num_policies)
    pA_info_gain = np.zeros(num_policies)
    pB_info_gain = np.zeros(num_policies)

    if F is None:
        F = spm_log_single(np.ones(num_policies) / num_policies)

    if E is None:
        lnE = spm_log_single(np.ones(num_policies) / num_policies)
    else:
        lnE = spm_log_single(E)

    for p_idx, policy in enumerate(policies):

        qo_seq_pi[p_idx] = get_expected_obs(qs_seq_pi[p_idx], A)

        if use_utility:
            expected_utility[p_idx] = calc_expected_utility(qo_seq_pi[p_idx], C)
            G[p_idx] += expected_utility[p_idx]

        if use_states_info_gain:
            info_gain[p_idx] = calc_states_info_gain(A, qs_seq_pi[p_idx])
            G[p_idx] += info_gain[p_idx]

        if use_param_info_gain:
            if pA is not None:
                pA_info_gain[p_idx] = calc_pA_info_gain(pA, qo_seq_pi[p_idx], qs_seq_pi[p_idx])
                G[p_idx] += pA_info_gain[p_idx]
            if pB is not None:
                pB_info_gain[p_idx] = calc_pB_info_gain(pB, qs_seq_pi[p_idx], prior, policy)
                G[p_idx] += pB_info_gain[p_idx]

    q_pi = softmax(G * gamma - F + lnE)

    if use_param_info_gain == False:
        return q_pi, G, expected_utility, info_gain
    else:
        return q_pi, G, expected_utility, info_gain, pA_info_gain, pB_info_gain

def infer_policies_info(self):
        """
        Perform policy inference by optimizing a posterior (categorical) distribution over policies.
        This distribution is computed as the softmax of ``G * gamma + lnE`` where ``G`` is the negative expected
        free energy of policies, ``gamma`` is a policy precision and ``lnE`` is the (log) prior probability of policies.
        This function returns the posterior over policies as well as the negative expected free energy of each policy.

        Returns
        ----------
        q_pi: 1D ``numpy.ndarray``
            Posterior beliefs over policies, i.e. a vector containing one posterior probability per policy.
        G: 1D ``numpy.ndarray``
            Negative expected free energies of each policy, i.e. a vector containing one negative expected free energy per policy.
        """

        if self.inference_algo == "VANILLA":
            q_pi, G = control.update_posterior_policies(
                self.qs,
                self.A,
                self.B,
                self.C,
                self.policies,
                self.use_utility,
                self.use_states_info_gain,
                self.use_param_info_gain,
                self.pA,
                self.pB,
                E = self.E,
                gamma = self.gamma
            )
        elif self.inference_algo == "MMP":

            future_qs_seq = self.get_future_qs()

            if self.use_param_info_gain == True:
              q_pi, G, expected_utility, info_gain, pA_info_gain, pB_info_gain = update_posterior_policies_full_info(
                future_qs_seq,
                self.A,
                self.B,
                self.C,
                self.policies,
                self.use_utility,
                self.use_states_info_gain,
                self.use_param_info_gain,
                self.latest_belief,
                self.pA,
                self.pB,
                F = self.F,
                E = self.E,
                gamma = self.gamma
              )

            else:
              q_pi, G, expected_utility, info_gain = update_posterior_policies_full_info(
                future_qs_seq,
                self.A,
                self.B,
                self.C,
                self.policies,
                self.use_utility,
                self.use_states_info_gain,
                self.use_param_info_gain,
                self.latest_belief,
                self.pA,
                self.pB,
                F = self.F,
                E = self.E,
                gamma = self.gamma
              )

        if hasattr(self, "q_pi_hist"):
            self.q_pi_hist.append(q_pi)
            if len(self.q_pi_hist) > self.inference_horizon:
                self.q_pi_hist = self.q_pi_hist[-(self.inference_horizon-1):]

        self.q_pi = q_pi
        self.G = G
        if self.use_param_info_gain == True:
          return q_pi, G, expected_utility, info_gain, pA_info_gain, pB_info_gain
        else:
          return q_pi, G, expected_utility, info_gain

def select_highest_equivalent(options_array):
    """Equivalent to control.select_highest()"""
    # options_with_idx = np.array(list(enumerate(options_array)))
    # same_prob = options_with_idx[
    #     abs(options_with_idx[:, 1] - np.amax(options_with_idx[:, 1])) <= 1e-8][:, 0]
    # if len(same_prob) > 1:
    #     return int(same_prob[np.random.choice(len(same_prob))])
    # return int(same_prob[0])
    max_val = np.max(options_array)  
    max_indices = np.where(np.abs(options_array - max_val) < 1e-8)[0]  
    return max_indices[0]  # Always choose first index
    

def infer_states_info(self, observation, distr_obs=False):
        from pymdp import inference
        """
        Test version of ``infer_states()`` that additionally returns intermediate variables of MMP, such as
        the prediction errors and intermediate beliefs from the optimization. Used for benchmarking against SPM outputs.
        """
        observation = tuple(observation) if not distr_obs else observation

        if not hasattr(self, "qs"):
            self.reset()

        if self.inference_algo == "VANILLA":
            if self.action is not None:
                empirical_prior = control.get_expected_states(
                    self.qs, self.B, self.action.reshape(1, -1) #type: ignore
                )
            else:
                empirical_prior = self.D
            qs = inference.update_posterior_states(
            self.A,
            observation,
            empirical_prior,
            **self.inference_params
            )
        elif self.inference_algo == "MMP":

            self.prev_obs.append(observation)
            if len(self.prev_obs) > self.inference_horizon:
                latest_obs = self.prev_obs[-self.inference_horizon:]
                latest_actions = self.prev_actions[-(self.inference_horizon-1):]
            else:
                latest_obs = self.prev_obs
                latest_actions = self.prev_actions

            qs, F, xn, vn = inference._update_posterior_states_full_test(
                self.A,
                self.B,
                latest_obs,
                self.policies,
                latest_actions,
                prior = self.latest_belief,
                policy_sep_prior = self.edge_handling_params['policy_sep_prior'],
                **self.inference_params
            )

            self.F = F # variational free energy of each policy

        if hasattr(self, "qs_hist"):
            self.qs_hist.append(qs)

        self.qs = qs

        return qs, xn, vn

def update_A_MMP_distributional(agent, obs, distr_obs=False):
    # Process observations based on distr_obs flag
    if not distr_obs:
        # Convert integer observations to distributional format
        num_modalities = len(agent.A)
        num_observations = [agent.A[m].shape[0] for m in range(num_modalities)]
        obs_processed = utils.process_observation(obs, num_modalities, num_observations)
        obs = utils.to_obj_array(obs_processed)

    # Bayesian model average
    current_beliefs_by_policy = utils.obj_array(len(agent.policies))
    for p_idx in range(len(agent.policies)):
        current_beliefs_by_policy[p_idx] = agent.qs[p_idx][0]

    qs_for_learning = inference.average_states_over_policies(
        current_beliefs_by_policy,
        agent.q_pi
    )

    # Dirichlet update using distributional observations
    for m in range(len(obs)):
        update = maths.spm_cross(obs[m], qs_for_learning)
        agent.pA[m] += agent.lr_pA * update
        agent.A[m] = utils.norm_dist(agent.pA[m])

    return agent.A

import copy

def update_gamma(self, num_iterations=16, step_size=2.0):
    """
    Update policy precision (gamma) using iterative precision updates based on expected free energy prediction errors.
    This should be called after infer_states() and infer_policies() to refine the policy precision based on the computed F and G values.

    Parameters
    ----------
    num_iterations : int, default 16
        Number of variational iterations for precision updates
    step_size : float, default 2.0
        Step size parameter (psi) for gradient descent on precision

    Returns
    -------
    gamma_history : np.ndarray
        History of gamma values across iterations (for dopamine computation)
    policy_posteriors_history : np.ndarray
        History of policy posteriors across iterations
    """

    if not hasattr(self, 'F') or not hasattr(self, 'G'):
        raise ValueError("Must call infer_states() and infer_policies() before update_gamma()")

    # Get current values from agent state - use deep copy to avoid mutations
    F = copy.deepcopy(self.F)  # Variational free energy from infer_states()
    G = copy.deepcopy(self.G)  # Expected free energy from infer_policies()
    E = copy.deepcopy(self.E) if hasattr(self, 'E') else np.ones(len(self.policies)) / len(self.policies)

    # Initialize precision parameters
    gamma_0 = self.gamma
    gamma = gamma_0
    beta_prior = 1.0 / gamma
    beta_posterior = beta_prior

    # Storage for dopamine computation
    gamma_history = []
    policy_posteriors_history = []

    for i in range(num_iterations):
        # Compute policy priors and posteriors using current gamma
        log_E = np.log(E + 1e-16)

        # Prior over policies (without variational free energy F)
        exp_term_prior = np.exp(log_E - gamma * G)
        pi_0 = exp_term_prior / np.sum(exp_term_prior)

        # Posterior over policies (including variational free energy F)
        exp_term_posterior = np.exp(log_E - gamma * G - F)
        pi_posterior = exp_term_posterior / np.sum(exp_term_posterior)

        # Expected free energy prediction error
        G_error = np.dot((pi_posterior - pi_0), -G)

        # Update precision using gradient descent
        beta_update = beta_posterior - beta_prior + G_error
        beta_posterior = beta_posterior - beta_update / step_size
        gamma = 1.0 / beta_posterior

        # Store for dopamine computation - use deep copy to prevent mutations
        gamma_history.append(copy.deepcopy(gamma))
        policy_posteriors_history.append(copy.deepcopy(pi_posterior))

    # Update agent's gamma and store dopamine-related variables
    self.gamma = gamma
    self.beta = beta_posterior
    self.gamma_history = np.array(gamma_history)
    self.policy_posteriors_history = np.array(policy_posteriors_history).T

    # Compute dopamine signals (tonic and phasic)
    gamma_extended = np.concatenate(([gamma_0, gamma_0, gamma_0], self.gamma_history))
    self.tonic_dopamine = copy.deepcopy(gamma_extended)
    self.phasic_dopamine = copy.deepcopy(np.gradient(gamma_extended))

    # Update final policy posterior with refined gamma
    log_E = np.log(E + 1e-16)
    exp_term_final = np.exp(log_E - self.gamma * G - F)
    self.q_pi = exp_term_final / np.sum(exp_term_final)

    return self.gamma, self.beta, copy.deepcopy(self.gamma_history), copy.deepcopy(self.policy_posteriors_history)

def update_C_MMP_distributional(self, obs, lr_pC=1.0, distr_obs=False, initial_scale=1.0, modalities_to_learn=None, monitoring=False):
    """
    Update agent's prior preferences (C) using observed outcomes.
    Implements a Dirichlet update analogous to SPM MDP.

    Parameters
    ----------
    agent : object
        The agent containing C (preferences) and pC (Dirichlet params).
    obs : array or list
        Observed outcomes for each modality (can be ints or one-hot arrays).
    lr_C : float, default 1.0
        Learning rate for Dirichlet updates.
    distr_obs : bool, default False
        Treat obs as distributional if True, else convert.

    Returns
    -------
    new_C : object array
        Updated normalized preferences (prior over outcomes).
    """
    if not hasattr(self, 'pC'):
        self.pC = utils.dirichlet_like(self.C, scale=initial_scale)
        if monitoring==True:
          print(f"No pC attribute found. Defining prior pC with scale {initial_scale}:")
          print(f"self.pC = {self.pC}")
    C = copy.deepcopy(self.C)
    pC = copy.deepcopy(self.pC)

    if not distr_obs:
        if monitoring == True:
          print(f"distr_obs=False : Processing raw observation {obs}...")
        num_modalities = len(C)
        num_outcomes = [C[m].shape for m in range(num_modalities)]
        if monitoring == True:
          print(f"num_outcomes = [C[m].shape for m in range(num_modalities)] = {num_outcomes}")
        obs_processed = copy.deepcopy(utils.process_observation(obs, num_modalities, num_outcomes))
        obs = copy.deepcopy(utils.to_obj_array(obs_processed))
        if monitoring == True:
          print(f"Processing obs = {obs}")

    # Dirichlet update: counts incremented by observed outcomes modulated by learning rate (as in SPM MDP)
    if modalities_to_learn is None:
        modalities_to_learn= list(range(len(obs)))
        if monitoring == True:
          print(f"modalities_to_learn = {modalities_to_learn}")
    for m in modalities_to_learn:
        update = lr_pC * copy.deepcopy(obs[m])
        pC[m] += update
        if monitoring == True:
          print(f"pC[{m}] += {update} = {pC[m]}")
        # Normalize (convert Dirichlet params to preference probabilities)
        C[m] = utils.norm_dist(pC[m])
        if monitoring == True:
          print(f"C[{m}] = {C[m]}")

    self.C = copy.deepcopy(C)
    self.pC = copy.deepcopy(pC)
    if monitoring == True:
      print(f"Final updated C = {C}")
      print(f"Final updated pC = {pC}")

    return C

def update_E(self, q_pi, lr_pE=1.0, initial_scale=1.0, monitoring=False):
      """
      Update policy prior parameters (habits) using Dirichlet learning
      c.f.
      - Adams et. al, 2022: Everything is connected: Inference and attractors in delusions
        - uses spm_MDP_VB_X_rand.m (a copy of the original allowing random seed setting) and their code https://github.com/Peter-Vincent/MDP_Delusions
          which involves habit learning by adding the posterior distribution over policies, multiplied by a learning rate, to the prior distribution over habits. E is then set to be the normalized posterior over policies.
          This means NOT adding individual counts, from which arises conceptually an interesting issue, e.g., 'stochastic' action selection could involve building habits
          for policies that were high probability but *not actually executed*; alternatively if sampled actions were used to *actually* increment pE counts, this would assume
          the agent knew which action it selected in a way that escapes self-observation (i.e. if there were no observation modality for 'witnessing' the sampled action).
      - Active Inference 2022 textbook, Appendix for learning rules: `e = e + \boldsymbol{pi}` i.e. we add Q(policies) to the Dirichlet prior `e` on E
      - Saini et. al 2022:  https://www.nature.com/articles/s41598-022-22277-y
      Parameters
      ----------
      q_pi : numpy.ndarray
          Posterior beliefs over policies from current timestep
      """
      if not hasattr(self, 'pE'):
        self.pE = utils.dirichlet_like(self.E, scale=initial_scale)
        if monitoring == True:
          print(f"No pE attribute found. Defining prior pE with scale {initial_scale}:")
          print(f"self.pE = {self.pE}")

      if self.pE is not None:
          qE = copy.deepcopy(self.pE[0]) + np.dot(lr_pE, copy.deepcopy(q_pi))     #)
          # Update Dirichlet parameters
          self.pE = qE
          # Update policy prior for next timestep
          self.E = utils.norm_dist(copy.deepcopy(qE))
          if monitoring == True:
            print(f"Updated E: new self.E with type {type(self.E)} and length {len(self.E)} = {self.E}")
            print(f"self.pE with type {type(self.pE)} and length {len(self.pE)} = {self.pE}")

          return qE


def compute_current_qs_bma(agent, current_timestep_idx=None, save_history=False):
    from pymdp import inference
    # c.f. Agent class `set_latest_beliefs(self,last_belief=None)`

    if save_history:
      if hasattr(agent, 'qs_current_bma'):
          agent.qs_prev_bma = agent.qs_current_bma
      else:
          agent.qs_prev_bma = agent.D

    if current_timestep_idx is None:
      # The temporal window of beliefs in MMP's `qs` returned by run_mmp_factorized() -- e.g., when len(agent.qs[0]) = 5 means 5 timesteps are represented in beliefs about states --
      # is `infer_len = past_len + future_len = len(lh_seq) + policy.shape[0]` where lh_seq is the log-likelihood sequence of prev_obs stored and policy.shape[0] is the temporal depth (timesteps) of
      # the policy passed.
      total_timesteps = len(agent.qs[0])

      # Extract latest_obs length (this is what determines past_len in MMP; there is no check if custom policies supplied to Agent() are necessarily equal to policy_len so it is safer)
      # to use the past_len based on comparing the agent's prev_obs and inference_horizon as point of reference for determining current_timestep_idx)
      # The current timestep in the MMP `qs` is at the end of the past_len horizon, i.e. the beliefs resulting from the most recent real observation.
      latest_obs_len = min(len(agent.prev_obs), agent.inference_horizon)
      current_timestep_idx = latest_obs_len - 1


    # Compute BMA
    current_qs_pi = utils.obj_array(len(agent.policies))
    for p_idx in range(len(agent.policies)):
        current_qs_pi[p_idx] = agent.qs[p_idx][current_timestep_idx]

    qs_current_bma = inference.average_states_over_policies(
        current_qs_pi, agent.q_pi
    )
    agent.qs_current_bma = qs_current_bma

    if save_history == True:
      if not hasattr(agent, 'qs_current_bma_hist'):
        agent.qs_current_bma_hist = []
      agent.qs_current_bma_hist.append(qs_current_bma)

    return qs_current_bma

def sample_action_timestep_dependent(agent, timestep=0):
    """
    External function to sample actions from the appropriate timestep of policies.

    Parameters
    ----------
    agent : pymdp.agent.Agent
        The agent instance with inferred policy posterior (q_pi)
    timestep : int
        The timestep index to sample actions from (0-indexed)

    Returns
    -------
    selected_action : np.ndarray
        Vector containing the indices of the actions for each control factor
    """

    # Ensure timestep is within policy bounds
    max_policy_len = max([policy.shape[0] for policy in agent.policies])
    policy_timestep = min(timestep, max_policy_len - 1)

    if agent.sampling_mode == "marginal":
        action = _sample_action_marginal_external(agent, policy_timestep)
    elif agent.sampling_mode == "full":
        action = _sample_policy_external(agent, policy_timestep)
    else:
        raise ValueError(f"Unknown sampling mode: {agent.sampling_mode}")

    # Store the action and advance time (crucial for MMP inference)
    agent.action = action
    agent.step_time()

    return action

def _sample_action_marginal_external(agent, timestep):
    """Marginal action sampling from specified timestep"""
    num_factors = len(agent.num_controls)
    action_marginals = utils.obj_array_zeros(agent.num_controls)

    # Weight actions from the specified timestep across all policies
    for pol_idx, policy in enumerate(agent.policies):
        if timestep < policy.shape[0]:  # Check timestep exists in policy
            for factor_i, action_i in enumerate(policy[timestep, :]):
                action_marginals[factor_i][int(action_i)] += agent.q_pi[pol_idx]

    action_marginals = utils.norm_dist_obj_arr(action_marginals)

    selected_action = np.zeros(num_factors)
    for factor_i in range(num_factors):
        if agent.action_selection == 'deterministic':
            selected_action[factor_i] = select_highest_equivalent(action_marginals[factor_i])
        elif agent.action_selection == 'stochastic':
            log_marginal_f = control.spm_log_single(action_marginals[factor_i])
            p_actions = control.softmax(log_marginal_f * agent.alpha)
            selected_action[factor_i] = utils.sample(p_actions)

    return selected_action.astype(int)

def _sample_policy_external(agent, timestep):
    """Full policy sampling using specified timestep"""
    num_factors = len(agent.num_controls)

    # Sample a policy using the equivalent function
    if agent.action_selection == "deterministic":
        policy_idx = select_highest_equivalent(agent.q_pi)
    elif agent.action_selection == "stochastic":
        log_qpi = control.spm_log_single(agent.q_pi)
        p_policies = control.softmax(log_qpi * agent.alpha)
        policy_idx = utils.sample(p_policies)

    # Extract action from the appropriate timestep
    selected_action = np.zeros(num_factors)
    policy = agent.policies[policy_idx]

    if timestep < policy.shape[0]:
        for factor_i in range(num_factors):
            selected_action[factor_i] = policy[timestep, factor_i]
    else:
        # If timestep exceeds policy length, use the last timestep
        for factor_i in range(num_factors):
            selected_action[factor_i] = policy[-1, factor_i]

    return selected_action.astype(int)

def compute_xn_bma(agent, xn, policy_len=None, current_timestep_idx=None, save_history=False, reduce=True):
    """
    Compute Bayesian model average of iterative posterior beliefs (xn) across policies.

    Parameters
    ----------
    agent : Agent or None
        The agent instance containing q_pi (policy posterior). If None, q_pi must be provided separately
        and agent-related logic is skipped.
    xn : numpy.ndarray of dtype object
        Nested structure xn[p][itr][f] containing beliefs across policies, iterations, and factors.
        Each xn[p][itr][f] is a 2D array of shape (num_states[f], infer_len).
    policy_len : int, optional
        The policy length (future_len) to use when agent is None. Used to compute current_timestep_idx
        dynamically from xn structure: current_timestep_idx = infer_len - policy_len - 1.
        Ignored if agent is provided.
    current_timestep_idx : int, optional
        The timestep index to extract from the infer_len dimension. If None, computed automatically.
    save_history : bool, default False
        Whether to save the xn_bma to agent history (only if agent is not None)
    reduce : bool, default True
        If True, reduces xn_bma to structure xn_bma[itr][f] where each element is a 1D array
        of shape (num_states[f],) by extracting beliefs at current_timestep_idx.
        If False, retains full temporal dimension with structure xn_bma[itr][f] where each
        element is a 2D array of shape (num_states[f], infer_len).

    Returns
    -------
    xn_bma : numpy.ndarray of dtype object
        If reduce=True: Structure xn_bma[itr][f] containing BMA beliefs across iterations and factors.
        Each xn_bma[itr][f] is a 1D array of shape (num_states[f],) at the current timestep.
        If reduce=False: Structure xn_bma[itr][f] containing BMA beliefs across iterations and factors.
        Each xn_bma[itr][f] is a 2D array of shape (num_states[f], infer_len).

    Sample code (without agent):
    xn_bma_current = compute_xn_bma(None, xn, policy_len=2)
    vn_bma_current = compute_xn_bma(None, vn, policy_len=2)
    """

    if agent is None:
        # When agent is None, compute current_timestep_idx from xn structure and policy_len
        if current_timestep_idx is None and policy_len is not None:
            # Get infer_len from the shape of xn
            infer_len = xn[0][0][0].shape[1]
            # Calculate past_len: infer_len = past_len + future_len, where future_len = policy_len
            past_len = infer_len - policy_len
            # Current timestep is the last observed timestep (end of past window)
            current_timestep_idx = past_len - 1
        # Extract q_pi from xn structure - assume uniform policy weights
        num_policies = len(xn)
        q_pi = np.ones(num_policies) / num_policies
    else:
        # Agent-provided path
        if save_history:
            if hasattr(agent, 'xn_current_bma'):
                agent.xn_prev_bma = agent.xn_current_bma

        # Determine current timestep index if not provided
        if current_timestep_idx is None:
            latest_obs_len = min(len(agent.prev_obs), agent.inference_horizon)
            current_timestep_idx = latest_obs_len - 1

        q_pi = agent.q_pi

    # Get dimensions from xn structure
    num_policies = len(xn)
    num_iterations = len(xn[0])  # Number of MMP iterations
    num_factors = len(xn[0][0])  # Number of hidden state factors

    # Initialize xn_bma with structure [itr][f]
    xn_bma = utils.obj_array(num_iterations)

    for itr in range(num_iterations):
        # For each iteration, compute BMA across policies for each factor
        xn_bma[itr] = utils.obj_array(num_factors)

        for f in range(num_factors):
            # Get the shape from the first policy's xn array
            num_states_f, infer_len = xn[0][itr][f].shape

            if reduce:
                # Initialize the BMA array for this iteration and factor (1D)
                xn_bma[itr][f] = np.zeros(num_states_f)

                # Compute weighted average across policies using q_pi, extracting current timestep
                for p_idx in range(num_policies):
                    policy_weight = q_pi[p_idx]
                    xn_bma[itr][f] += xn[p_idx][itr][f][:, current_timestep_idx] * policy_weight
            else:
                # Initialize the BMA array for this iteration and factor (2D)
                xn_bma[itr][f] = np.zeros((num_states_f, infer_len))

                # Compute weighted average across policies using q_pi
                for p_idx in range(num_policies):
                    policy_weight = q_pi[p_idx]
                    xn_bma[itr][f] += xn[p_idx][itr][f] * policy_weight

    # Store in agent if requested and agent is not None
    if agent is not None:
        agent.xn_current_bma = xn_bma

        if save_history:
            if not hasattr(agent, 'xn_current_bma_hist'):
                agent.xn_current_bma_hist = []
            agent.xn_current_bma_hist.append(xn_bma)

    return xn_bma

# SPM uses log and convolution for filtering; this is a basic moving-window convolution to mimic SPM's bandpass.
def spm_conv(data, window_size):
    # Cap window size at length of data to avoid mismatched shapes
    win = min(window_size, data.shape[0])
    kernel = np.ones(win) / win
    smoothed = np.array([
        np.convolve(data[:, i], kernel, mode='same')
        for i in range(data.shape[1])
    ]).T
    return smoothed

def stack_xn_over_iterations(xn, factor_idx):
    # Collect as array: [iterations, states]
    return np.stack([np.asarray(xn_iter[factor_idx], dtype=np.float64) for xn_iter in xn])

def compute_spm_lfp(xn, factor_idx, win_fast=2, win_slow=16):
    xn_factor = stack_xn_over_iterations(xn, factor_idx)
    log_xn = np.log(xn_factor + 1e-8)
    # windows are capped inside spm_conv
    fast = spm_conv(log_xn, win_fast)
    slow = spm_conv(log_xn, win_slow)
    lfp = fast - slow
    return lfp


def plot_spm_lfp(xn, factor_idx=0, win_fast=2, win_slow=16):
    import matplotlib.pyplot as plt
    lfp = compute_spm_lfp(xn, factor_idx, win_fast, win_slow)
    plt.figure(figsize=(6, 3))
    # Each column of lfp is one hidden state for the factor being plotted
    for state in range(lfp.shape[1]):
        plt.plot(lfp[:, state], label=f'State {state}')
    plt.title(f'SPM-style Synthetic LFP for Hidden State Factor {factor_idx}')
    plt.xlabel('VB Iteration')
    plt.ylabel('Bandpassed Log-Belief (LFP)')
    plt.legend()
    plt.tight_layout()
    plt.show()


def compute_spm_dopamine(gamma_history):
    gamma_history = np.asarray(gamma_history, dtype=np.float64)
    # Compute discrete first-order difference (gradient)
    dn = np.diff(gamma_history, prepend=gamma_history[0])
    # Optionally center (demean) for visualization
    dn_centered = dn - np.mean(dn)
    return dn_centered

def plot_spm_dopamine(gamma_history):
    import matplotlib.pyplot as plt
    dn = compute_spm_dopamine(gamma_history)
    plt.figure(figsize=(8, 3))
    plt.plot(dn, color='purple', lw=2)
    plt.title('SPM-style Dopaminergic Response (ΔGamma/Precision)')
    plt.xlabel('VB Iteration')
    plt.ylabel('ΔGamma (centered)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

####### EXPERIMENTAL ##########################################################
def update_preferences_dirichlet_m(pC_m, obs_m, lr=1.0):  
    """  
    JAX version of C matrix learning for a single modality.  
      
    Parameters  
    ----------  
    pC_m : Array  
        Dirichlet parameters for modality m  
    obs_m : Array  
        Observed outcome for modality m (one-hot or distributional)  
    lr : float  
        Learning rate  
          
    Returns  
    -------  
    new_pC_m : Array  
        Updated Dirichlet parameters  
    C_m : Array  
        Updated preference distribution (expected value)  
    """  
    # Dirichlet update: pC_m += lr * obs_m  
    new_pC_m = pC_m + lr * obs_m  
    C_m = dirichlet_expected_value(new_pC_m)  
      
    return new_pC_m, C_m  
  
def update_preferences_dirichlet(pC, C, obs, *, onehot_obs, num_obs, lr, modalities_to_learn=None):  
    """  
    JAX version of C matrix learning across all modalities.  
      
    Parameters  
    ----------  
    pC : list  
        Dirichlet parameters for each modality  
    C : list  
        Current preference distributions  
    obs : list  
        Observations for each modality  
    onehot_obs : bool  
        Whether observations are already one-hot encoded  
    num_obs : list  
        Number of outcomes per modality  
    lr : float  
        Learning rate  
    modalities_to_learn : list, optional  
        Indices of modalities to update (None = all)  
          
    Returns  
    -------  
    qC : list  
        Updated Dirichlet parameters  
    E_qC : list  
        Updated preference distributions  
    """  
    from jax import nn  
      
    obs_m = lambda o, dim: nn.one_hot(o, dim) if not onehot_obs else o  
      
    # Determine which modalities to learn  
    if modalities_to_learn is None:  
        modalities_to_learn = list(range(len(pC)))  
      
    update_C_fn = lambda pC_m, o_m, dim, m_idx: (  
        None if pC_m is None or m_idx not in modalities_to_learn   
        else update_preferences_dirichlet_m(pC_m, obs_m(o_m, dim), lr=lr)  
    )  
      
    result = tree_map(  
        update_C_fn,  
        pC, obs, num_obs, list(range(len(pC))),  
        is_leaf=lambda x: x is None  
    )  
      
    qC = []  
    E_qC = []  
    for i, r in enumerate(result):  
        if r is None:  
            qC.append(pC[i])  
            E_qC.append(C[i])  
        else:  
            qC.append(r[0])  
            E_qC.append(r[1])  
      
    return qC, E_qC

def update_policy_prior_dirichlet(pE, E, q_pi, lr=1.0):  
    """  
    JAX version of E matrix (policy prior/habits) learning.  
      
    Updates Dirichlet parameters over policy priors based on posterior   
    beliefs over policies, implementing habit learning as described in:  
    - Adams et al. 2022: Everything is connected  
    - Active Inference 2022 textbook: e = e + π  
      
    Parameters  
    ----------  
    pE : Array  
        Dirichlet parameters for policy prior (shape: num_policies)  
    E : Array  
        Current policy prior distribution (shape: num_policies)  
    q_pi : Array  
        Posterior beliefs over policies from current timestep  
    lr : float  
        Learning rate  
          
    Returns  
    -------  
    qE : Array  
        Updated Dirichlet parameters  
    E_qE : Array  
        Updated policy prior distribution (expected value)  
    """  
    # Dirichlet update: pE += lr * q_pi  
    qE = pE + lr * q_pi  
    E_qE = dirichlet_expected_value(qE)  
      
    return qE, E_qE

# # Integration with Agent (into agent infer_parameters method)
# if self.learn_E:  
#     lr = jnp.broadcast_to(lr_pE, (self.batch_size,))  
#     qE, E_qE = vmap(update_policy_prior_dirichlet)(  
#         self.pE,  
#         self.E,  
#         q_pi,  # from the most recent infer_policies call  
#         lr=lr  
#     )  
      
#     agent = tree_at(lambda x: (x.E, x.pE), agent, (E_qE, qE))

import ast

def safe_literal_eval(val):
    try:
        # Only evaluate strings that look like lists
        if isinstance(val, str) and val.startswith('[') and val.endswith(']'):
            return ast.literal_eval(val)
        else:
            # Return original value if not a valid list string
            return val
    except Exception:
        # Return original value for any parse error
        return val

# Andrew Pashea 2025
