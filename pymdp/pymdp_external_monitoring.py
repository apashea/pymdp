import numpy as np
import copy

########################################
# Minimal utils (from utils.py)
########################################

EPS_VAL = 1e-16

def obj_array(num_arr):
    return np.empty(num_arr, dtype=object)

def obj_array_zeros(shape_list):
    arr = obj_array(len(shape_list))
    for i, shape in enumerate(shape_list):
        arr[i] = np.zeros(shape)
    return arr

def obj_array_uniform(shape_list):
    arr = obj_array(len(shape_list))
    for i, shape in enumerate(shape_list):
        arr[i] = norm_dist(np.ones(shape))
    return arr

def is_obj_array(arr):
    return arr.dtype == "object"

def to_obj_array(arr):
    if is_obj_array(arr):
        return arr
    obj_array_out = obj_array(1)
    obj_array_out[0] = arr.squeeze()
    return obj_array_out

def norm_dist(dist):
    if dist.ndim == 3:
        new_dist = np.zeros_like(dist)
        for c in range(dist.shape[2]):
            new_dist[:, :, c] = np.divide(dist[:, :, c], dist[:, :, c].sum(axis=0))
        return new_dist
    else:
        return np.divide(dist, dist.sum(axis=0))

def get_model_dimensions(A=None, B=None):
    if A is None and B is None:
        raise ValueError("Must provide either `A` or `B`")
    if A is not None:
        num_obs = [a.shape[0] for a in A] if is_obj_array(A) else [A.shape[0]]
        num_modalities = len(num_obs)
    else:
        num_obs, num_modalities = None, None
    if B is not None:
        num_states = [b.shape[0] for b in B] if is_obj_array(B) else [B.shape[0]]
        num_factors = len(num_states)
    else:
        if A is not None:
            num_states = list(A[0].shape[1:]) if is_obj_array(A) else list(A.shape[1:])
            num_factors = len(num_states)
        else:
            num_states, num_factors = None, None
    return num_obs, num_states, num_modalities, num_factors

def process_observation(obs, num_modalities, num_observations):
    # This is a reduced version consistent with your use-case (discrete indices).
    # obs can be int (single modality) or tuple/list of int (multi-modality).
    if num_modalities == 1:
        if isinstance(obs, int):
            vec = np.zeros(num_observations[0])
            vec[obs] = 1.0
            return to_obj_array(vec)
        else:
            # already a vector or one-hot-like
            return to_obj_array(np.array(obs))
    else:
        # multi-modality: tuple/list of ints
        out = obj_array(num_modalities)
        if isinstance(obs, (list, tuple)):
            for m in range(num_modalities):
                if isinstance(obs[m], int):
                    vec = np.zeros(num_observations[m])
                    vec[obs[m]] = 1.0
                    out[m] = vec
                else:
                    out[m] = np.array(obs[m])
        else:
            # already an object array of one-hots
            return obs
        return out

def process_observation_seq(obs_seq, n_modalities, n_observations):
    proc_obs_seq = obj_array(len(obs_seq))
    for t, obs_t in enumerate(obs_seq):
        proc_obs_seq[t] = process_observation(obs_t, n_modalities, n_observations)
    return proc_obs_seq

########################################
# Minimal maths (from maths.py)
########################################

from itertools import chain
from scipy import special

def softmax(dist):
    output = dist - dist.max(axis=0)
    output = np.exp(output)
    output = output / np.sum(output, axis=0)
    return output

def spm_dot(X, x, dims_to_omit=None):
    if is_obj_array(x):
        dims = list(range(X.ndim - len(x), len(x) + X.ndim - len(x)))
    else:
        dims = [1]
        x = to_obj_array(x)
    if dims_to_omit is not None:
        arg_list = [X, list(range(X.ndim))] + list(
            chain(*([x[xdim_i], [dims[xdim_i]]] for xdim_i in range(len(x)) if xdim_i not in dims_to_omit))
        ) + [dims_to_omit]
    else:
        arg_list = [X, list(range(X.ndim))] + list(
            chain(*([x[xdim_i], [dims[xdim_i]]] for xdim_i in range(len(x))))
        ) + [[0]]
    Y = np.einsum(*arg_list)
    if np.prod(Y.shape) <= 1.0:
        Y = Y.item()
        Y = np.array([Y]).astype("float64")
    return Y

def spm_norm(A):
    A = A + EPS_VAL
    normed_A = np.divide(A, A.sum(axis=0))
    return normed_A

def spm_log_single(arr):
    return np.log(arr + EPS_VAL)

def compute_accuracy(log_likelihood, qs):
    ndims_ll, n_factors = log_likelihood.ndim, len(qs)
    dims = list(range(ndims_ll - n_factors, n_factors + ndims_ll - n_factors))
    arg_list = [log_likelihood, list(range(ndims_ll))] + list(
        chain(*([qs[xdim_i], [dims[xdim_i]]] for xdim_i in range(n_factors)))
    )
    return np.einsum(*arg_list)

def calc_free_energy(qs, prior, n_factors, likelihood=None):
    free_energy = 0
    for factor in range(n_factors):
        negH_qs = qs[factor].dot(np.log(qs[factor][:, np.newaxis] + 1e-16))
        xH_qp = -qs[factor].dot(prior[factor][:, np.newaxis])
        free_energy += negH_qs + xH_qp
    if likelihood is not None:
        free_energy -= compute_accuracy(likelihood, qs)
    return free_energy

def dot_likelihood(A, obs):
    s = np.ones(np.ndim(A), dtype=int)
    s[0] = obs.shape[0]
    X = A * obs.reshape(tuple(s))
    X = np.sum(X, axis=0, keepdims=True)
    LL = np.squeeze(X)
    if np.prod(LL.shape) <= 1.0:
        LL = LL.item()
        LL = np.array([LL]).astype("float64")
    return LL

def get_joint_likelihood(A, obs, num_states):
    if isinstance(num_states, int):
        num_states_local = [num_states]
    else:
        num_states_local = num_states
    A_local = to_obj_array(A) if not is_obj_array(A) else A
    obs_local = to_obj_array(obs) if not is_obj_array(obs) else obs
    ll = np.ones(tuple(num_states_local))
    for modality in range(len(A_local)):
        ll = ll * dot_likelihood(A_local[modality], obs_local[modality])
    return ll

def get_joint_likelihood_seq(A, obs, num_states):
    ll_seq = obj_array(len(obs))
    for t, obs_t in enumerate(obs):
        ll_seq[t] = get_joint_likelihood(A, obs_t, num_states)
    return ll_seq

########################################
# Standalone _run_mmp_testing (from mmp.py)
########################################

def _run_mmp_testing(
    lh_seq,
    B,
    policy,
    prev_actions=None,
    prior=None,
    num_iter=10,
    grad_descent=True,
    tau=0.25,
    last_timestep=False
):
    """
    Marginal message passing scheme for updating marginal posterior beliefs about hidden states over time,
    conditioned on a particular policy. Test version that also records xn and vn per iteration.
    """

    # window lengths
    past_len = len(lh_seq)
    future_len = policy.shape[0]

    if last_timestep:
        infer_len = past_len + future_len - 1
    else:
        infer_len = past_len + future_len

    future_cutoff = past_len + future_len - 2

    # dimensions
    _, num_states, _, num_factors = get_model_dimensions(A=None, B=B)

    # beliefs: qs_seq[t][f]
    qs_seq = obj_array(infer_len)
    for t in range(infer_len):
        qs_seq[t] = obj_array_uniform(num_states)

    # last message
    qs_T = obj_array_zeros(num_states)

    # prior
    if prior is None:
        prior = obj_array_uniform(num_states)

    # transposed transition
    trans_B = obj_array(num_factors)
    for f in range(num_factors):
        trans_B[f] = spm_norm(np.swapaxes(B[f], 0, 1))

    if prev_actions is not None:
        policy = np.vstack((prev_actions, policy))

    # storage for xn and vn: [itr][f] arrays of shape (num_states_f, infer_len)
    xn = obj_array(num_iter)
    vn = obj_array(num_iter)
    for itr in range(num_iter):
        xn[itr] = obj_array(num_factors)
        vn[itr] = obj_array(num_factors)
        for f in range(num_factors):
            xn[itr][f] = np.zeros((num_states[f], infer_len))
            vn[itr][f] = np.zeros((num_states[f], infer_len))

    # main VB loop
    for itr in range(num_iter):
        F = 0.0
        for t in range(infer_len):
            for f in range(num_factors):
                # likelihood term
                if t < past_len:
                    lnA = spm_log_single(spm_dot(lh_seq[t], qs_seq[t], [f]))
                else:
                    lnA = np.zeros(num_states[f])

                # past message
                if t == 0:
                    lnB_past = spm_log_single(prior[f])
                else:
                    action_idx = int(policy[t - 1, f])
                    past_msg = B[f][:, :, action_idx].dot(qs_seq[t - 1][f])
                    lnB_past = spm_log_single(past_msg)

                # future message
                if t >= future_cutoff:
                    lnB_future = qs_T[f]
                else:
                    future_msg = trans_B[f][:, :, int(policy[t, f])].dot(qs_seq[t + 1][f])
                    lnB_future = spm_log_single(future_msg)

                # inference step
                if grad_descent:
                    sx = qs_seq[t][f].copy()
                    lnqs = spm_log_single(sx)
                    coeff = 1 if (t >= future_cutoff) else 2
                    err = (coeff * lnA + lnB_past + lnB_future) - coeff * lnqs
                    lnqs = lnqs + tau * (err - err.mean())
                    qs_seq[t][f] = softmax(lnqs)

                    # free energy accumulation (same structure as mmp.py)
                    if (t == 0) or (t == (infer_len - 1)):
                        F += sx.dot(0.5 * err)
                    else:
                        F += sx.dot(0.5 * (err - (num_factors - 1) * lnA / num_factors))

                    # record xn and vn
                    xn[itr][f][:, t] = qs_seq[t][f]
                    vn[itr][f][:, t] = err
                else:
                    qs_seq[t][f] = softmax(lnA + lnB_past + lnB_future)
                    # if not grad_descent, xn/vn tracking could be added similarly

        if not grad_descent:
            # in the non‑gradient version, F is computed from calc_free_energy
            for t in range(infer_len):
                if t < past_len:
                    F += calc_free_energy(qs_seq[t], prior, num_factors, likelihood=spm_log_single(lh_seq[t]))
                else:
                    F += calc_free_energy(qs_seq[t], prior, num_factors)

    return qs_seq, F, xn, vn

########################################
# Standalone _update_posterior_states_full_test (from inference.py)
########################################

def _update_posterior_states_full_test(
    A,
    B,
    prev_obs,
    policies,
    prev_actions=None,
    prior=None,
    policy_sep_prior=True,
    **kwargs,
):
    """
    Update posterior over hidden states using marginal message passing (TEST VERSION),
    returning qs_seq_pi, F, xn_seq_pi, vn_seq_pi for each policy.
    """

    num_obs, num_states, num_modalities, num_factors = get_model_dimensions(A, B)

    prev_obs = process_observation_seq(prev_obs, num_modalities, num_obs)

    lh_seq = get_joint_likelihood_seq(A, prev_obs, num_states)

    if prev_actions is not None:
        prev_actions = np.stack(prev_actions, 0)

    qs_seq_pi = obj_array(len(policies))
    xn_seq_pi = obj_array(len(policies))
    vn_seq_pi = obj_array(len(policies))
    F = np.zeros(len(policies))

    for p_idx, policy in enumerate(policies):
        qs_seq_pi[p_idx], F[p_idx], xn_seq_pi[p_idx], vn_seq_pi[p_idx] = _run_mmp_testing(
            lh_seq,
            B,
            policy,
            prev_actions=prev_actions,
            prior=prior[p_idx] if policy_sep_prior else prior,
            **kwargs
        )

    return qs_seq_pi, F, xn_seq_pi, vn_seq_pi

########################################
# Main entry point: infer_states_info_monitoring
########################################

def infer_states_info_monitoring(agent, observation, distr_obs=False):
    """
    Standalone monitoring version of infer_states_info.

    For inference_algo == "MMP", it:
    - updates agent.prev_obs / prev_actions window,
    - runs a full MMP test update over all policies,
    - stores agent.F and agent.qs,
    - returns (qs, xn, vn) for monitoring.
    """

    obs = tuple(observation) if not distr_obs else observation

    if not hasattr(agent, "qs"):
        agent.reset()

    # VANILLA path is included for completeness; you mainly care about MMP.
    if agent.inference_algo == "VANILLA":
        if agent.action is not None:
            # Here you could either inline get_expected_states or call control.get_expected_states.
            # For strict standalone, we'd need a minimal reimplementation; for now, use agent.D when no action.
            empirical_prior = agent.D
        else:
            empirical_prior = agent.D

        # A minimal fixed‑point update could be inlined, but since your target is MMP,
        # we return the current qs unchanged for VANILLA.
        qs = agent.qs if hasattr(agent, "qs") else empirical_prior
        xn, vn = None, None
        agent.qs = qs
        return qs, xn, vn

    # MMP path (main case)
    agent.prev_obs.append(obs)
    if len(agent.prev_obs) > agent.inference_horizon:
        latest_obs = agent.prev_obs[-agent.inference_horizon:]
        latest_actions = agent.prev_actions[-(agent.inference_horizon - 1):]
    else:
        latest_obs = agent.prev_obs
        latest_actions = agent.prev_actions

    qs, F, xn, vn = _update_posterior_states_full_test(
        agent.A,
        agent.B,
        latest_obs,
        agent.policies,
        latest_actions,
        prior=agent.latest_belief,
        policy_sep_prior=agent.edge_handling_params['policy_sep_prior'],
        **agent.inference_params
    )

    agent.F = F
    if hasattr(agent, "qs_hist"):
        agent.qs_hist.append(qs)
    agent.qs = qs

    return qs, xn, vn
