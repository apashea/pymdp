# {{card_advisor}}
# Define the logic for the CardAdvisor task

import numpy as np
import pandas as pd


class CardAdvisor(object):


  def __init__(self, trustworthy_process = None, correct_card = None, safety_belief=None):
    """
    Simulates an environment for a card-advice trust task.

    The CardAdvisor class models an environment where an agent must select the color of an initially unseen card, where
    the card can only be blue or green.
    The advisor provides advice to the agent to select between two card colors ("blue" or "green") prior to revealing the true color.
    The advisor can be either trustworthy (gives correct advice) or untrustworthy (gives incorrect advice) relative to the true color of the card.
    The agent receives feedback as to whether the card was correct or not and simulated arousal responses based on outcomes.

    Args:
        trustworthy_process (str, optional): Advisor's trustworthiness for the episode.
            - "trustworthy": Advisor gives correct advice.
            - "untrustworthy": Advisor gives incorrect advice.
            - If None, randomly chosen at initialization.
        correct_card (str, optional): The correct card for the episode.
            - "blue" or "green".
            - If None, randomly chosen at initialization.

    Attributes:
        correct_card_names (list): Possible correct card colors ("blue", "green").
        trustworthy_names (list): Possible advisor types ("trustworthy", "untrustworthy").
        correct_card (str): The correct card for the current trial.
        trustworthy_process (str): Advisor's trustworthiness for the current trial.
        choice_obs (list): Possible observed card choices.
        feedback_obs (list): Possible feedback outcomes.
        #stage_obs (list): Possible task stages.
        advice_obs (list): Possible advice options.
        arousal_obs (list): Possible arousal states ("high", "low").

    Public Methods:
        step(trust_action, card_action):
            Processes the agent's trust and card actions, returning a list of observations:
            [advice, feedback, arousal, choice].
    """
    # Set environment's true advisor trustworthiness and true correct card color
    self.correct_card_names = ["blue", "green"]
    self.trustworthy_names = ["trustworthy", "untrustworthy"]
    # Set possible agent actions
    self.choice_trust_actions_names = ['trust','distrust']
    self.choice_card_actions_names = ['blue','green','null','withdraw']

    if correct_card == None:
      # randomly sample correct card color (blue or green)
      self.correct_card = np.random.choice(self.correct_card_names, p=[0.5, 0.5])
    else:
      self.correct_card = correct_card

    if trustworthy_process == None:
      # randomly sample advisor trustworthiness (trustworthy, untrustworthy)
      self.trustworthy_process = np.random.choice(self.trustworthy_names, p=[0.5, 0.5])
    else:
      self.trustworthy_process = trustworthy_process

    self.choice_obs = ['blue', 'green','null','withdraw']
    self.feedback_obs = ['correct', 'incorrect', 'null','withdraw']     # 12FEB2026 edit from previous `['correct', 'incorrect', 'null']`
    #self.stage_obs = ['null', 'advice', 'decision']
    self.advice_obs = [ 'blue', 'green', 'null']
    self.arousal_obs = ['high', 'low']


  def step(self, trust_action,  card_action, agent_for_arousal_modulation=False, monitoring=False):
    """Process agent actions and return environment observations.
    The arguments pass the lower level agent's actions to the environment, advancing the stages of the trial.

    Args:
        trust_action (str): Trust-related action from agent. One of:
            - "trust": Trust advisor (outcome of 'low' arousal with probability 0.667, else returns 'high' arousal)
            - "distrust": Distrust advisor (outcome of 'high' arousal with probability 0.667, else returns 'low' arousal)
        card_action (str): Card selection action. One of:
            - "blue"/"green": Card choice (during final stage, outcome of 'correct' if card choice is equivalent to true card color, else 'incorrect')
            - "null": No selection (null feedback/choice)

    Returns:
        list: Observation components [advice, feedback, arousal, choice]:
            - advice (str): Advisor's recommendation ("blue"/"green")
            - feedback (str): "correct"/"incorrect" if card played, else "null"
            - arousal (str): "high"/"low" emotional response
            - choice (str): Matches card_action or "null"

    Observation Logic:
        - Advice reflects advisor's trustworthiness (truthful vs. reversed)
        - Feedback shows if card matched environment's correct card
        - Arousal probabilities depend on trust_action type
        - Choice directly mirrors card_action input
    """

    # observed_advice: uncontrollable by agent; advice matches true color if trustworthy else does not match)
    if self.trustworthy_process == "trustworthy":
          if self.correct_card == "blue":
              observed_advice = "blue"            # if trustworthy, then match correct card to observed advice
          elif self.correct_card == "green":
              observed_advice = "green"
    elif self.trustworthy_process == "untrustworthy":
          if self.correct_card == "blue":
              observed_advice = "green"
          elif self.correct_card == "green":
              observed_advice = "blue"

    # observed_arousal: dependent on agent's 'trust' action
    if trust_action == "trust":
      # if choose to trust, 2/3 probability of low arousal
      observed_arousal = np.random.choice(self.arousal_obs, p=[0.3333, 0.6667])
    elif trust_action == "distrust":
      # if choose to distrust, 2/3 probability of high arousal
      observed_arousal = np.random.choice(self.arousal_obs, p=[0.6667, 0.3333])
    elif trust_action == "null":
      # for initial observation generation at start of 'null' stage (independent of agent)
      observed_arousal = np.random.choice(self.arousal_obs, p=[0.5, 0.5])

    # observed_choice and observed_feedback: dependent on agent's choice (always 'null' in at end of initial 'null' stage; one of 'blue','green','withdraw' at end of 'advice' stage')
    if card_action == "null":
      observed_choice = "null"
      observed_feedback = "null"
    if card_action == "withdraw":
      observed_choice = "withdraw"
      observed_feedback = "withdraw"          # 12FEB2026 edit from `observed_feedback = "null"`
    if card_action == "blue":
      observed_choice = "blue"   # map observed choice to card choice (observe self)
      if self.correct_card == "blue":
          observed_feedback = "correct"
      elif self.correct_card == "green":
          observed_feedback = "incorrect"
    if card_action == "green":
      observed_choice = "green"
      if self.correct_card == "blue":
          observed_feedback = "incorrect"
      elif self.correct_card == "green":
          observed_feedback = "correct"

    # Interoceptive 'arousal' observation based on beliefs (Autonomic state gives rise to cardiac outcome, tachy/bradycardia)
    if agent_for_arousal_modulation != False:
      #true_trustworthiness_idx = self.trustworthy_names.index(self.trustworthy_process)   # get true trustworthy state from environment
      #true_correct_color_idx = self.correct_card_names.index(self.correct_card)           # get true correct card state from environment
      # get agent's affect belief state
      if monitoring==True:
        print(f"Extract max_affect_idx for arousal modulation from A matrix...")
      # Use the affective state (positive/negative) agent more strongly infers
      if hasattr(agent_for_arousal_modulation, 'qs_current_bma'):
        beliefs_set = agent_for_arousal_modulation.qs_current_bma    # if agent already has posterior beliefs, use posteror belief (from Bayesian model average)
      else:
        beliefs_set = agent_for_arousal_modulation.D                 # if first timestep of first trial, use prior over states
      # Get 'most probable state' via argmax per hidden state factor belief
      max_trustworthiness_belief_idx = np.argmax(beliefs_set[0])
      max_correct_color_belief_idx = np.argmax(beliefs_set[1])
      max_affect_belief_idx = np.argmax(beliefs_set[2])
      max_choice_belief_idx = np.argmax(beliefs_set[3])
      max_stage_belief_idx = np.argmax(beliefs_set[4])

      if monitoring==True:
        print(f"max_affect_idx = {max_affect_idx}")
      arousal_probs_array = agent_for_arousal_modulation.A[2][:,max_trustworthiness_belief_idx,max_correct_color_belief_idx,max_affect_belief_idx,max_choice_belief_idx,max_stage_belief_idx]

      if monitoring==True:
        print(f"arousal_probs_array = {arousal_probs_array}")
        print(f"arousal_probs_array.shape = {arousal_probs_array.shape}")
      observed_arousal = self.arousal_obs[np.argmax(arousal_probs_array)]

      # Arousal from A matrix in (Adams, 2022) ###############################################################
      # true_trustworthiness_idx = self.trustworthy_names.index(self.trustworthy_process)   # get true trustworthy state from environment
      # true_correct_color_idx = self.correct_card_names.index(self.correct_card)           # get true correct card state from environment
      # # get agent's affect belief state
      # if monitoring==True:
      #   print(f"Extract max_affect_idx for arousal modulation from A matrix...")
      # # Use the affective state (positive/negative) agent more strongly infers
      # if hasattr(agent_for_arousal_modulation, 'qs_current_bma'):
      #   max_affect_idx = np.argmax(agent_for_arousal_modulation.qs_current_bma[2])  # if agent already has posterior beliefs, use posteror belief (from Bayesian model average)
      # else:
      #   max_affect_idx = np.argmax(agent_for_arousal_modulation.D[2])    # if first timestep of first trial, use prior over states
      # if monitoring==True:
      #   print(f"max_affect_idx = {max_affect_idx}")
      # true_choice_idx = self.choice_obs.index(card_action)   # track card choice
      # if card_action == "null":
      #   if trust_action == "null":
      #     true_stage_idx = 0
      #   else:
      #     true_stage_idx = 1
      # elif card_action != "null":
      #   true_stage_idx = 2
      # # Get agent's likelihood belief about arousal, given true trustworthiness of advisor, true card color, believed affect, true choice, and true stage
      # # Note: This should be updated, it is unreal (the agent never has psychic access to 'true' trustworthiness/color/choice/stage, it only infers
      # #       and this won't capture false inference about these states)
      # arousal_probs_array = agent_for_arousal_modulation.A[2][:,true_trustworthiness_idx,true_correct_color_idx,max_affect_idx,true_choice_idx,true_stage_idx]  # Per (Adams et al, 2022)
      # if monitoring==True:
      #   print(f"arousal_probs_array = {arousal_probs_array}")
      #   print(f"arousal_probs_array.shape = {arousal_probs_array.shape}")
      # observed_arousal = self.arousal_obs[np.argmax(arousal_probs_array)]
      # ###########################################################################################

    obs = [observed_advice, observed_feedback, observed_arousal, observed_choice]

    #return obs, arousal_probs_array[0]

    if agent_for_arousal_modulation != False:
        return obs, arousal_probs_array[0]
    else:
        return obs


# Review environment rules
env_test = CardAdvisor(trustworthy_process='trustworthy', correct_card='blue')
def make_env_rules_df(
    CardAdvisor, 
    true_trustworthiness_list, 
    true_color_list, 
    choice_trust_actions, 
    choice_card_actions):
    records = []
    for true_color in true_color_list:
        for true_trustworthiness in true_trustworthiness_list:
            env_test = CardAdvisor(trustworthy_process=true_trustworthiness, correct_card=true_color)
            for card_action in choice_card_actions:
                for trust_action in choice_trust_actions:
                    obs_label = env_test.step(trust_action=trust_action, card_action=card_action)
                    record = {
                        'true_trustworthiness': true_trustworthiness,
                        'true_color': true_color,
                        'trust_action': trust_action,
                        'card_action': card_action,
                        'observed_advice': obs_label[0],
                        'observed_feedback': obs_label[1],
                        'observed_arousal': obs_label[2],
                        'observed_choice': obs_label[3]
                    }
                    records.append(record)
    df = pd.DataFrame.from_records(records)
    return df

# # Example usage of make_env_rules_df(): Print all possible combinations of environment dynamics
# env_rules_df = make_env_rules_df(
#     CardAdvisor,
#     true_trustworthiness_list=['trustworthy','untrustworthy'],
#     true_color_list=['blue','green'], 
#     choice_trust_actions=['trust','distrust'],
#     choice_card_actions=['blue','green','null','withdraw']
# )
# display(env_rules_df.sort_values('card_action'))
