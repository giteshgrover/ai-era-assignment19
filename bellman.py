import numpy as np

class State:
    def __init__(self, row_index, col_index):
        self.row_index = row_index
        self.col_index = col_index

class Action:
    def __init__(self, row_step, col_step):
        self.row_step = row_step
        self.col_step = col_step


class Environment:
    def __init__(self, reward):
       self.reward = reward
       self.m, self.n = reward.shape

    def get_next_allowed_actions(self, state):
        allowed_actions = []
        max_allowed_actions = [Action(1, 0), 
                             Action(-1, 0), 
                             Action(0, 1), 
                             Action(0, -1)]
        for a in max_allowed_actions:
            if state.row_index + a.row_step >= 0 and state.row_index + a.row_step < self.m and state.col_index + a.col_step >= 0 and state.col_index + a.col_step < self.n:
                allowed_actions.append(a)
        
        return allowed_actions
    
    def get_landing_reward(self, state):
        return self.reward[state.row_index, state.col_index]
    
    def get_action_reward(self, state, action):
        return -1 # Every action has a cost of -1 no matter what the action is
    
    def get_reward(self, current_state, action, next_state):
        return self.get_landing_reward(next_state) + self.get_action_reward(current_state, action)
    
    def is_terminal_state(self, state):
        return state.row_index == self.m - 1 and state.col_index == self.n - 1
    
    def get_all_states(self):
        return [State(i, j) for i in range(self.m) for j in range(self.n)]
    
def get_next_step(self, state, allowed_actions):
    return np.random.choice(allowed_actions)
    # max_reward = -float('inf')
    # for action in allowed_actions:
    #     next_step = State(state.row_index + action.row_step, state.col_index + action.col_step)
    #     landing_reward = self.get_landing_reward(next_step)
    #     action_reward = self.get_action_reward(state, action)

    #     total_reward = landing_reward + action_reward
    #     if total_reward > max_reward:
    #         max_reward = total_reward
    #         next_step_state = next_step
    #         next_action = action


    # return next_step_state, next_action, max_reward
# def back_propagate_values(episode, values, environment, gamma):
#     values_new = values.copy()
#     for i in range(len(episode) - 1, -1, -1): # reverse order iteration
#         current_state = episode[i]["current_state"]
#         next_state = episode[i]["next_state"]
#         action = episode[i]["action"]
        
#         next_state_value = values[next_state.row_index, next_state.col_index]
#         reward = environment.get_reward(current_state, action, next_state)

          
#         values_new[current_state.row_index, current_state.col_index] = reward + gamma * next_state_value

#     return values_new

def train():
    reward = np.zeros((4, 4))
    reward[3, 3] = 0
    environment = Environment(reward)

    gamma = 1.0 # no discounting
    theta_threshold = 1e-4 # Stop when maximum change in V(s) across all states is < 1e - 4.
    start_state = State(0, 0)

    m, n = reward.shape
    values = np.zeros((m, n))
    while True:
        values_new = values.copy()
        for state in environment.get_all_states():
            if not environment.is_terminal_state(state):
                allowed_actions = environment.get_next_allowed_actions(state)
                max_reward = -float('inf')

                # As all the actions have the same probability, we can just calculate it once and then pick the best reward
                value = 0
                for action in allowed_actions:
                    next_state = State(state.row_index + action.row_step, state.col_index + action.col_step)
                    next_state_value = 1 / len(allowed_actions) * ( gamma * values[next_state.row_index, next_state.col_index]) # calculate the expected value of the next state with probability 1/len(allowed_actions)
                    value += next_state_value
                    reward = environment.get_reward(state, action, next_state)
                    if reward > max_reward:
                        max_reward = reward
                        # next_state_action = action
                        # next_state_state = next_state
                values_new[state.row_index, state.col_index] = value + max_reward
        
        if np.max(np.abs(values - values_new)) < theta_threshold:
            values = values_new
            print(values)
            break
        values = values_new
        

    # while True:
    #     episode = []
    #     state = start_state
    #     is_terminal_reached = False
    #     while not is_terminal_reached:
    #         allowed_actions = environment.get_next_allowed_actions(state)
    #         next_action = np.random.choice(allowed_actions)
    #         # print(next_action)
    #         next_state = State(state.row_index + next_action.row_step, state.col_index + next_action.col_step)
    #         episode.append({"current_state": state, "next_state": next_state, "action": next_action})
    #         state = next_state
    #         if environment.is_terminal_state(state):
    #             print("Terminal state")
    #             is_terminal_reached = True
        
    #     values_new = back_propagate_values(episode, values, environment, gamma)
    #     if np.max(np.abs(values - values_new)) > theta_threshold:
    #         values = values_new
    #         print(values)
    #     else:
    #         print("Converged")
    #         print(values)
    #         break

    

train()