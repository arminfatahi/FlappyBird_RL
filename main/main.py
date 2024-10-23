import utils
import flappy_bird_gym
import random
import time
import numpy as np


class SmartFlappyBird:
    def __init__(self, iterations):
        self.Qvalues = utils.Counter()
        self.landa = 1
        self.epsilon = 0.2  # change to proper value
        self.alpha = 0.9  # change to proper value
        self.iterations = iterations
        self.show_states = set()

        self.num_bins = 10
        self.discrete_states = [
            np.linspace(0, 1.7, num=(self.num_bins + 1))[1:-1],
            np.linspace(-0.4, 0.5, num=(self.num_bins + 1))[1:-1],
        ]

        self.num_actions = 2
        num_states = self.num_bins ** len(self.discrete_states)
        self.q = np.zeros(shape=(num_states, self.num_actions))

        print("Q-table shape (number_states, number_actions):", self.q.shape)

    def policy(self, state):
        # implement the policy
        #implemented in get_action
        return NotImplemented

    @staticmethod
    def get_all_actions():
        return [0, 1]

    def convert_continuous_to_discrete(self, state):
        discrete_state = sum(np.digitize(feature, self.discrete_states[i]) * (self.num_bins ** i) for i, feature in enumerate(state))
        self.show_states.add(discrete_state)
        return discrete_state

    def compute_reward(self, prev_info, new_info, done, observation):
        # implement the best way to compute reward base on observation and score
        if done:
            return -1000

        if observation[0]<0.3:
            if abs(observation[1])<0.07:
                return 5
            else:
                return -5

        if prev_info['score']<new_info['score']:
            return 20

        return 1

    def get_action(self, state):
        # implement the best way to get action base on current state
        # you can use `utils.flip_coin` and `random.choices`

        if (1-self.epsilon) <= np.random.uniform():
            # make a random action to explore
            next_action = np.random.randint(0, self.num_actions)

        else:
            # take the best action
            next_action = np.argmax(self.q[self.convert_continuous_to_discrete(state)])


        return next_action

    def maxQ(self, state):
        # return max Q value of a state
        # used max
        return NotImplemented

    def max_arg(self, state):
        # return argument of the max q of a state
        # used np.argmax
        return NotImplemented

    def update(self, reward, state, action, next_state):
        # update q table
        discrete_state = self.convert_continuous_to_discrete(state)

        discrete_next_state = self.convert_continuous_to_discrete(next_state)

        self.q[discrete_state, action] = (1-self.alpha) * self.q[discrete_state, action] + self.alpha * (reward + self.landa * np.max(self.q[discrete_next_state, :]))

    def update_epsilon_alpha(self):
        # update epsilon and alpha base on iterations

        self.epsilon *=(1-5e-3)

        self.alpha = max(1e-5, self.alpha * (1 - 2e-3))



    def run_with_policy(self, landa):
        self.landa = landa
        env = flappy_bird_gym.make("FlappyBird-v0")
        observation = env.reset()
        info = {'score': 0}
        for _ in range(self.iterations):
            print(_)
            observation = env.reset()
            done = False

            while not done:
                action = self.get_action(observation)  # policy affects here
                this_state = observation
                prev_info = info
                observation, reward, done, info = env.step(action)
                reward = self.compute_reward(prev_info, info, done, observation)
                self.update(reward, this_state, action, observation)

            self.update_epsilon_alpha()

        env.close()


    def run_with_no_policy(self, landa):
        self.landa = landa

        env = flappy_bird_gym.make("FlappyBird-v0")
        observation = env.reset()
        info = {'score': 0}
        done = False
        while not done:
            action = np.argmax(self.q[self.convert_continuous_to_discrete(observation)])

            observation, reward, done, info = env.step(action)

            env.render()
            time.sleep(1 / 30)  # FPS

        env.close()

    def run(self):
        self.run_with_policy(1)
        np.savetxt("qTable.csv", self.q, delimiter=',')
        self.run_with_no_policy(1)


program = SmartFlappyBird(iterations=1250)
program.run()


