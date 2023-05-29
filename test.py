from model import ActorCritic
import torch
import gym
from PIL import Image
import numpy as np


from gym.envs.box2d import LunarLander
 
class LunarLanderModifiedDensity(LunarLander):
    def __init__(self, density_percentage=100):
        super().__init__()
 
        # Modify the lander fixture definition
        for fixture in self.lander.fixtures:
            fixture.density *= (density_percentage / 100.0)
 
        self.lander.ResetMassData()

 
class LunarLanderModifiedThrust(LunarLander):
    def __init__(self, main_engine_power_factor=1.0, side_engine_power_factor=1.0):
        self.main_engine_power_factor = main_engine_power_factor
        self.side_engine_power_factor = side_engine_power_factor
        super().__init__()

    def _create_lander(self):
        lander = super()._create_lander()

        # Modify main engine power and side engine power
        self.MAIN_ENGINE_POWER *= self.main_engine_power_factor
        self.SIDE_ENGINE_POWER *= self.side_engine_power_factor

        print("MAIN_ENGINE_POWER:", self.MAIN_ENGINE_POWER)
        print("SIDE_ENGINE_POWER:", self.SIDE_ENGINE_POWER)

        return lander

    def step(self, action):


        return super().step(action)
    
class LunarLanderModifiedInitial(LunarLander):
    def __init__(self, initial_state=None):
        super().__init__()
        self.initial_state = initial_state

    def reset(self):
        if self.initial_state is not None:
            self.state = self.initial_state.copy()
        else:
            self.state = super().reset()
        
        return self.state
    """ Works !!"""
class LunarLanderModifiedReward(LunarLander): 
    def __init__(self, reward_factor=1.0):
        self.reward_factor = reward_factor
        super().__init__()

    def step(self, action):
        obs, reward, done, info = super().step(action)
        
        # Modify the reward
        reward *= self.reward_factor

        return obs, reward, done, info
    """ Works !!"""
class LunarLanderWithWind(LunarLander):
    def __init__(self, wind_scale=1.0):
        self.wind_scale = wind_scale
        super().__init__()

    def step(self, action):
        # Apply wind
        wind = np.random.normal(scale=self.wind_scale)
        self.lander.ApplyForceToCenter((wind, wind), True)

        return super().step(action)
    """ Works !!"""
class LunarLanderWithNoise(LunarLander):
    def __init__(self, noise_scale=0.01):
        self.noise_scale = noise_scale
        super().__init__()

    def step(self, action):
        observation, reward, done, info = super().step(action)

        # Add noise to the observation
        noise = np.random.normal(scale=self.noise_scale, size=observation.shape)
        observation += noise

        return observation, reward, done, info
    
def testfunction(n_episodes=2, name='LunarLander_TWO.pth'):
    #env = gym.make('LunarLander-v2')
    """env = LunarLanderModifiedThrust(main_engine_power_factor=21.0, side_engine_power_factor=90.0)
    initial_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  
    env = LunarLanderModifiedReward(2.0)
    env = LunarLanderWithWind(wind_scale=200.0)"""
    env = LunarLanderWithNoise(noise_scale=10.0)

    
    policy = ActorCritic()
    
    policy.load_state_dict(torch.load('./preTrained/{}'.format(name)))
    
    render = True
    save_gif = False

    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        running_reward = 0
        for t in range(10000):
            action = policy(state)
            state, reward, done, _ = env.step(action)
            running_reward += reward
            if render:
                 env.render()
                 if save_gif:
                     img = env.render(mode = 'rgb_array')
                     img = Image.fromarray(img)
                     img.save('./gif/{}.jpg'.format(t))
            if done:
                break
        print('Episode {}\tReward: {}'.format(i_episode, running_reward))
    env.close()
            
if __name__ == '__main__':
    testfunction()
