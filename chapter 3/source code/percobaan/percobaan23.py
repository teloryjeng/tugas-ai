# Example 3.23 OpenAI Gymnasium CartPole 
import gymnasium as gym

# Gunakan CartPole-v1 dan tambahkan render_mode
env = gym.make('CartPole-v1', render_mode='human')

for i_episode in range(20):
    observation, info = env.reset()
    for t in range(100):
        print(observation)
        action = env.action_space.sample()
        
        # Gymnasium mengembalikan 5 nilai (terminated & truncated)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
            
env.close()
