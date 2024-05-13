from agents.DQN import DQN
from agents.DoubleDQN import DDQN
from agents.PolicyGradient import PolicyGradient
from agents.ActorCritic import ActorCritic
import gym
import matplotlib.pyplot as plt


def train_DQN():
    print("Training Start!")

    env = gym.make("CartPole-v1", render_mode="rgb_array")
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = DQN(n_states = n_states,
                n_actions = n_actions,
                n_hidden = 128,
                memory_size = 1000,
                sample_size=32,
                learning_rate = 1e-3,
                discount_rate = 0.9,
                target_update_interval=100
                )
    
    epsilon = 0.9
    agent_update_interval = 100
    max_episode = 5000
    step = 0
    return_record = []
    for episode in range(max_episode):
        epsilon = episode/max_episode
        state = env.reset()[0]       
        episode_return = 0
        done = False

        while not done:
            if episode > 9996:
                arr=env.render()
                plt.imshow(arr)
                plt.show(block=False)
                plt.pause(0.1)
            action = agent.choose_action(state, epsilon)
            next_state, reward, done, _, _ = env.step(action)
            if done:
                reward = 0
            state = next_state
            episode_return += reward
            agent.store_transition(state, action, reward, next_state)

            if step > agent.memory_size and step % agent_update_interval == 0:
                agent.update()

            step += 1            

        return_record.append(episode_return)
        print("Step: {}, Episode: {}, Return: {}".format(step, episode, episode_return))

    env.close()
    print("Training End!")

    plt.close()
    plt.plot(return_record)
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('DQN on CartPole-v1')
    plt.show()

    agent.plot_loss()


def train_DDQN():
    print("Training Start!")

    env = gym.make("CartPole-v1", render_mode="rgb_array")
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = DDQN(n_states = n_states,
                n_actions = n_actions,
                n_hidden = 128,
                memory_size = 1000,
                sample_size=32,
                learning_rate = 1e-3,
                discount_rate = 0.9,
                target_update_interval=100
                )
    
    epsilon = 0.9
    agent_update_interval = 100
    max_episode = 5000
    step = 0
    return_record = []
    for episode in range(max_episode):
        epsilon = episode/max_episode
        state = env.reset()[0]       
        episode_return = 0
        done = False

        while not done:
            if episode > 99996:
                arr=env.render()
                plt.imshow(arr)
                plt.show(block=False)
                plt.pause(0.1)
            action = agent.choose_action(state, epsilon)
            next_state, reward, done, _, _ = env.step(action)
            if done:
                reward = 0
            state = next_state
            episode_return += reward
            agent.store_transition(state, action, reward, next_state)

            if step > agent.memory_size and step % agent_update_interval == 0:
                agent.update()

            step += 1            

        return_record.append(episode_return)
        print("Step: {}, Episode: {}, Return: {}".format(step, episode, episode_return))

    env.close()
    print("Training End!")

    
    plt.close()
    plt.plot(return_record)
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('DDQN on CartPole-v1')
    plt.show()

    agent.plot_loss()


def train_PolicyGradient():
    print("Training Start!")

    env = gym.make("CartPole-v1", render_mode="rgb_array")
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = PolicyGradient(n_states = n_states,
                n_actions = n_actions,
                n_hidden = 16,
                learning_rate = 2e-3,
                discount_rate = 0.9,
                )
    
    max_episode = 2000
    return_record = []

    for episode in range(max_episode):
        state = env.reset()[0]       
        episode_return = 0
        agent.memory = []
        done = False

        while not done:
            if episode > 1998:
                arr=env.render()
                plt.imshow(arr)
                plt.show(block=False)
                plt.pause(0.1)
            action = agent.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state)
            state = next_state
            episode_return += reward
            

        agent.update()        

        return_record.append(episode_return)
        print("Episode: {}, Return: {}".format(episode, episode_return))

    env.close()
    print("Training End!")

    
    plt.close()
    plt.plot(return_record)
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('PolicyGradient on CartPole-v1')
    plt.show()

    agent.plot_loss()


def train_ActorCritic():
    print("Training Start!")

    env = gym.make("CartPole-v1", render_mode="rgb_array")
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = ActorCritic(n_states = n_states,
                n_actions = n_actions,
                n_hidden = 16,
                actor_learning_rate = 1e-3,
                critic_learning_rate = 1e-2,
                discount_rate = 0.9,
                )
    
    max_episode = 2000
    return_record = []

    for episode in range(max_episode):
        state = env.reset()[0]       
        episode_return = 0
        agent.memory = []
        done = False


        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)

            agent.store_transition(state, action, reward, done, next_state)

            state = next_state
            episode_return += reward
            
        agent.update()       

        return_record.append(episode_return)
        print("Episode: {}, Return: {}".format(episode, episode_return))

    env.close()
    print("Training End!")

    
    plt.close()
    plt.plot(return_record)
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('ActorCritic on CartPole-v1')
    plt.show()

    agent.plot_loss()


if __name__ == '__main__':
    #启动ActorCritic算法训练
    train_ActorCritic()
