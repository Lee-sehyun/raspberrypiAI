import numpy as np
import matplotlib.pyplot as plt
import gym
import random
import tensorflow as tf

def one_hot_encoder(state_in):
    return np.identity(16)[state_in:state_in + 1]                      # 해당 state값만 1인 16 크기 행렬 리턴

env = gym.make("FrozenLake-v0")
input_size = env.observation_space.n
output_size = env.action_space.n
learning_rate = 0.7

X = tf.placeholder(shape=[1, input_size], dtype=tf.float32)            # placeholder는 데이터 input이 들어가는 자리
W = tf.Variable(tf.random_uniform([input_size, output_size], 0, 0.01)) # Variable은 학습시킬 변수의 자리

Qpredict = tf.matmul(X, W)                                             # (1, input_size) X (input_size, output_size) = (1, output_size)
Y = tf.placeholder(shape=[1, output_size], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(Y-Qpredict)) # 오차 제곱의 합
train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss) # 경사하강법을 사용하여 오차를 최소화하는 방향으로 학습을 시키겠다

num_episodes = 1000
e = 0.2
r = 0.9
rList = []
successRate = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(num_episodes):
        state = env.reset() # 초기 state 설정
        total_reward = 0
        done = None
        local_loss = []
        while not done :
            Qs = sess.run(Qpredict, feed_dict={X:one_hot_encoder(state)}) # feed_dict은 placeholder에 실제 값을 넣기 위해 설정하는 옵션
            
            rand = random.random()
            if (rand < e / (i/50 + 10)):
                action = env.action_space.sample()
            else :
                action = np.argmax(Qs) # 예측값에서 가장 큰 기대값을 다음 action으로 선택
            
            new_state, reward, done, _ = env.step(action) # new_state: 다음 state
            if done:
                Qs[0, action] = reward
            else:
                new_Qs = sess.run(Qpredict, feed_dict={X:one_hot_encoder(new_state)})
                Qs[0:action] = reward + r * np.max(new_Qs)
            sess.run(train, feed_dict={X:one_hot_encoder(state), Y:Qs})
            total_reward += reward
            state = new_state
        rList.append(total_reward)
        successRate.append(sum(rList) / (i + 1))

print("Final DQN")
print(Qs)
print("Success Rate : ", successRate[-1])
plt.plot(range(len(rList)), rList)
plt.plot(range(len(successRate)), successRate)
plt.show()
