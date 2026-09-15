import numpy as np
import pylab as plt

# Definisikan topologi graph 8-state (0 sampai 7), goal state = 7
points_list = [(0, 1), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 7)]
goal = 7
MATRIX_SIZE = 8

# Bentuk R Matrix
R = np.matrix(np.ones(shape=(MATRIX_SIZE, MATRIX_SIZE))) * -1
for point in points_list:
    if point[1] == goal:
        R[point] = 100
    else:
        R[point] = 0
    if point[0] == goal:
        R[point[::-1]] = 100
    else:
        R[point[::-1]] = 0
R[goal, goal] = 100

Q = np.matrix(np.zeros([MATRIX_SIZE, MATRIX_SIZE]))
gamma = 0.8

def available_actions(state):
    return np.where(R[state,] >= 0)[1]

def sample_next_action(act_range):
    return int(np.random.choice(act_range, 1))

def update(state, action):
    max_index = np.where(Q[action,] == np.max(Q[action,]))[1]
    if max_index.shape[0] > 1:
        max_index = int(np.random.choice(max_index, size=1))
    else:
        max_index = int(max_index)
    max_val = Q[action, max_index]
    Q[state, action] = R[state, action] + gamma * max_val

# Pelatihan
for i in range(1000):
    current_state = np.random.randint(0, MATRIX_SIZE)
    av_act = available_actions(current_state)
    action = sample_next_action(av_act)
    update(current_state, action)

# Pengujian rute dari state 0 menuju goal 7
curr = 0
path = [curr]
while curr != goal:
    next_step = int(np.argmax(Q[curr,]))
    path.append(next_step)
    curr = next_step

print("Jalur optimal dari 0 ke 7:", path)