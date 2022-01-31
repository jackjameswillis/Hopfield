import numpy as np
import matplotlib.pyplot as plt
import cv2

def E(W, V):

  energy = W*(V*V.T)

  for i in range(len(energy)):

      energy[i,i] = 0

  return [-np.sum(energy), -np.sum(energy, axis=0)]

def ESLOW(W, V):

    energy = []

    for i in range(len(V[0])):

        e = 0

        for j in range(len(V[0])):

            if i != j: e += W[i,j] * V[0,i] * V[0,j]

        energy.append(e)

    energy = np.array(energy)

    return [-np.sum(energy), energy]

def async_random(W, V):

    i = np.random.randint(0, len(V[0]))

    s = W[i,:] * V[0]

    s[i] = 0

    V[0,i] = np.sign(np.sum(s))

    return V

def energy_threshold(W, V):

    node_energy = ESLOW(W, V)[1]

    thresholds = (node_energy - np.min(node_energy))/np.ptp(node_energy)

    threshold = np.random.uniform(0, 1)

    active = thresholds >= threshold

    new_V = []

    for i in range(len(active)):

        if active[i]:

            s = W[i,:] * V[0]

            s[i] = 0

            new_V.append(np.sign(np.sum(s)))

        else:

            new_V.append(V[0][i])

    return np.array([new_V])

def energy_threshold_learning(W, V):

    node_energy = ESLOW(W, V)[1]

    thresholds = (node_energy - np.min(node_energy))/np.ptp(node_energy)

    threshold = np.random.uniform(0, 1)

    active = thresholds >= threshold

    new_V = []

    new_W = []

    for i in range(len(active)):

        if active[i]:

            s = W[i,:] * V[0]

            s[i] = 0

            v = np.sign(np.sum(s))

            w = W[i] + r * V[0, i] * V[0]

            w[i] = W[i, i]

            w = np.maximum(-1, np.minimum(1, w))

            new_V.append(v)

            new_W.append(w)

        else:

            new_V.append(V[0][i])

            new_W.append(W[i])

    return np.array(new_W), np.array([new_V])

def async_random_learning(W, V):

    i = np.random.randint(0, len(V[0]))

    s = W[i, :] * V[0]

    s[i] = 0

    V[0, i] = np.sign(np.sum(s))

    w = W[i] + r*V[0,i]*V[0]

    w[i] = W[i,i]

    w = np.maximum(-1, np.minimum(1, w))

    W[i] = w

    return W, V

N = 20

r = 0.1

V = np.random.choice((-1, 1), (1, N))

energy = []

spacetimes = []

for t in range(5):

    #W = np.random.choice((-0.1, 0.1), (N, N))

    W = np.random.uniform(-1, 1, (N,N))

    W = (W + W.T) / 2

    for i in range(N):

        W[i, i] = 1

    e = [ESLOW(W, V)[0]]

    spacetime = [V[0]]

    for iter in range(1000):

        #V = async_random(W, V)

        W, V = async_random_learning(W, V)

        spacetime.append(V[0])

        e.append(ESLOW(W, V)[0])

    spacetimes.append(spacetime)

    energy.append(e)


for e, spacetime in zip(energy, spacetimes):

    #spacetime = np.array(spacetime)

    #spacetime = (spacetime - np.min(spacetime))/(np.max(spacetime) - np.min(spacetime))

    #cv2.imshow(f"{e}", np.array(spacetime))

    plt.plot(e)

plt.ylabel('energy')

plt.xlabel('state updates')

plt.show()

'''
cv2.waitKey(0)

cv2.destroyAllWindows()

'''