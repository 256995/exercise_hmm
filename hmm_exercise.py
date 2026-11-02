import itertools
import math
import json


def log2(p):
    if p == 0:
        return -math.inf
    else:
        return math.log2(p)


def read_model(path):
    with open(path) as file:
        data = json.load(file)
    model = HMM(data['states'], data['alphabet'], data['pi'], data['a'], data['e'])
    return model


class HMM:
    def __init__(self, states, alphabet, pi, a, e):
        self.states = states
        self.alphabet = alphabet
        unique_states = set([l for l in states])
        unique_alphabet = set([l for l in alphabet])
        if (len(pi) == len(unique_states)) and (len(a) == len(unique_states)) and (len(e) == len(unique_states)):
            self.pi = pi
            if (len(row) == len(unique_states) for row in a):
                self.a = a
            if (len(row) == len(unique_alphabet) for row in e):
                self.e = e
        else:
            print('Incorrect length of pi, a or e.')
        # compute and store the log2 matrices
        self.log_pi = {}
        for key in self.pi:
            self.log_pi[key] = log2(self.pi[key])
        self.log_a = {state: {} for state in unique_states}
        for key in self.a:
            for k in self.a[key]:
                self.log_a[key][k] = log2(self.a[key][k])
        self.log_e = {state: {} for state in unique_states}
        for key in self.e:
            for k in self.e[key]:
                self.log_e[key][k] = log2(self.e[key][k])

    def joint_probability(self, x, z):
        result = self.pi[z[0]] * self.e[z[0]][x[0]]
        for i in range(1, len(x)):
            result = result * self.a[z[i-1]][z[i]] * self.e[z[i]][x[i]]
        return result

    def log_joint(self, x, z):
        result = self.log_pi[z[0]] + self.log_e[z[0]][x[0]]
        for i in range(1, len(x)):
            result = result + self.log_a[z[i-1]][z[i]] + self.log_e[z[i]][x[i]]
        return result

    def brute_force(self, x):
        paths = [''.join(path) for path in itertools.product(self.states, repeat=len(x))]
        best_path = ''
        best_prob = 0
        sum_joint = 0
        for path in paths:
            prob = self.joint_probability(x, path)
            sum_joint += prob
            if prob > best_prob:
                best_path = path
                best_prob = prob
        return best_path, best_prob, sum_joint

    def viterbi(self, x):
        v = {i: {} for i in range(len(x))}
        back = {i: {} for i in range(len(x))}
        for state in self.states:
            v[0][state] = self.log_pi[state] + self.log_e[state][x[0]]
            back[0][state] = ''
        for i in range(1, len(x)):
            for k in self.states:
                transition_list = [v[i-1][state] + self.log_a[state][k] for state in self.states]
                max_index = transition_list.index(max(transition_list))
                back[i][k] = self.states[max_index]
                v[i][k] = self.log_e[k][x[i]] + max(transition_list)



if __name__ == "__main__":
    print('--- TASK 1 ---')
    m = read_model("weather_model.json")
    m1 = read_model("toy_model.json")
    print(m1.a)
    print(m1.e)
    print(HMM.joint_probability(m1, 'phchc', 'SSSSS'))
    print(HMM.joint_probability(m, 'sr', 'HL'))
    print(HMM.log_joint(m, 'sr', 'HL'))
    print(HMM.brute_force(m, 'ssrssrrrss'))

    print('--- TASK 2 ---')
