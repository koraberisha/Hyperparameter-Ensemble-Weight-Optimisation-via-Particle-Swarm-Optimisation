from random import random
from random import uniform
from hewo import initialise_models
import numpy as np
import multiprocessing
from sko.PSO import PSO

def sphere_cost(x,total=0):
    for x_i in x:
        total+=x**2
    return total

class Particle:
    def __init__(self,upper_b,lower_b,dim):
        self.position_i = []  # particle position
        self.velocity_i = []  # particle velocity
        self.pos_best_i = []  # best position individual
        self.err_best_i = 0  # best error individual
        self.err_i = 0  # error individual
        self.n_dim=dim
        self.ub = np.array(upper_b)
        self.lb = np.array(lower_b)
        v_high = self.ub - self.lb
        self.velocity_i = np.random.uniform(low=-v_high, high=v_high, size=(self.n_dim,))
        self.position_i = np.random.uniform(low=self.lb, high=self.ub, size=(self.n_dim,))

    # evaluate current fitness
    def evaluate(self, costFunc):
        self.err_i = costFunc(self.position_i)
        if self.err_i > self.err_best_i:
            self.pos_best_i = self.position_i.copy()
            self.err_best_i = self.err_i

    def calculate_cognitive(self,c1,r1,pb,pi):
        return
    def update_velocity(self, pos_best_g):
        w = 0.8  # constant inertia weight (how much to weigh the previous velocity)
        c1 = 0.5  # cognative constant
        c2 = 0.5  # social constant

        for i in range(0, num_dimensions):

            r1 = np.random.rand()
            r2 = np.random.rand()

            vel_cognitive = c1 * r1 * (self.pos_best_i[i] - self.position_i[i])
            vel_social = c2 * r2 * (pos_best_g[i] - self.position_i[i])
            self.velocity_i[i] = w * self.velocity_i[i] + vel_cognitive + vel_social

    def update_position(self, bounds):
        for i in range(0, num_dimensions):
            self.position_i[i] = self.position_i[i] + self.velocity_i[i]
            if self.position_i[i] > bounds[i][1]:
                self.position_i[i] = bounds[i][1]
            if self.position_i[i] < bounds[i][0]:
                self.position_i[i] = bounds[i][0]


def maximise(costFunc, lower_b,upper_b, num_particles, maxiter, verbose=False):
    global num_dimensions
    num_dimensions = len(lower_b)
    err_best_g = 0  # best error for group
    pos_best_g = []  # best position for group
    swarm = []
    for i in range(0, num_particles):
        swarm.append(Particle(lower_b,upper_b,num_dimensions))
    bounds = list(zip(lower_b, upper_b))
    i = 0
    while i < maxiter:
        if verbose: print(err_best_g)

        for j in range(0, num_particles):
            swarm[j].evaluate(costFunc)

            if swarm[j].err_i > err_best_g:
                pos_best_g = list(swarm[j].position_i)
                err_best_g = float(swarm[j].err_i)

        for j in range(0, num_particles):
            swarm[j].update_velocity(pos_best_g)
            swarm[j].update_position(bounds)
        i += 1

    if verbose:
        print(pos_best_g,err_best_g)

    return err_best_g,

lower_b = [0.1, 0.1, 0.1, 0.1, 0.18,  #Vals 0 - 3 Control Ensemble Weights, Val 4 Controls Number of Features
               0.0, 0.0, 0.0, 0.0,   #vals 5-11 Control XGB Params
               0.0, 0.0, 0.0,                   #vals 12-14 Control KNN Params
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0,    #vals 15-20 Control RF Params
               0.0, 0.0, 0.0          #Vals 21-25
               ]

upper_b = [0.9, 0.9, 0.9, 0.9, 0.27,
           0.1, 0.3, 0.4, 0.3,
           4.8, 2.8, 0.1,
           0.7, 0.1, 0.9, 0.2, 0.2, 0.1,
           0.5, 0.1, 0.5
           ]



maximise(initialise_models, lower_b,upper_b, num_particles=21, maxiter=8, verbose=True)
