import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

class rHN:

  def __init__(self, W, W0, topologymask, T, dt, lr=0.0001, hebbian=False):

    self.W = W

    self.topologymask = topologymask

    self.T = T

    self.lr = lr

    self.dt = dt

    self.hebbian = hebbian

    self.W0 = W0

    self.E = torch.zeros(1, T)
  
  def set_state(self, S):

    self.S = S
  
  def hebbian_on(self):

    self.hebbian = True
  
  def hebbian_off(self):

    self.hebbian = False

  def set_lr(self, lr):

    self.lr = lr
  
  def relax(self):

    for t in range(T):

      self.S = self.S + self.dt * -self.S + self.dt * torch.tanh(self.W @ self.S)

      self.E[0, t] = -((self.S.T @ self.W0) @ self.S)/2
    
    if self.hebbian:

      deltaW = self.S @ self.S.T

      self.W = self.W + deltaW * self.lr

      self.W = self.W * self.topologymask
    
    return self.E