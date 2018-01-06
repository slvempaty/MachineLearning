from __future__ import print_function
import json
import numpy as np
import sys

def forward(pi, A, B, O):
  """
  Forward algorithm

  Inputs:
  - pi: A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
  - A: A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence (in terms of index, not the actual symbol)

  Returns:
  - alpha: A numpy array alpha[j, t] = P(Z_t = s_j, x_1:x_t)
  """
  S = len(pi)
  N = len(O)
  alpha = np.zeros([S, N])
  ###################################################
  #Base case:
  
  for i in range(S):
   alpha[i,0] = pi[i] * B[i,0]
  
  
  for j in range(1,N):
   for i in range(S):
      summation=np.dot(A[:,i],alpha[:,j-1])
      alpha[i,j] = B[i,O[j]] * summation
      
  ###################################################
  
  return alpha


def backward(pi, A, B, O):
  """
  Backward algorithm

  Inputs:
  - pi: A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
  - A: A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence (in terms of index, not the actual symbol)

  Returns:
  - beta: A numpy array beta[j, t] = P(Z_t = s_j, x_t+1:x_T)
  """
  S = len(pi)
  N = len(O)
  beta = np.zeros([S, N])
  ###################################################
  #basecase:
  for i in range(S):
   beta[i,N-1] = 1
   
  for i in range(N-2,-1,-1):
   for j in range(S):
      summation=0
      for idx in range(S):
         summation+= beta[idx,i+1] * A[j,idx] * B[idx,O[i+1]]
      
      beta[j,i]=summation  
  
  ###################################################
  
  return beta

def seqprob_forward(alpha):
  """
  Total probability of observing the whole sequence using the forward algorithm

  Inputs:
  - alpha: A numpy array alpha[j, t] = P(Z_t = s_j, x_1:x_t)

  Returns:
  - prob: A float number of P(x_1:x_T)
  """
  prob = 0
  ###################################################
  # Q3.2 Edit here
  m,n=alpha.shape
  prob=alpha.sum(axis=0)[n-1]
  ###################################################
 # print(prob)
  return prob


def seqprob_backward(beta, pi, B, O):
  """
  Total probability of observing the whole sequence using the backward algorithm

  Inputs:
  - beta: A numpy array beta: A numpy array beta[j, t] = P(Z_t = s_j, x_t+1:x_T)
  - pi: A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence
      (in terms of the observation index, not the actual symbol)

  Returns:
  - prob: A float number of P(x_1:x_T)
  """
  prob = 0
  ###################################################
  for i in range(beta.shape[0]):
   prob+= beta[i,0] * pi [i] * B[i,O[0]]
  
  
  ###################################################
  
  return prob

def viterbi(pi, A, B, O):
  """
  Viterbi algorithm

  Inputs:
  - pi: A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
  - A: A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence (in terms of index, not the actual symbol)

  Returns:
  - path: A list of the most likely hidden state path k* (in terms of the state index)
    argmax_k P(s_k1:s_kT | x_1:x_T)
  """
  path = []
  ###################################################
  # Q3.3 Edit here
  delta=np.zeros((len(pi),len(O)))
  prevState=np.zeros(delta.shape)
  #base case:
  for s in range(len(pi)):
   delta[s,0] = B[s,O[0]] * pi[s]
   prevState[s,0]=-1
  
  
  for t in range(1,len(O)):
   for s in range(len(pi)):
      temp=np.zeros(len(pi))
      for prevs in range(len(pi)):
         temp[prevs] = delta[prevs,t-1]*A[prevs,s]*B[s,O[t]]
      delta[s,t]=np.amax(temp)
      prevState[s,t]=np.argmax(temp)
      
  #print (delta)
  #print (prevState)
  last=np.argmax(delta[:,len(O)-1])
  
  #print(last)
  path.append(last)
  
  for prev in range(prevState.shape[1]-1,0,-1):
  # print(prev)
  # print(last)
   last = int(prevState[last,prev])
   path.append(last)
  
  #print(path)  
  path=list(reversed(path))
  
  ###################################################
  
  return path


##### DO NOT MODIFY ANYTHING BELOW THIS ###################
def main():
  model_file = sys.argv[1]
  Osymbols = sys.argv[2]

  #### load data ####
  with open(model_file, 'r') as f:
    data = json.load(f)
  A = np.array(data['A'])
  B = np.array(data['B'])
  pi = np.array(data['pi'])
  #### observation symbols #####
  obs_symbols = data['observations']
  #### state symbols #####
  states_symbols = data['states']

  N = len(Osymbols)
  O = [obs_symbols[j] for j in Osymbols]

  alpha = forward(pi, A, B, O)
  beta = backward(pi, A, B, O)

  prob1 = seqprob_forward(alpha)
  prob2 = seqprob_backward(beta, pi, B, O)
  print('Total log probability of observing the sequence %s is %g, %g.' % (Osymbols, np.log(prob1), np.log(prob2)))

  viterbi_path = viterbi(pi, A, B, O)

  print('Viterbi best path is ')
  for j in viterbi_path:
    print(states_symbols[j], end=' ')

if __name__ == "__main__":
  main()
