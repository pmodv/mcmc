
# solving knapskac 0/1 problem with nearest-neighbor random-walk mcmc


import numpy as np

# let n = 6

n = 6

# values vector - used to evaluate our inner product w.r.t x at each iteration of the MCMC
v = [50,50,64,46,50,5]

# weight vector
w = [56,59,80,64,75,17]

# max weight
W = 190

# optimal solution: [1,1,0,0,1,0]

# to 
# choose random vector in x

x_start = (np.random.randint(2,size = n)).tolist()

# list to store results
l_results = []

# only increment count when an update occurs
for i in range(1,10000):
    #print(x_start)
    # uniformally choose random index of x to permute
    idx_x = np.random.randint(n,size=1).item()  

    x_swap = 1 - x_start[idx_x]
    
    x_post = x_start[idx_x+1:n]
    x_pre = x_start[0:idx_x]    
      
    x_new = x_pre + [x_swap] + x_post
    
    # flip that bit in x

    # evaluate inner product of prospective new state weights, x_new
    # use comprehension vs reduce to be 'pythonic'

    ip_new_weight = sum([x*y for x,y in zip((x_new),w)])
    #print(ip_new_weight)
    # if weight inner product > W, don't move
    if ip_new_weight > W:
        #print(ip_new_weight)
        continue
    
    #... if not, move to x_new with probability min{1, p(z')/p(z)}) contingent on ratio of probs
    # beta factor gives us annealing: just use log(time), for now

    
    beta = np.log(i)
    
    
    

    ip_current = sum([x*y for x,y in zip((x_start),v)])

    ip_new = sum([x*y for x,y in zip((x_new),v)])

    #print(ip_new,x_new,v)
    # for any 0/1 backpack state, we have same prob to go to any neighboring state
    # therefore, these probabilities cancel-out in ratio-prob, and we just need to score evaluation 
    ratio_prob = np.exp(beta*ip_new)/np.exp(beta*ip_current)
    
    alpha = min(1, ratio_prob)
    # test for u ~ U(0,1) < alpha 
    u = np.random.rand()

    # if u < alpha, keep proposed state
    if (u < alpha):
        #print('new!')
        x_start = x_new
        l_results.append(x_new)
        #print(x_new)
        
    # if u > alpha, don't change x_start

l_tup_results = []
for l in l_results:
     l_tup_results.append(tuple(l))

freq = {}

#print(l_results)

for e in l_tup_results:
    freq[e] = freq.get(e, 0) + 1

keyList = freq.keys()
for w in sorted(freq, key=freq.get,reverse=True):
    print(w,freq[w])

