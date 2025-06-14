
import math
import numpy as np
from scipy.stats import invgamma

# load the data and put it in a dictionary
allData = {}
with open('data.txt', 'r') as data:
  for line in data:
    vals = [float(x) for x in line.split()]
    allData[int(vals[0])] = (vals[1], vals[2])

# parameters on the prior for m
mu_zero_m = 5.0
sigma_zero_m = 10.0

# parameters on the prior for c 
mu_zero_c = 50.0
sigma_zero_c = 100.0

# parameters on the prior for sigma^2
alpha = 10.0
beta = 1.0

# initial estimates for the three model parameters
m = 20.0
c = 50.0
sigma = 200.0


# write this for 1a)
def SampleSigma():
    data = allData
    n = len(data)
    alpha_p = alpha + float(n)/2
    sum_var = 0
    for i, (h, w) in data.items():
        x_i = float(w - (h * m + c))
        sum_var = sum_var + (x_i**2)
    beta_p = beta + sum_var/2
    pos = invgamma.rvs(a=alpha_p, scale=beta_p)
    return math.sqrt(pos)

print("sigma")	
for x in range(10):
	print(SampleSigma())
print("\n")

# write this for 1b)
def SampleC ():
    data = allData
    n = float(len(data))
    mu_prior = mu_zero_c
    sigma_prior = sigma_zero_c
    sum_wd = 0
    for i, (h, w) in data.items():
        x_i = float(w - (h * m))
        sum_wd = sum_wd + x_i
    
    mu_pos = (1 / (1/sigma_prior**2 + n/sigma**2)) * (mu_prior/ sigma_prior**2 + sum_wd/ sigma**2)
    sigma_pos = math.sqrt((1/sigma_prior**2 + n/sigma**2)**(-1))
    return np.random.normal(mu_pos, sigma_pos)

print("c")	
for x in range(10):
	print(SampleC ())
print("\n")

# write this for 1c)
def SampleM ():
    data = allData
    mu_prior = mu_zero_m
    sigma_prior = sigma_zero_m
    sum_wd = 0
    d_i = 0
    n_i = 0
    for i, (h, w) in data.items():
        x_i = float((w - c)/h)
        sum_wd = sum_wd + x_i
        d_i = d_i + h**2/ sigma**2
        n_i = n_i + x_i * h**2
    mu_pos = (1 / (1/sigma_prior**2 + d_i)) * (mu_prior/ sigma_prior**2 + n_i/ sigma**2)
    sigma_pos = math.sqrt((1/sigma_prior**2 + d_i)**(-1))
    return np.random.normal(mu_pos, sigma_pos)

print("M")
for x in range(10):
	print(SampleM ())
print("\n")

# this computes the error of the current model
def getError ():
	error = 0.0
	count = 0
	for x in allData:
		y = allData[x]
		error += (c + y[0] * m - y[1]) * (c + y[0] * m - y[1])
		count += 1
	return error / count

# for part 2, you run 1000 iteratins of a Gibbs sampler
print("The initial values:")
print("m =", m)
print("c =", c)
print("sigma =",sigma)
print("\n")
error_vals = []
for xz in range(1000):
	error_vals.append(getError ());
	sigma = SampleSigma ();
	m = SampleM ();
	c = SampleC ();

print("The first 5 error values are", error_vals[:5])
print("\n")
print("The last 5 error values are", error_vals[-5:])
print("\n")
print("The final values:")
print("m =", m)
print("c =", c)
print("sigma =",sigma)

