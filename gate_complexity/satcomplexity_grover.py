# Computes the complexity of using Grover's algorithm to solve a k-SAT instance.
# Outputs T-depth and T-count.

import sys
from math import *

TOFFOLI_T_DEPTH = 1
TOFFOLI_ANCILLAS = 4
TOFFOLI_T_COUNT = 7
TOFFOLI_CLIFFORD_DEPTH = 6
TOFFOLI_CLIFFORD_COUNT = 16

# k-SAT thresholds
ratios = (4.267,9.931,21.117,43.37,87.79,176.54,354.01,708.92,1418.71,2838.28,5677.41,11355.67,22712.20)

# Logs T-depth and T-count
def LOG(x, p=""):
    if MULTI == 0:
        out = str(x[0]) + " " + str(x[1]) + " (  " + format_exp(x[1]) + "  )"
        if len(x) == 3:
            out = out + "\t " + str(x[2]) + " (  " + format_exp(x[2]) + "  )"
        print(p + ": " + out)
    return x

def format_exp(x):
    s = '{:.1E}'.format(x)
    if 'E+' in s:
        base, exponent = s.split('E+')
        exps = str(int(exponent)) # remove leading zeroes
    else:
        base, exponent = s.split('E-')
        exps = str(-int(exponent)) # remove leading zeroes
    return base + " \\times 10^{" + exps + "}"


# Compute k copies of a subroutine in parallel
# Returns T-depth, T-count
def compute_parallel_copies(k, params):
    return [params[0], params[1] * k]

# Compute 2 copies of a subroutine in parallel
# Returns T-depth, T-count
def compute_in_parallel(params1, params2):
    return [max(params1[0],params2[0]), params1[1] + params2[1]]

# Compute 2 copies of a subroutine serially
# Returns T-depth, T-count
def double(params):
    return [2*params[0], 2*params[1]]

# A Toffoli gate with "controls" control bits.
# Returns T-depth, T-count
def toffoli(controls):
    if controls <= 1:
        return [0,0]
    elif controls == 2:
        return [1,1]
    else:
        return [2*ceil(log(controls,2)-1)+1, controls - 1]

# Fans out 1 bit to n bits
# Returns T-depth, T-count
def copy(n, controls):
    gdepth, gcount = toffoli(controls + 1)
    # In the surface code, can implement CNOT gates with multiple targets at low cost.
    # So here just assume that the cost is the same as one CNOT gate.
    return [gdepth, gcount]
    #return [ceil(log(n,2)) * gdepth, (n-1)*gcount]

def grover(n, m, k):
    one_iteration = list(zip(
        [0,0], # Hadamards for diffusion op
        LOG(toffoli(n), "AND for diffusion op"),
        LOG(double(compute_parallel_copies(n, copy(m,0))), "fan out bits (and fan in)"),
        LOG(double(compute_parallel_copies(m, toffoli(k))), "check clauses"),
        LOG(toffoli(m), "AND of all clauses")
        ))
    #print("T-count of just oracle op: " + str(format_exp(sum(one_iteration[1]) - toffoli(n)[1] - 2)))
    return [sum(one_iteration[0]), 7*sum(one_iteration[1])]

    # 3.642 multiple allows for overhead for repetitions, as in quant-ph/9902049
    # return [3.642 * sum(one_iteration[0]), 3.642 * sum(one_iteration[1])]
    #return [sqrt(2**n)*sum(one_iteration[0]),sqrt(2**n)*sum(one_iteration[1])]

MULTI = 1

# if len(sys.argv) == 4:
#     MULTI = 1
#     k, nmin, nmax = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
# elif len(sys.argv) == 3:
#     MULTI = 0
#     k, n = int(sys.argv[1]), int(sys.argv[2])
# else:
#     print("Usage: " + sys.argv[0] + " k n_min [n_max]")
#     quit()

# if MULTI == 1:
#     for n in range(nmin, nmax+1):
#         m = ceil(ratios[k-3] * n)
#         tdepth, tcount = grover(n, m, k)
#         print(str(n) + "," + str(int(tdepth)) + "," + str(int(tcount)))
#     quit()

# m = ceil(ratios[k-3] * n)
# print("Using " + str(m) + " clauses")

# tdepth, tcount = grover(n, m, k)

# print("Overhead T-depth: " + str(tdepth/3.642) + " (" + "{:.2E}".format(tdepth/3.642) + ")")
# print("Overhead T-count: " + str(tcount/3.642) + " (" + "{:.2E}".format(tcount/3.642) + ")")
