# Computes the complexity of using the quantum backtracking algorithm algorithm to solve a random k-SAT instance.
# Outputs T-depth and T-count.
# Parameters specify k, a minimum value of n, potentially a maximum, and the number of control bits
# (which should be 1 for the standard algorithm).

import sys
from math import *

# k-SAT thresholds
ratios = (4.267,9.931,21.117,43.37,87.79,176.541,354.01,708.92,1418.71,2838.28,5677.41,11355.67,22712.20)

# Constants giving the scaling of the classical backtracking algorithm when applied to random k-SAT.
# Assumes scaling is of the form 2^(a n + b); first list gives a values, second list b values.
scaling_exponent = (0.350428587462, 0.458089945604, 0.539532169465, 0.598187369729, 0.641543863063, 0.675948520566, 0.702053421923, 0.723147609441, 0.738435871761, 0.752189508291, 0.761170941953, 0.780097731419, 0.790757975992)
scaling_const = (3.69923514128, 3.65486920732, 3.52207968481, 3.45818175001, 3.44645157012, 3.43422736973, 3.46039532801, 3.52248928932, 3.6101779279, 3.68497545708, 3.80869097272, 3.71996880138, 3.7644399149)

def LOGL(x, p=""):
 #   out = ""
 #   for i in range(len(x)):
 #       out += str(x[i]) + " "
 #   print(p + ": " + out)
    return x

def LOGT(x, p=""):
    if MULTI == 0:
        out = str(x[0]) + " " + str(x[1]) + " (  " + format_exp(x[1]) + "  )"
        if len(x) == 3:
            out = out + "\t " + str(x[2]) + " (  " + format_exp(x[2]) + "  )"
        print(p + ": " + out)
    return x

def LOGD(x, p=""):
    return x

def LOG(x, p=""):
    #print(p + ": " + x)
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

# Expands a pair [T-depth, Toff-count] to [T-depth, Toff-count, T-count]
def toff_expand(x):
    return [x[0], x[1], 0]

# Expands a pair [T-depth, T-count] to [T-depth, Toff-count, T-count]
def t_expand(x):
    return [x[0], 0, x[1]]

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
# Returns Toff-depth, Toff-count
def toffoli(controls):
    if controls <= 1:
        return [0,0]
    elif controls == 2:
        return [1,1]
    else:
        return [2*ceil(log(controls,2)-1)+1, 2 * controls - 3]

# Fans out 1 bit to n bits
# Returns T-depth, T-count
def copy(n, controls):
    gdepth, gcount = toffoli(controls + 1)
    # In the surface code, can implement CNOT gates with multiple targets at low cost.
    # So here just assume that the cost is the same as one CNOT gate.
    return [gdepth, gcount]
    #return [ceil(log(n,2)) * gdepth, (n-1)*gcount]

# Converts a string (i1,v1),...(in,vn) to a string x.
# Returns T-depth, T-count
def convert(n,controls):
    r = 2
    s = int(ceil(log(n+1,2)))

    toffcost = toffoli(controls + 1)
    swapcost = [3*toffoli(controls + 2)[0],3*toffoli(controls + 2)[1]]

    outputs = list(zip(
        LOGD(double(compute_parallel_copies(n * (r + s), copy(n,controls))), "fan-out for conversion (and fan-in)"),
        LOGD(double(compute_parallel_copies(n * n / 2, toffoli(controls + s))), "check whether l=p (and uncompute)"), # Division by 2 is because check isn't necessary for even p
        LOGD(double(compute_parallel_copies(n * n * r, toffoli(controls + s + 2))), "copy v_p (and uncompute)"),
        LOGD(double(compute_parallel_copies(n * n,
            compute_in_parallel(compute_parallel_copies(r, swapcost),
                compute_parallel_copies(s, swapcost)))), "swap ops (and uncompute)"),
        LOGD([4*int(ceil(log(n,2)))*toffcost[0], 4*n*(n-1)*r*toffcost[1]], "sum"),
        ))
    return [sum(outputs[0]),sum(outputs[1])]

# Determines which of two 7-bit number is greater, a la Draper et al
# Restores the numbers and ancillas to their original values
# Returns T-depth, T-count
def gtr(controls):
    toff_gdepth, toff_gcount = toffoli(controls + 2)
    cnot_gdepth, cnot_gcount = toffoli(controls + 1)
    not_gdepth, not_gcount = toffoli(controls)

    return [9 * toff_gdepth + 2 * cnot_gdepth + 2 * not_gdepth,
    29 * toff_gcount + 12 * cnot_gcount + 15 * not_gcount]

# Controlled increment based on in-place adder on n=10 bits from Draper et al.
# Returns T-depth, T-count
# Currently ignores n.
# Last parameter is whether this is a decrement operation. If not, we don't
# need to compute the control bit, as it's already been computed elsewhere.
def controlled_inc(n, controls, isdec):
    r = 2
    if isdec == 1:
        control_gdepth, control_gcount = double(toffoli(controls + r))
    else:
        control_gdepth, control_gcount = toffoli(controls + 1)
    toff_gdepth, toff_gcount = toffoli(controls + 2)
    cnot_gdepth, cnot_gcount = toffoli(controls + 1)

    return [control_gdepth + 14 * toff_gdepth + 3 * cnot_gdepth,
        control_gcount + 38 * toff_gcount + 25 * cnot_gcount]

# Perform diffusion map as used in R_A.
# Only correct for 1 or 0 control bits.
def diffusion(n, t, controls):
    r = 2
    s = int(ceil(log(n+1,2)))
    # Perform one level of amplitude amplification to produce a superposition state.
    # Then sandwich a phase inversion between two uses of this.

    # The T-count required to approximately produce an arbitrary state alpha|0> + beta|1>
    # for real alpha, beta. Based on Bocharov et al, arXiv:1404.5320.
    tcount = ceil(1.15 * log(64*sqrt(t*n),2) + 9.2)
    hpp = [tcount + 8, tcount + 20] # H'' operation, controlled on 1 qubit
    hp = [hpp[0], hpp[1] + 2*r] # H' operation, controlled on 1 qubit

    # if MULTI == 0:
        # print("T-count for single-qubit states: " + str(tcount))

    init = list(zip(
        LOGD(double(toffoli(s + controls)), "first toffoli (and uncompute)"),
        LOGD(double(toffoli(3 + controls)), "second toffoli (and uncompute)")
        ))

    v = t_expand([2, 2*r]) # V = H^{\otimes r}

    d = list(zip(
        LOGD([2*v[0], 2*v[1], 2*v[2]], "V and uncompute"),
        LOGD(toff_expand(toffoli(r + controls + 1)), "toffoli for D")
        ))

    dp = list(zip(
        t_expand([tcount,tcount]), # synthesise alpha|0>+beta|1>
        [sum(d[0])+2,sum(d[1]),sum(d[2])+8], # add extra control line to D
        t_expand([4,4]), # controlled-Hadamards
        toff_expand(double(toffoli(r+controls))), # check a (and uncompute)
        toff_expand(toffoli(controls + 3)), # controlled phase inversion
        ))

    total = list(zip(
        LOGD(toff_expand([sum(init[0]),sum(init[1])]), "diff init"),
        LOGD([sum(d[0]),sum(d[1]),sum(d[2])], "diff d"),
        LOGD([sum(dp[0]),sum(dp[1]),sum(dp[2])], "diff dp")
        ))

    out = [sum(total[0]),sum(total[1]),sum(total[2])]
    return out

# Perform diffusion map as used in R_B.
# Only correct for 1 or 0 control bits.
def diffusion_rb(n, t, controls):
    r = 2
    s = int(ceil(log(n+1,2)))
    # Perform one level of amplitude amplification to produce a superposition state.
    # Then sandwich a phase inversion between two uses of this.

    # The T-count required to approximately produce an arbitrary state alpha|0> + beta|1>
    # for real alpha, beta. Based on Bocharov et al, arXiv:1404.5320.
    tcount = ceil(1.15 * log(64*sqrt(t*n),2) + 9.2)
    hpp = [tcount + 8, tcount + 20] # H'' operation, controlled on 1 qubit
    hp = [hpp[0], hpp[1] + 2*r] # H' operation, controlled on 1 qubit

    if MULTI == 0:
        print("T-count for single-qubit states: " + str(tcount))

    init = list(zip(
        LOGD(double(toffoli(controls + 2)), "first toffoli (and uncompute)")
        ))

    v = t_expand([2, 2*r]) # V = H^{\otimes r}

    d = list(zip(
        LOGD([2*v[0], 2*v[1], 2*v[2]], "V and uncompute"),
        LOGD(toff_expand(toffoli(r + 1)), "toffoli for D")
        ))

    total = list(zip(
        LOGD(toff_expand([sum(init[0]),sum(init[1])]), "diff init"),
        LOGD([sum(d[0]),sum(d[1]),sum(d[2])], "diff d")
        ))

    out = [sum(total[0]),sum(total[1]),sum(total[2])]
    return out

# The complexity of evaluating the p function.
def p(n, m, k, controls):
    outputs = list(zip(
        LOGD(double(compute_parallel_copies(m, toffoli(k + controls))), "check clauses (and uncompute)"),
        LOGD(double(compute_parallel_copies(n, toffoli(2 + controls))), "check 2's (and uncompute)"),
        LOGD(toffoli(n + controls), "OR of 2's"),
        LOGD(toffoli(m + 1 + controls), "AND of all edges"),
        LOGD(double(compute_parallel_copies(n, toffoli(2 + controls))), "check *'s (and uncompute)"),
        LOGD(toffoli(n + 1 + controls), "check full assignment")
        ))
    out = [sum(outputs[0]),sum(outputs[1])]
    return out


# The complexity of evaluating the h function.
# Returns T-depth, T-count
def h(n, m, controls):
    s = int(ceil(log(n+1,2)))

    outputs = list(zip(
        LOGL(compute_parallel_copies(s, copy(1, controls)), "copy l to h"),
        LOGL(controlled_inc(s, controls, 0), "increment h")
    ))

    return [sum(outputs[0]), sum(outputs[1])]

# The complexity of the overall (controlled-)R_A and R_B operations.
def rarb(n, m, k, t, controls, aorb):
    r = 2
    s = int(ceil(log(n+1,2)))

    if aorb == 0:
        diff = diffusion(n, t, controls)
    else:
        diff = diffusion_rb(n, t, controls)

    outputs = list(zip(
        LOGT(toff_expand(double(convert(n,controls))), "convert (and uncompute)"), # convert input into correct form
        LOGT(toff_expand(double(compute_parallel_copies(r * n, copy(m,controls)))), "fan out and fan in"), # produce many copies of it
        toff_expand(compute_in_parallel(LOGT(p(n,m,k,controls+1), "p"), LOGT(h(n,m,controls+1), "h"))),
        LOGT(toff_expand(controlled_inc(n, controls, 1)), "controlled dec"),
        LOGT(toff_expand(toffoli(controls+2)), "invert phase"),
        LOGT(diff, "diffusion"),
        LOGT(toff_expand(double(toffoli(r + controls))), "check a and uncompute"),
        LOGT(toff_expand(controlled_inc(n, controls, 0)), "controlled inc"),
        toff_expand(compute_in_parallel(LOGT(p(n,m,k,controls), "p uncompute"), LOGT(h(n,m,controls+1), "h uncompute")))
    ))

    return [sum(outputs[0]),sum(outputs[1]),sum(outputs[2])]

def bt(n, m, k, t, controls=1):
    ra_params = LOGT(rarb(n, m, k, t, controls, 0), "RA")
    rb_params = LOGT(rarb(n, m, k, t, controls, 1), "RB")
    reps = ceil(32*sqrt(n) * sqrt(3/2)) # include cost of dummy "2" values for variables
    parallel_reps = 79

    #print("T-count of BT op: ", format_exp(ra_params[1]+rb_params[1]))
    return [ra_params[0]+rb_params[0], # t-depth
            7*(ra_params[1]+rb_params[1]) + # toffoli-count
            ra_params[2]+rb_params[2]] # T-count
    # return [reps * (ra_params[0]+rb_params[0]),
    #         parallel_reps * reps * (ra_params[1]+rb_params[1]),
    #         parallel_reps * reps * (ra_params[2]+rb_params[2])]

MULTI = 1
# if len(sys.argv) == 5:
#     MULTI = 1
#     k, nmin, nmax, controls = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
# elif len(sys.argv) == 4:
#     MULTI = 0
#     k, n, controls = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
# else:
#     print("Usage: " + sys.argv[0] + " k n_min [n_max] controls")
#     quit()

# if MULTI == 1:
#     for n in range(nmin, nmax+1):
#         m = ceil(ratios[k-3] * n)
#         t = 2 ** (scaling_exponent[k-3] * n + scaling_const[k-3])
#         tdepth, toffcount, tcount = bt(n, m, k, t, controls)
#         print(str(n) + "," + str(int(tdepth)) + "," + str(int(toffcount)) + "," + str(int(tcount)))
#     quit()

# m = ceil(ratios[k-3] * n)
# t = 2 ** (scaling_exponent[k-3] * n + scaling_const[k-3])

# tdepth, toffcount, tcount = bt(n, m, k, t, controls)

# print("Overhead T-depth: " + str(tdepth) + " (" + "{:.2E}".format(tdepth) + ")")
# print("Overhead Toff-count: " + str(toffcount) + " (" + "{:.2E}".format(toffcount) + ")")
# print("Overhead T-count: " + str(tcount) + " (" + "{:.2E}".format(tcount) + ")")
