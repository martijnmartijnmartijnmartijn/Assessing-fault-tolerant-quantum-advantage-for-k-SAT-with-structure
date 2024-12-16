"""
@author: Martijn Brehm (m.a.brehm@uva.nl)
@date: 15/01/2024
@location: Amsterdam

This script generates SAT instances and solves them. If solved using a modern
SAT solver, the running time is saved. If solved using a simple backtracking
algorithm, features collected during the solving are used to compute
(upper-bounds on) the time complexity of three different quantum algorithms: a
quantum backtracking detection algorithm, quantum backtracking
search algorithm and Grover's algorithm.

Specifically, the script requires input parameters
- k (integer)
- n1 (integer)
- n2 (integer)
- stepsize (integer)
- reps (integer)
- solver ("BT" (default) if the base backtracking solver should be used,
          "CDCL" if the modern solver should be used)
- mode ("random" (default) or "community")
- beta (0 (default), 1)
- T (1 (default))

The script then generates "k"-SAT instances from class "mode" in "n1" up
to and including "n2" variables, with steps of "stepsize" between each n. For
each n, it generates "reps" satisfiable and "reps" unsatisfiable instances. It
saves the following data into the file
data/solver-k-sat-mode_beta_T-n1-n2-reps.csv for satisfiable instance, and the
same for unsatisfiable instance but "sat" in changed to "unsat" (where "beta"
and "T" are written only if the mode is "community"):
- the number of variables
- the number of seconds the classical solver needed to solve this instance.
and if BT is used also saves:
- the size of the backtracking tree.
- the depth of the backtracking tree.
- the number of satisfying assignments in the backtracking tree.
- the total number of satisfying assignments.
- the size of the backtracking tree upon finding the first solution.
- the depth of the backtracking tree upon finding the first solution.
- query complexity, t-depth and t-count of quantum detection algorithm.
- query complexity, t-depth and t-count of quantum search algorithm.
- query complexity, t-depth and t-count of Grover's algorithm.

A time out of 5 minutes is set, which can be changed by altering TIMEOUT.
This means that the whole program will be terminates if generating and
solving 'reps' instances of a given number of variables n exceeds TIMEOUT
seconds.

Example 1: the command "python3 run_experiment.py 3 10 40 50 BT random"
generates 50 uniformly random satisfiable
3-SAT instances in 10, 11, ..., 40 variables, and the same for unsatisfiable instances. It solves them using the backtracking algorithm and writes all the relevant data to "data/BT-3-sat-random-10-40-50.csv" and
"data/BT-3-unsat-random-10-40-50.csv".

Example 2: "python3 run_experiment.py 3 10 40 50 BT community 0.6 2"
would do the same, except not generating random instances but instances iwth community where beta = 0.6 and T = 2. It saves the data to "data/BT-3-sat-community_0.6_2-10-40-50.csv" and "data/BT-3-sat-community_0.6_2-10-40-50.csv".
"""
from sys import argv
from math import ceil, sqrt, log, comb, floor
from random import randint
from csv import reader, writer
from cnfgen.families.randomformulas import RandomKCNF as RandomKCNF
from subprocess import check_output
from time import time
from os import remove
from pysat.solvers import Solver
from pysat.formula import CNF
from gate_complexity.satcomplexity import bt
from gate_complexity.satcomplexity_grover import grover

# Indices into data from .csv files, and constants.
N_VARS = 0
TIME = 1
SIZE = 2
N_SOLUTIONS = 4
N_SOLUTIONS_TOTAL = 5
SOL_1_DEPTH = 7
TIMEOUT = 60 * 15
# See Appendix A of the paper for optimal (C,n_reps) for different delta.
DELTA = 0.001
C = 3.73538045831806
n_reps = 23

# Clauses/variables ratios of phase transition point for 3-SAT, ..., 13-SAT.
ratios = (4.267,9.931,21.117,43.37,87.79,176.54,354.01,708.92,1418.71,2838.28,
          5677.41,11355.67,22712.20)

def get_n_reps(delta, n):
    """
    Given error probability delta of an algorithm. Outputs the number of
    repetitions required to guarantee that n succesive runs of the algorithm
    are still correct with probability 1-delta.
    """
    target = 1 - (1-delta)**(1/n)
    exp = lambda l : delta**l * sum([comb(l, i) * ((1-delta)/delta)**i for i in range(1 + floor(l/2))])
    l = 3
    while exp(l) > target:
        l += 2
    return l

def q_bt(C, n_reps, R, W):
    """
    Given constants C and n, and upper-bounds R on effective resistance and W
    on the sum of weights in the graph, computes the number of queries Belovs'
    detection algorithm makes.
    """
    return n_reps * (2**(log(sqrt((1 + C**2) * (1 + C * W * R)), 2))) - n_reps

def q_bt_b_search(C, n_reps, R, W, tree_depth, sol_depth):
    """
    Given constants C and n, upper-bounds R on effective resistance and W
    on the sum of weights in the graph, the depth of the backtracking tree, and
    the depth of the first solution that the heuristic walks towards, computes
    the number of queries that the binary search algorithm using Belovs'
    detection algorithm makes.
    """
    return q_bt(C, n_reps, R, W) * sol_depth * get_n_reps(DELTA, tree_depth)

def q_bt_b_search_est_W(C, n_reps, R, W, tree_depth):
    """
    Given constants C and n, upper-bounds R on effective resistance and W
    on the sum of weights in the graph and the depth of the backtracking tree, computes an upper-bound on the number of queries that the binary search algorithm using Belovs' detection algorithm makes, which determines by itself an upper-bound on W.
    """
    return sum([q_bt(C, n_reps, R, 2**w) * get_n_reps(DELTA, tree_depth) * tree_depth for w in range(ceil(log(W, 2)))])

def q_grover(L, t):
    """
    Given list size L and number of marked elements t, computes the expected
    number of queries to the list needed by Grover's algorithm.
    """
    if t == 0:
        return 9.2 * ceil(log(1/DELTA, 3)) * sqrt(L)
    F = 2.0344 if L/4 <= t else 9/4 * L/sqrt((L - t)*t) + ceil(log(L/sqrt((L - t)*t), 6/5)) - 3
    return F * (1 + 1/(1 - F/(9.2 * sqrt(L))))

# Read input arguments.
k = int(argv[1])
n1 = int(argv[2])
n2 = int(argv[3])
stepsize = int(argv[4])
reps = int(argv[5])
solver = argv[6] if len(argv) > 6 else "BT"
mode = argv[7] if len(argv) > 7 else "random"
beta = argv[8] if len(argv) > 8 else 0
T = argv[9] if len(argv) > 9 else 1

# Set-up the files that we write data to.
mode = "community_" + str(beta) + "_" + str(T) if mode == "community" else "random"
end = mode + "-" + str(n1) + "-" + str(n2) + "-" + str(stepsize) + "-" + str(reps) + ".csv"
file_sat = "data2/" + solver + "-" + str(k) + "-sat-" + end
file_unsat = "data2/" + solver + "-" + str(k) + "-unsat-" + end
print("Saving data to files:\n{}\n{}".format(file_sat, file_unsat))
file_sat = open(file_sat, "w", newline='')
writer_sat = writer(file_sat, delimiter=',')
file_unsat = open(file_unsat, "w", newline='')
writer_unsat = writer(file_unsat, delimiter=',')

# # Write initial line to files.
first_line = ["n", "time (classical)", "T", "depth", "M tree", "M total", "T (first solution)", "depth (first solution)", "queries (detect)", "T-depth (detect)", "T-count (detect)", "queries (search)", "T-depth (search)", "T-count (search)", "queries (Grover)", "T-depth (Grover)", "T-count (Grover)"]
writer_sat.writerow(first_line)
writer_unsat.writerow(first_line)

for n in range(n1, n2 + 1, stepsize):
    print("n={}: ".format(n), end='')
    m = int(ceil(ratios[k-3] * n))
    total_generated, sat_count = 0, 0
    r_sat, r_unsat = 0, 0
    start = time()

    # Generate and solve 'reps' instances, or until exceeding the timeout.
    while r_sat < reps or r_unsat < reps:
        if time() - start > TIMEOUT:
            print("Timeout: current iter took {}s which exceeds the alloted time of {}s".format(time() - start, TIMEOUT))
            print("\t", r_sat,r_unsat)
            break
            exit(0)

        # Generate instance according to specified generation mode.
        if "community" in mode:
            formula = check_output(["./generators-IJCAI17/psc3.1", "-n " +
               str(n), "-m " + str(m), "-K " + str(k), "-T " + str(T), "-b " +
               str(beta), "-s " + str(randint(0,1000000))], encoding="utf8")
        else: # "random" in mode:
            formula = RandomKCNF(k, n, m).to_dimacs()
        tempfile = "{}.dimacs".format(randint(0,10000000))
        temp = open(tempfile, "w")
        temp.write(formula)
        temp.close()

        # Solve instance according to specified solver.
        data = [n]
        if solver == "CDCL":
            solver_start = time()
            # s = Solver(name="cd", bootstrap_with=CNF(from_string=formula), use_timer=True)
            # solve = s.solve()
            # data.append(r(s.time_accum()))
            output = check_output(["./solvers/SBVA/sbva_wrapped", "solvers/CaDiCaL/build/cadical", tempfile, "h"], encoding="utf8")
            data.append(float("{0:.5e}".format(time() - solver_start)))
            solve = "s SATISFIABLE" in output
        else: # solver == "BT":
            output = check_output(["./solvers/bt/solver", tempfile, str(k)],
                encoding="utf8").split(',')
            data += [float(output[0])]
            data += [int(val) for val in output[1:]]
            solve = data[N_SOLUTIONS] > 0
            sol_depth = data[SOL_1_DEPTH] if data[SOL_1_DEPTH] > 0 else n

            # Add q backtracking alg query complexity, t depth and t count.
            bt_t_depth, bt_t_count = bt(n, m, k, data[SIZE])
            data += [q_bt(C, n_reps, n, data[SIZE])]
            data += [data[-1] * bt_t_depth, data[-1] * bt_t_count]
            data += [q_bt_b_search_est_W(C, n_reps, n, data[SIZE], n)]
            data += [data[-1] * bt_t_depth, data[-1] * bt_t_count]

            # Add grover query complexity, t depth and t count.
            grover_t_depth, grover_t_count = grover(n, m, k)
            data += [q_grover(2**n, data[N_SOLUTIONS_TOTAL])]
            data += [data[-1] * grover_t_depth, data[-1] * grover_t_count]
        remove(tempfile)

        # Save to sat/unsat file depending on if the instance was satisfiable.
        if solve and r_sat < reps:
            writer_sat.writerow(data)
            r_sat += 1
        elif not solve and r_unsat < reps:
            writer_unsat.writerow(data)
            r_unsat += 1

        # Keep track of what fraction of generated instances was satisfiable.
        total_generated += 1
        sat_count = sat_count + 1 if solve else sat_count
    elapsed = time() - start
    if elapsed > TIMEOUT:
        break
    print("  Took {}m{}s  Fraction satisfiable: {} ({}/{})".format(int(elapsed // 60), int(elapsed % 60), round(sat_count/total_generated,2), sat_count, total_generated))

file_sat.close()
file_unsat.close()
