"""
@author: Martijn Brehm (m.a.brehm@uva.nl)
@date: 15/01/2024
@location: Amsterdam

This script takes the CSV files in the data/ folder produced by the
the "run_experiment.py" script. It fits the complexity of the algorithms
described in the CSV files and provides various function to generate various
tables and plots showcasing these fits.

"""
from sys import argv
import numpy as np
import matplotlib.pyplot as plt
from math import log, log2
from os import listdir
# from scipy.stats import iqr

def round_3(n):
    """
    Round a number to 3 significant digits.
    """
    return float("{0:.2e}".format(n)) if len(str(n)) > 3 else n

def format_number(n):
    """
    Format a number as xxx cdot 10^{xxx} with 2 significant digits.
    """
    return "{0:.1e}".format(n).replace("e", "\\cdot 10^{") + "}"

def format_time(t):
    if t < 60:
        return "{0:.0f}".format(t) + " seconds"
    if t < 60*60:
        return "{0:.0f}".format(t//60) + " months"
    if t < 60*60*24:
        return "{0:.0f}".format(t//(60*60)) + " hours"
    if t < 60*60*24*365:
        return "{0:.0f}".format(t//(60*50*24)) + " days"
    if t < 60*60*24*365*1000:
        return "{0:.0f}".format(t//(60*60*24*365)) + " years"
    # if t < 60*60*24*365*1000*10000:
    return "{0:.0f}".format(t//(60*60*24*365*1000)) + " millenia"
    # return ">>"

T_COUNT = 1 # If 1, use T-count, if 0 use T-depth.
T_MAX = 100 # Temperature for random data.

# Column index of algorithms in the .csv files.
algs = {
    "Classical" : 1,
    "Detection" : 9 + T_COUNT, # 8 = query complexity 9 = t-depth 10 = t-coun
    "Binary search" : 12 + T_COUNT, # 11 = query complexity 12 = t-depth 13 = t-coun
    "Grover" : 15 + T_COUNT, # 14 = query complexity 15 = t-depth 16 = t-coun
}

# What number of variables to start exponential fit at for classical sat solver.
clas_fit = [160, 64, 47, 30, 25, 20, 15, 15, 14, 15, 16]
files = ["data/" + f for f in listdir("data/") if "-sat-" in f] if len(argv) == 1 else argv[1:]

def compute_fits(files):
    """
    Read data and compute linear least squares fits of log of median complexities.
    Save the slope and intercept of these fits to 'fits' organised by algorithm,
    then by satisfiability, then by clause size k, and then by temperature,
    where we take random data to have temperature 100.
    """
    fits = {alg : {"sat" : {}, "unsat" : {}, "both" : {}} for alg in algs}
    data = {}
    for file in files:
        print(file)
        # Extract data from file.
        f = file.split('/')[1].split('-')
        solver, k, n1, n2, stepsize, mode, reps = f[0],int(f[1]),int(f[4]),int(f[5]),int(f[6]),f[3], int(f[7].split('.')[0])
        T = T_MAX if mode == "random" else float(f[3].split('_')[2])
        dims = ((n2 - n1) // stepsize + 1, reps)
        data["sat"] = np.loadtxt(open(file, "rb"), delimiter=",", skiprows=1).T
        data["unsat"] = np.loadtxt(open(file.replace("sat","unsat"), "rb"), delimiter=",", skiprows=1).T

        # For CDCL, only consider clasical alg; for BT only consider quantum algs.
        xs = range(n1, n2 + 1, stepsize)
        for alg in algs:
            if "Clas" in alg and solver == "BT" or "Clas" not in alg and solver == "CDCL":
                continue
            print(alg)

            # Compute and save exponential fits.
            n = (clas_fit[k - 3] - n1) // stepsize if "Clas" in alg else 0
            shaped = {}
            shaped["sat"] = np.reshape(data["sat"][algs[alg]], dims)
            shaped["unsat"] = np.reshape(data["unsat"][algs[alg]], dims)
            shaped["both"] = np.array(list(zip(shaped["sat"], shaped["unsat"]))).reshape(dims[0], dims[1] * 2)
            for type in shaped:
                # for a in shaped[type]:
                    # print(iqr(a) * 1.253 / np.sqrt(len(a)) /np.median(a))
                fits[alg][type][k] = fits[alg][type][k] if k in fits[alg][type] else {}
                ys = np.median(shaped[type], axis=1)
                s, i = np.polyfit(xs[n:], np.log2(ys)[n:], 1)
                fits[alg][type][k][T] = (s, i, clas_fit[k - 3] if "Clas" in alg else n1, n2, xs, ys)
    return fits

def create_table(fits, type="both", scaling=True, color_one_day=False, no_text=False, meas_time=1e-9):
    """
    Generates a latex table over T and k denoting scaling of each of the four
    algorithms. Cells (T,k) is colored depending on which algorithm has the
    best scaling for that type of SAT instances.

    Params:
    - type : if "both" then a 50/50 mix of satisfiable and unsatfiable instances
             is used. other options are "sat" or "unsat".
    - scaling : if set, display the scaling of the algorithm. Otherwise display
                the constants required to match classical given measurement times
                10^{-6}, 10^{-7} and 10^{-8}.
    - color_one_day : if set, cells are colored not based on scaling but on
                      which alg can solve the largest instance in a single day.
    - meas_time : measurement time used to determine size of largest instance
                  quantum alg can solve in a day.
    - no_text : if set, not text is generated, yielding a table of colored cells.
    """
    Ts = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, T_MAX] if scaling else [2.0, 3.0, 5.0, 10.0, T_MAX]
    ks = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12] if scaling else [6, 7, 8, 9, 10, 11, 12]
    alg_to_c = {"Classical" : "C0", "Detection" : "C1", "Binary search" : "C2", "Grover" : "C3"}

    print("\\begin{table} \\hspace{-1.8cm}")
    print("\\begin{tabular}{c|c|c|c|c|c|c|c|c}")
    print("\\backslashbox{$k$}{$T$} & $\\frac 12$ & 1 & $\\frac 32$ & 2 & 3 & 5 & 10 & $\\infty$\\\\\\hline")
    times, colors = {}, {}
    for k in ks:
        Ts_current = Ts if k >= 7 else Ts[2:] if k == 6 else Ts[3:] if k == 5 else Ts[4:] if k == 4 else Ts[5:] if k == 3 else [T_MAX]
        preamble = "& \\cellcolor{" + alg_to_c["Classical"] + "}"
        preamble *= (8 - len(Ts_current))
        print(str(k) + preamble,end='')
        for T in Ts_current:

            # Colors based on which alg can solve largest instance in one day.
            c_s, c_i = fits["Classical"][type][k][T][:2]
            best_one_day = (16.3987 - c_i) / c_s
            colors[T] = alg_to_c["Classical"]
            times[T] = {}
            prev = c_s

            # Check if quantum can solve a larger instance in one day.
            for alg in ["Detection", "Binary search", "Grover"]:
                s, i = fits[alg][type][k][T][:2]

                # Compute time needed to match classical, given meas. time. Color based on best scaling.
                if s < c_s:
                    times[T][alg] = []
                    for t in [1e-6, 1e-7, 1e-8]:
                        n = (c_i - i - log2(t))/(s - c_s)
                        # Universe age.
                        if s * n + i  > log2(436117077000000000/t):
                            times[T][alg].append(">>")
                        else:
                            times[T][alg].append(format_time(2**(s * n + i) * t))

                    if not color_one_day and (alg != "Grover" or (alg == "Grover" and s < prev)):
                        colors[T] = alg_to_c[alg]

                # Color cell if this alg can solve largest instance in one day given given meas. time.
                q_val = (16.3987 - 1.4427 * log(meas_time) - i) / s
                if color_one_day and q_val > best_one_day and not alg == "Detection":
                    colors[T] = alg_to_c[alg]
                    best_one_day = q_val
                prev = s

            # Print scaling for classical alg.
            print(" & \\footnotesize \\cellcolor{" + colors[T] + "} ", end='')
            if not no_text and scaling:
                print("$" + str(round_3(c_s))[1:] + "n + " + str(round_3(c_i)), end='$')

        # Print scaling of quantum algs on separate rows. Skip this if no text.
        if no_text:
            print("\\\\")
            continue
        for alg in  ["Detection", "Binary search", "Grover"]:
            print("\\\\" + preamble,end='')
            for T in Ts_current:
                s = str(round_3(fits[alg][type][k][T][0]))[1:] + "n + " + str(round_3(fits[alg][type][k][T][1]))
                if not scaling:
                    s = times[T][alg][0] + "," + times[T][alg][1] + "," + times[T][alg][2] if alg in times[T] else ""
                # print(" & "+ str(s)[1:] + c + "\\cellcolor{" + colors[T] + "}", end='')
                print(" & \\footnotesize $" + s + "$ \\cellcolor{" + colors[T] + "}", end='')
            print()
        print("\\\\\\hline")
    print("\\end{tabular}")
    T = "T-count" if T_COUNT else "T-depth"
    print("\\caption{" + type, end='')
    print(", scaling" if scaling else "constants", end='')
    print(", colored based on one day speed-up" if color_one_day else ", color based on scaling", end='')
    print(", T-count" if T_COUNT else ", T-depth", end='')
    print("}")
    print("\\end{table}")

def create_table_meta(fits):
    """
    Generate latex table displaying ranges of n and repetitions done for each type of SAT instance.
    """
    Ts = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, T_MAX]
    ks = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    print("\\begin{figure}")
    print("\\begin{tabular}{c|c|c|c|c|c|c|c|c}")
    print("\\backslashbox{$k$}{$T$} & $\\frac 12$ & 1 & $\\frac 32$ & 2 & 3 & 5 & 10 & $\\infty$\\\\\\hline")
    for k in ks:
        Ts_current = Ts if k >= 7 else Ts[2:] if k == 6 else Ts[3:] if k == 5 else Ts[4:] if k == 4 else Ts[5:] if k == 3 else [T_MAX]
        preamble = " &" * (8 - len(Ts_current))
        print(str(k) + preamble,end='')

        for T in Ts_current:
            n1, n2 = fits["Classical"]["sat"][k][T][2:4]
            print(" & " + str(n1) + " - " + str(n2), end='')
        print("\\\\")
        print(preamble,end='')
        for T in Ts_current:
            n1, n2 = fits["Detection"]["sat"][k][T][2:4]
            print(" & " + str(n1) + " - " + str(n2), end='')
        print("\\\\\\hline")
    print("\\end{tabular}")
    print("\\caption{}")
    print("\\end{figure}")

def plot_fits(fits, shift_plots):
    """
    Plot fits of complexity over number of variables.
    """
    cs = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    plt.figure(figsize=[13.5, 7])
    plt.subplots_adjust(left=0.07, right=0.93)
    plt.rcParams.update({"text.usetex":True, "font.family":"serif", "font.size":13})
    plt.title("Median runtime of CaDiCaL 1.7.5 on random $k$-SAT instances")
    plt.ylabel("Number of seconds")
    plt.xlabel("Number of variables")
    plt.yscale("log")
    ks = [3, 4, 5, 6, 7, 8, 9, 10, 11]

    for j, alg in enumerate(algs):
        for sat in ["both"]:
            for k in ks:
                print(alg,sat,k)
                for T in fits[alg][sat][k]:
                    if not "Clas" in alg or T != T_MAX:
                        continue

                    # Extract information.
                    mode = "random" if T == T_MAX else "community"
                    (s, i, n1, n2, xs, ys) = fits[alg][sat][k][T]
                    if alg == "Grover" and mode == "random":
                        print(k, s)
                    offset = 2**i if shift_plots else 1
                    print(n1,n2)
                    # Plot data points.
                    plt.scatter(xs, ys/offset, marker="o", c=cs[j], linewidth=0.01)

                    # Plot fit.
                    style = '-' if sat == "sat" else '--'
                    m = "$k="+str(k)+",T="+str(T)+"$" if mode == "Community" else "$k="+str(k)+"$"
                    # label = "{} {} {}\n{}\n${}\cdot 2^".format(mode, sat, m, alg, format_number(2**i)) + "{" + str(round(s,3)) + "n}$"
                    label = "{} {} {}\n{}\n${}\\cdot 2^".format(mode, sat, m, alg, format_number(2**i)) + "{" + str(round(s,3)) + "n}$"
                    plt.plot(range(n1,n2), [2**(s * n + i) / offset for n in range(n1,n2)], ls=style, label=label, c=cs[j])
                    plt.annotate(m, xy=(xs[-1]+2, ys[-1]/offset), c=cs[j])

    # plt.plot(range(0,100), [2**(0.5 * x) for x in range(0, 100)], c=cs[6])
    xint = [int(each) for each in plt.xticks()[0]][1:-1]
    plt.xticks(range(min(xint),max(xint)+1,10))
    # plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    plt.show()

fits = compute_fits(files)


# Table 2. Colored grid based on scaling and solving instances in one day.
create_table(fits, "both", scaling=True, color_one_day=False, no_text=True)
create_table(fits, "both", scaling=True, color_one_day=True, no_text=True, meas_time=1e-6)
create_table(fits, "both", scaling=True, color_one_day=True, no_text=True, meas_time=1e-9)

# Colored table with scaling of all algs and then with crossover time of all
# algs. Snippets are used in Section 3 for Table 3 and Table 4.
create_table(fits, "both", scaling=True, color_one_day=False, no_text=False)
create_table(fits, "both", scaling=True, color_one_day=False, no_text=False)

# Table 5. Colore grid based on scaling, sat vs. unsat.
create_table(fits, "both", scaling=True, color_one_day=False, no_text=True)
create_table(fits, "sat", scaling=True, color_one_day=False, no_text=True)
create_table(fits, "unsat", scaling=True, color_one_day=False, no_text=True)

# Appendix: big table for sat, unsat, and a mixture, for t-count and t-depth.
create_table(fits, "both", scaling=True, color_one_day=False, no_text=False)
create_table(fits, "sat", scaling=True, color_one_day=False, no_text=False)
create_table(fits, "unsat", scaling=True, color_one_day=False, no_text=False)

# Appendix Table 7. Ranges of experiments.
create_table_meta(fits)

# Extra table: cross-over time for each regime.
# create_table(fits, "both", scaling=False, color_one_day=False, no_text=False)
# create_table(fits, "sat", scaling=False, color_one_day=False, no_text=False)
# create_table(fits, "unsat", scaling=False, color_one_day=False, no_text=False)

# Plot scaling of algs.
# plot_fits(fits, False)
