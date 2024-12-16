// Simple backtracking algorithm for k-SAT.
// Takes as input a file specifying a k-CNF formula in DIMACS format.
// Determines satisfiability using the algorithm in the paper.
// Various heuristics available (variables ordered 1...n, variables ordered by
// appearance count at start, variables ordered by appearance count in
// simplified formula)
//
// @author original from https://doi.org/10.5523/bris.19va21gun3c7629f291kmd6w37
// modified by Martijn Brehm, 27 November 2023, Amsterdam.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/times.h>

#define FALSE 0
#define TRUE 1
#define INDETERMINATE 2
// #define VERBOSE
#define LINE_MAX 500

// @change moved variables used for timing to main function.
int n, k, m;
int found_satisfying_assignment;
unsigned char *satisfying_assignment;
int **clauses;

// @change: added these variables to extract relevant data from the sat solver.
// added PRINT_PATHS option to print the paths to marked elements
int nodes = 0;
int depth = 0;
int first_solution_nodes = 0;
int first_solution_depth = 0;
int n_backtracking_sols = 0;
int n_leaf_sols = 0;
// #define PRINT_PATHS

typedef struct {
  int index;
  int count;
} order;
order *ordered_vars;

int isindet(int *clause, unsigned char *x) {
  int j, c;
  for (j = 0; j < k; j++) {
    c = clause[j];
    if (x[abs(c) - 1] == 0)
      return TRUE;
  }
  return FALSE;
}

int appearsin(int *clause, int i) {
  int j, c;
  for (j = 0; j < k; j++) {
    c = clause[j];
    if (i == (abs(c) - 1))
      return TRUE;
  }
  return FALSE;
}

int psat(unsigned char *x) {
  int i, j, c;
  int sat;
  int indet = FALSE;

  for (i = 0; i < m; i++) {
    sat = FALSE;
    for (j = 0; j < k; j++) {
      c = clauses[i][j];
      if (x[abs(c) - 1] == 0) // indeterminate -> next clause
      {
        indet = TRUE;
        sat = TRUE;
        break;
      }
      if ((c > 0) && (x[c - 1] == 2)) {
        sat = TRUE;
        break;
      }
      if ((c < 0) && (x[-c - 1] == 1)) {
        sat = TRUE;
        break;
      }
    }
    if (sat == FALSE)
      return FALSE;
  }
  if (indet == TRUE)
    return INDETERMINATE;
  return TRUE;
}

int hcount(unsigned char *x) {
  int *counts;
  int index_max = 0;
  int counts_max = 0;
  int i, j;

  counts = calloc(n, sizeof(int));

  for (i = 0; i < n; i++) {
    if (x[i] == 0) {
      for (j = 0; j < m; j++) {
        if (isindet(clauses[j], x) && appearsin(clauses[j], i))
          counts[i]++;
      }
    }
  }
  for (i = 0; i < n; i++) {
    if (x[i] == 0) {
      if (counts[i] > counts_max) {
        counts_max = counts[i];
        index_max = i;
      }
    }
  }
  free(counts);
  return index_max;
}

int hsimp_ordered(unsigned char *x) {
  int i;
  for (i = 0; i < n; i++) {
    if (x[ordered_vars[i].index] == 0)
      return ordered_vars[i].index;
  }
  printf("h was passed a complete assignment!\n");
  return 0;
}

int hsimp(unsigned char *x) {
  int i;
  for (i = 0; i < n; i++) {
    if (x[i] == 0)
      return i;
  }
  printf("h was passed a complete assignment!\n");
  return 0;
}

// @change: added this function to get the depth of the current node (needed to
// know how deep binary search needs to go).
int getdepth(unsigned char *x) {
  int i;
  int k = 0;
  for (i = 0; i < n; i++) {
    if (x[i] == 1 || x[i] == 2)
      k += 1;
  }
  return k;
}

// @change: added this function to print the path represented by the current
// assignment as a bitstring, where a 0 means we branched to set the current
// variable F and 1 means we branched to set the current variable T. We nede
// these to compute the effective resistance between the root and the marked
// vertices.
void print_path(unsigned char *x) {
  int a, i;
  for (i = 0; i < n; i++) {
    a = ordered_vars[i].index;
    if (x[a] == 1)
      printf("0");
    if (x[a] == 2)
      printf("1");
  }
  printf("\n");
}

// @change: added this function for debugging purposes.
void print_assignment(unsigned char *x) {
  int i;
  for (i = 0; i < n; i++) {
    if (x[i] == 0)
      printf("*");
    if (x[i] == 1)
      printf("F");
    if (x[i] == 2)
      printf("T");
  }
  printf("\n");
}

int bt(unsigned char *x) {
  int i, next, ret, pval;
  unsigned char *y;

  // @change: added this to count the size of the backtracking tree, and
  // determine the maximum depth of the tree.
  nodes++;
  if (getdepth(x) > depth)
    depth = getdepth(x);

#ifdef VERBOSE
  printf("Calling BT at depth=%d/%d and size %d with x=", getdepth(x), depth,
         nodes);
  print_assignment(x);
#endif

  pval = psat(x);
#ifdef VERBOSE
  printf("satisfied? %d\n", pval);
#endif
  if (pval == TRUE) {
    n_backtracking_sols += 1;
    n_leaf_sols += (2 << (n - getdepth(x))) / 2; // Count all leafs below sol.
    found_satisfying_assignment = TRUE;
    memcpy(satisfying_assignment, x, n);
    if (n_backtracking_sols == 1) {
      first_solution_nodes = nodes;
      first_solution_depth = getdepth(x);
#ifdef VERBOSE
      printf("Found first solution. Depth=%d size=%d and assignment=",
             getdepth(x), nodes);
      print_assignment(x);
#endif
    }
#ifdef PRINT_PATHS
    print_path(x);
#endif
  }
  if ((pval == TRUE) || (pval == FALSE))
    return pval;
  y = malloc(n);
  memcpy(y, x, n);
  next = hsimp_ordered(x);
  for (i = 1; i <= 2; i++) {
    y[next] = i;
    ret = bt(y);
    // Uncomment this to model returning the first satisfying assignment.
    // if (ret == TRUE)
    //    break;
  }
  free(y);
  return ret;
}

void readclauses(const char *filename) {
  FILE *file;
  char line[LINE_MAX];
  char *ret;
  int c = 0;

  file = fopen(filename, "rb");
  do {
    ret = fgets(line, LINE_MAX, file);

    if (ret != NULL) {
      if (line[0] == 'c') {
        sscanf(line, "c Random %d-CNF", &k);
      } else if (line[0] == 'p') {
        sscanf(line, "p cnf %d %d", &n, &m);
        clauses = calloc(m, sizeof(int *));
        for (c = 0; c < m; c++)
          clauses[c] = calloc(k, sizeof(int));
        c = 0;
      } else {
        int i = 0;
        char *token = strtok(line, " ");
        while (token) {
          int tok = 0;
          sscanf(token, "%d", &tok);
          if (tok == 0)
            break;
          clauses[c][i] = tok;
          i++;
          token = strtok(NULL, " ");
        }
        c++;
      }
    }
    // printf("%s", line);
  } while (ret != NULL);
  fclose(file);
}

int cmpfunc(const void *a, const void *b) {
  return (((order *)b)->count - ((order *)a)->count);
}

// @change: cleaned up code, slightly changed the control flow to simplify
// things, and made the program print all the extra data we need. Specifically,
// we first output the path from root to each solution on a separate line as a
// bitstring, where 0 means we went left and 1 means we went right. We then
// output the following on one line, separated by commas:
// - tree size,
// - tree depth,
// - number of solutions,
// - size of tree upon finding first solution,
// - depth of the first solution,
// - number of seconds the solver ran for,.
int main(int argc, char *argv[]) {
  struct timeval begin, end;
  unsigned char *x;
  int i, j;
  int *counts;

  // Read input file and input value for k.
  if (argc != 3) {
    printf("Usage: %s [CNF file] [k]\n", argv[0]);
    return 0;
  }
  sscanf(argv[2], "%d", &k);
  readclauses(argv[1]);
  counts = calloc(n, sizeof(int));
  satisfying_assignment = calloc(n, sizeof(unsigned char));

  // Read SAT instance and count variables to set up heuristic.
  for (i = 0; i < m; i++) {
    for (j = 0; j < k; j++)
      counts[abs(clauses[i][j]) - 1]++;
  }
  ordered_vars = calloc(n, sizeof(order));
  for (i = 0; i < n; i++) {
    ordered_vars[i].index = i;
    ordered_vars[i].count = counts[i];
  }
  qsort(ordered_vars, n, sizeof(order), cmpfunc);

#ifdef VERBOSE
  printf("%d-SAT formula with %d vars, %d clauses\n", k, n, m);
  printf("Appearance counts: ");
  for (i = 0; i < n; i++)
    printf("%d ", counts[i]);
  printf("\n");
  for (i = 0; i < n; i++) {
    printf("Var %d was var %d (count %d); ", i, ordered_vars[i].index,
           ordered_vars[i].count);
  }
  printf("\n");
#endif

  // Backtracking begins
  gettimeofday(&begin, NULL);
  x = calloc(n, sizeof(unsigned char));
  bt(x);
  gettimeofday(&end, NULL);
  // Work out the elapsed real time and output result.
  double elapsed =
      (end.tv_sec - begin.tv_sec) + ((end.tv_usec - begin.tv_usec) / 1000000.0);
  printf("%.3f,%d,%d,%d,%d,%d,%d", elapsed, nodes, depth, n_backtracking_sols,
         n_leaf_sols, first_solution_nodes, first_solution_depth);

#ifdef VERBOSE
  if (found_satisfying_assignment) {
    printf("SATISFIABLE, e.g. x=");
    print_assignment(satisfying_assignment);
  } else
    printf("UNSATISFIABLE\n");
#endif

  free(x);
  return 0;
}
