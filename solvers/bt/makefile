CC=gcc
CFLAGS=-Wall -Wextra -std=c99

all: solver solver_verbose solver_print_paths

solver: bt.c
	$(CC) $(CFLAGS) $^ -o $@

solver_verbose: bt.c
	$(CC) $(CFLAGS) -DVERBOSE $^ -o $@

solver_print_paths: bt.c
	$(CC) $(CFLAGS) -DPRINT_PATHS $^ -o $@

clean:
	rm -f solver solver_verbose solver_print_paths
