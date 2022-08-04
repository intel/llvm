// RUN: cgeist %s %stdinclude -S | FileCheck %s
// RUN: cgeist %s %polyexec %stdinclude -O3 -o %s.execm && %s.execm | FileCheck %s --check-prefix EXEC
// requires loop restructure use after while fix
// XFAIL: *
/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/*
Nussinov.c

Builds a random sequence of base pairs, runs the Nussinov algorithm.
Unless FOUR_WAY_MAX_WITH_REDUNDANCY is define'd to true, it uses the more concise version as per the comment at the end of the file.

Programmed by Dave Wonnacott at Haverford College <davew@cs.haverford.edu>, with help from Allison Lake, Ting Zhou, and Tian Jin, based on algorithm by Nussinov, described in Allison Lake's senior thesis.


compile&run with:
   gcc -O3 -DNDEBUG -Wall -Wno-unused-value -Wno-unknown-pragmas  Nussinov.c -o Nussinov && ./Nussinov

Or, to set a specific size such as 5000, use:
   gcc -O3 -DNDEBUG -Dsize=5000 -Wall -Wno-unused-value -Wno-unknown-pragmas  Nussinov.c -o Nussinov && ./Nussinov

-------------
For debugging/verifying we hit all the right points and have all updates before any pure reads:
   gcc -Dsize=250 -DCHECK_DEPENDENCES=1 -DVERBOSE=1 -DVERBOSE_OUT=stderr -Wall -Wno-unused-value -Wno-unknown-pragmas  Nussinov.c -o Nussinov && ./Nussinov 2> /tmp/Nussinov-log-250.txt ; sort /tmp/Nussinov-log-250.txt -o /tmp/Nussinov-log-250.txt

-------------
Re-generate small or big matrix for comparison:
  gcc           -O3 -DPRINT_SIZE=5000 -Wall -Wno-unused-value -Wno-unknown-pragmas  Nussinov.c -o Nussinov-print && echo 250 | ./Nussinov-print > printed-matrix-250.out
Or do the comparison
  gcc           -O3 -DPRINT_SIZE=5000 -Wall -Wno-unused-value -Wno-unknown-pragmas  Nussinov.c -o Nussinov-print && echo 250 | ./Nussinov-print | diff printed-matrix-250.out -

-------------
Re-generate big matrices for comparison:
  gcc           -O3 -Dsize=5000 -DPRINT_SIZE=5000 -Wall -Wno-unused-value -Wno-unknown-pragmas  Nussinov.c -o Nussinov && ./Nussinov > printed-matrix-5000.out

  gcc -DSEED=17 -O3 -Dsize=5000 -DPRINT_SIZE=5000 -Wall -Wno-unused-value -Wno-unknown-pragmas  Nussinov.c -o Nussinov && ./Nussinov > printed-matrix-5000-seed17.out

*/

#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>

typedef int bool;
const int true = 1;
const int false = 0;

#if ! defined SCALAR_REPLACEMENT
#define SCALAR_REPLACEMENT 0
#endif

#if ! defined CHECK_DEPENDENCES
#define CHECK_DEPENDENCES false
#endif

#if defined NDEBUG
#define eassert(EXPR)   1
#else
#define eassert(EXPR)	eassert_func(__STRING(EXPR), EXPR)
void eassert_func(char *expr, bool value)
{
    if (!value) {
	fprintf(stderr, "assertion failed: %s\n", expr);
	exit(1);
	// printf("assertion failed: %s\n", expr);
    }
}
#endif


// NOTE THIS NEEDS TO EXIST HERE OTHERWISE MAX_SIZE UNDEFINED
#define MAX_SIZE 16307

#if ! defined FOUR_WAY_MAX_WITH_REDUNDANCY
#define FOUR_WAY_MAX_WITH_REDUNDANCY false
#endif
#if FOUR_WAY_MAX_WITH_REDUNDANCY
#define ZERO_IF_NO_REDUNDANCY 1
#else
#define ZERO_IF_NO_REDUNDANCY 0
#endif

#if ! defined PRINT_SIZE
#define PRINT_SIZE  48  /* print for debugging if not bigger than this */
#endif
#if ! defined VERBOSE
#define VERBOSE       false
#endif
#if ! defined VERBOSE_OUT
#define VERBOSE_OUT stdout
#endif
#if VERBOSE
#if ! defined REALLY_VERBOSE
#define REALLY_VERBOSE false
#endif
#endif

#if ! defined SEED
#define SEED 42
#endif

#define SLOWER (CHECK_DEPENDENCES | VERBOSE)

double cur_time(void)
{
	struct timeval tv;
	struct timezone tz;
	gettimeofday(&tv, &tz);
	return tv.tv_sec + tv.tv_usec*1.0e-6;
}

// for bases, use 0, 1, 2, 3, with (0, 3) and (1, 2) being matches
char *base_print[4] = { "A", "C", "G", "U" };
typedef int base;	// could also try char, short
typedef int score;	// could use uint, short, anything that can count high enough

inline score match(base b1, base b2)
{
	return (b1+b2) == 3 ? 1 : 0;
}

inline score max_score(score s1, score s2)
{
	return (s1 >= s2) ? s1 : s2;
}

base  seq[MAX_SIZE];
score   N_array[MAX_SIZE][MAX_SIZE];
#define debug_N(x, y)	(N_array[x][y])  /* read secretly without triggering has_been_read */

#if ! CHECK_DEPENDENCES
#define N(x, y) 	(eassert(0 <= x && x < size && 0 <= y && y < size), N_array[x][y])
#if SCALAR_REPLACEMENT
#define MAX_N_DECLS()	int max_tmp_i, max_tmp_j; score max_tmp
#define MAX_N_START(x,y)	((max_tmp_i=x), (max_tmp_j=y), (max_tmp = 0))
#define MAX_N(x, y, v)	(eassert(max_tmp_i==x && max_tmp_j==y), eassert(0 <= x && x < size && 0 <= y && y < size), (max_tmp = max_score(max_tmp, v)))
#define MAX_N_END(x,y)	(eassert(max_tmp_i==x && max_tmp_j==y), ((N_array[x][y]) = max_score(N_array[x][y], max_tmp)))
#else  /* else not scalar replacement (inside check_deps) */
#define MAX_N(x, y, v)	(eassert(0 <= x && x < size && 0 <= y && y < size), ((N_array[x][y]) = max_score(N_array[x][y], v)))
#endif /* else not scalar replacement (inside check_deps) */
#else  /* not check_deps */
bool    N_array_has_been_read[MAX_SIZE][MAX_SIZE];
#define N(x, y) 	(eassert(0 <= x && x < size && 0 <= y && y < size), \
			 (REALLY_VERBOSE?fprintf(VERBOSE_OUT, "i, j, k = %d, %d, %d: reading N[%d][%d]\n", i, j, k, x, y):1), \
			 (N_array_has_been_read[x][y] = (true)), \
			 N_array[x][y]+0)
#if SCALAR_REPLACEMENT  /* inside not check_deps */
#error("Not yet ready to do scalar replacement and check_deps at the same time :-(\00")
#else  /* else not scalar replacement (inside not check_deps) */
#define MAX_N(x, y, v)	(eassert(0 <= x && x < size && 0 <= y && y < size), \
			 eassert(!N_array_has_been_read[x][y]), \
			 (N_array[x][y] = max_score(N_array[x][y], v)))
#endif /* else not scalar replacement (inside not check_deps) */
#endif
#if ! SCALAR_REPLACEMENT
#define MAX_N_DECLS()
#define MAX_N_START(x,y)
#define MAX_N_END(x,y) 
#endif


/* Convenience function */
int getint(char *prompt)
{
#if VERBOSE_OUT == stderr
	char *terminate = "\n";
#else
	char *terminate = "";
#endif
	int result;
	int i=0;
	while (
		fprintf(stderr, "%s%s", prompt, terminate),
		result = scanf("%d", &i),
		result != 1 && result != EOF
	) {
		fprintf(stderr, "Sorry, I didn't understand that...\n");
	}
	if (result == 1) {
		return i;
	} else {
		fprintf(stderr, "Giving up ... can't read input\n");
		exit(1);
	}
}


int main(int argc, char *argv[])
{
#if ! SLOWER
	double start_time, end_time; // , speed;
#endif
#if ! defined size
 	int size = getint("Enter length of random mRNA sequence (2200 is average for human mRNA): ");  // Average (human) mRNA length is 2200; there's one that's 5000, though
#endif

        int i, j, k=-1;
	MAX_N_DECLS();
	char *options;
#if VERBOSE
#if CHECK_DEPENDENCES
	options = " [DV]";
#else
	options = " [V]";
#endif
#else
#if CHECK_DEPENDENCES
	options = " [D]";
#else
	options = "";
#endif
#endif

	printf("Running Nussinov RNA algorithm%s for sequence of length %d, random data with seed %d.\n",
	       options, size, SEED);

	if (size > MAX_SIZE) {
		fprintf(stderr, "size (%d) < MAX_SIZE (%d)\n", MAX_SIZE, size);
		exit(1);
	}

	/* seed it with a constant so we can check/compare the results */
	srand(SEED);
	for (i = 0; i < size; i++)
		seq[i] = rand() % 4;
	
#if ! SLOWER
	start_time = cur_time();
#endif

// "OPTION 1"
#pragma scop
	for (i = size-1; i >= 0; i--) {
		for (j=i+1; j<size; j++) {
#if VERBOSE
			fprintf(VERBOSE_OUT, "i, j, k = %d, %d, %d\n", i, j, k); /* outer k is -1 to indicate no k */
#endif
			MAX_N_START(i, j);
#if FOUR_WAY_MAX_WITH_REDUNDANCY
			if (j-1>=0)   MAX_N(i, j, N(i, j-1));
			if (i+1<size) MAX_N(i, j, N(i+1, j));
#endif
			if (j-1>=0 && i+1<size) {
			  if (i<j-1) MAX_N(i, j, N(i+1, j-1)+match(seq[i], seq[j]));  /* don't allow adjacent elements to bond */
			  else       MAX_N(i, j, N(i+1, j-1));
			}

			{
			int k;  /* local k to allow N macro to look at k and get -1 above and real k here */
			for (k=i+ZERO_IF_NO_REDUNDANCY; k<j; k++) {
#if VERBOSE
				fprintf(VERBOSE_OUT, "i, j, k = %d, %d, %d\n", i, j, k);
#endif
				MAX_N(i, j, N(i, k)+N(k+1, j));
			}
			} /* end of local k */
			MAX_N_END(i, j);
		}
	}
#pragma endscop

#if !SLOWER
	end_time = cur_time();
	printf("done.\nTime elapsed: %fs\n", end_time-start_time);
#endif
	printf("N(0, size-1) = %d\n", N(0, size-1));

	if (size <= PRINT_SIZE) {
		for (i=0; i<size; i++)
			printf("%3s ", base_print[seq[i]]);
		printf("\n");
		for(i = 0; i < size; i++) {
			for(j = 0; j < size; j++) printf("%3d ", debug_N(i, j));
			printf("\n");
		}
	}

	eassert(k == -1);

	return 0;
}

/*

Q: What is FOUR_WAY_MAX_WITH_REDUNDANCY?

A: The original description of the Nussinov algorithm (#1 below) is equivalent to the simpler version (#2 below, also Allie's thesis).
   (Note that all variants define N(i,j) for 1 <= i < j <= L; use N=0 when reading other elements.)

   This seems to have essentially zero performance impact, as it is in the O(L^2) rather than O(L^3) part of the code

----- 1. Original Nussinov -----

(as described on http://ultrastudio.org/en/Nussinov_algorithm)

N(i, j) = max(
     N(i+1, j  )
     N(i  , j-1)
     N(i+1, j-1) + w(i, j)
     max_{i< k<j}(N(i, k) + N(k+1, j)
)

Note that N(i  , j-1) is redundant, considering two cases after noting i<j from context, so i<=j-1:
Case 1: i = j-1, i.e., the maximal value for any column
	Here, N(i, j-1) is N(i, i), which is 0 and can be ignored

Case 2:	i < j-1
	here, N(i, j-1) is the same the max term's biggest k, i.e. k=j-1 gives N(i, j-1)+N(j-1+1, j) and the N(j,j) term is 0


----- 2. Simpler Variant (possibly known as Nussinov-Jacobsen?) -----

http://www.ibi.vu.nl/teaching/masters/prot_struc/2008/ps-lec12-2008.pdf

N(i, j) = max(
     N(i+1, j-1) + w(i, j)
     max_{i<=k<j} N(i, k) + N(k+1, j)
)

break out the "=" from i<=k:

N(i, j) = max(
     N(i+1, j-1) + w(i, j)
     max_{i=k<j}  N(i, k) + N(k+1, j)
     max_{i< k<j} N(i, k) + N(k+1, j)
)

rewrite k's as i's in i=k version, drop i<j due to redundancy with context

N(i, j) = max(
     N(i+1, j-1) + w(i, j)
     N(i, i) + N(i+1, j)
     max_{i< k<j} N(i, k) + N(k+1, j)
)

Note all N(i, i) are zero:

N(i, j) = max(
     N(i+1, j-1) + w(i, j)
     N(i+1, j)
     max_{i< k<j} N(i, k) + N(k+1, j)
)

See also http://www.pnas.org/content/77/11/6309.full.pdf

 */
