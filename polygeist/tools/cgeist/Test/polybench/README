* * * * * * * * * * * * * * *
* PolyBench/C 4.2.1 (beta)  *
* * * * * * * * * * * * * * *

Copyright (c) 2011-2016 the Ohio State University.

Contact:
   Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
   Tomofumi Yuki <tomofumi.yuki@inria.fr>


PolyBench is a benchmark suite of 30 numerical computations with
static control flow, extracted from operations in various application
domains (linear algebra computations, image processing, physics
simulation, dynamic programming, statistics, etc.). PolyBench features
include:
- A single file, tunable at compile-time, used for the kernel
  instrumentation. It performs extra operations such as cache flushing
  before the kernel execution, and can set real-time scheduling to
  prevent OS interference.
- Non-null data initialization, and live-out data dump.
- Syntactic constructs to prevent any dead code elimination on the kernel.
- Parametric loop bounds in the kernels, for general-purpose implementation.
- Clear kernel marking, using pragma-based delimiters.


PolyBench is currently available in C and in Fortran:
- See PolyBench/C 4.2.1 for the C version
- See PolyBench/Fortran 1.0 for the Fortran version (based on PolyBench/C 3.2)

Available benchmarks (PolyBench/C 4.2.1)

Benchmark	Description
2mm		2 Matrix Multiplications (alpha * A * B * C + beta * D)
3mm		3 Matrix Multiplications ((A*B)*(C*D))
adi		Alternating Direction Implicit solver
atax		Matrix Transpose and Vector Multiplication
bicg		BiCG Sub Kernel of BiCGStab Linear Solver
cholesky	Cholesky Decomposition
correlation	Correlation Computation
covariance	Covariance Computation
deriche		Edge detection filter
doitgen		Multi-resolution analysis kernel (MADNESS)
durbin		Toeplitz system solver
fdtd-2d		2-D Finite Different Time Domain Kernel
gemm		Matrix-multiply C=alpha.A.B+beta.C
gemver		Vector Multiplication and Matrix Addition
gesummv		Scalar, Vector and Matrix Multiplication
gramschmidt	Gram-Schmidt decomposition
head-3d		Heat equation over 3D data domain
jacobi-1D	1-D Jacobi stencil computation
jacobi-2D	2-D Jacobi stencil computation
lu		LU decomposition
ludcmp		LU decomposition followed by Forward Substitution
mvt		Matrix Vector Product and Transpose
nussinov	Dynamic programming algorithm for sequence alignment
seidel		2-D Seidel stencil computation
symm		Symmetric matrix-multiply
syr2k		Symmetric rank-2k update
syrk		Symmetric rank-k update
trisolv		Triangular solver
trmm		Triangular matrix-multiply


See the end of the README for mailing lists, instructions to use
PolyBench, etc.

--------------------
* New in 4.2.1-beta:
--------------------
 - Fix a bug in PAPI support, introduced in 4.2
 - Support PAPI 5.4.x

-------------
* New in 4.2:
-------------
 - Fixed a bug in syr2k.
 - Changed the data initialization function of several benchmarks.
 - Minor updates in the documentation and PolyBench API.

-------------
* New in 4.1:
-------------
 - Added LICENSE.txt
 - Fixed minor issues with cholesky both in documentation and implementation.
   (Reported by François Gindraud)
 - Simplified the macros for switching between data types. Now users
   may specify DATA_TYPE_IS_XXX where XXX is one of FLOAT/DOUBLE/INT
   to change all macros associated with data types.

-------------
* New in 4.0a:
-------------
 - Fixed a bug in jacobi-1d (Reported by Sven Verdoolaege)

-------------
* New in 4.0:
-------------

This update includes many changes. Please see CHANGELOG for detailed
list of changes. Most of the benchmarks have been edited/modified by
Tomofumi Yuki, thanks to the feedback we have received by PolyBench
users for the past few years.

- Three benchmarks are out: dynprog, reg-detect, fdtd-apml.
- Three benchmarks are in: nussinov, deriche, heat-3d.
- Jacobi-1D and Jacobi-2D perform two time steps in one time loop
  iteration alternating the source and target fields, to avoid the
  field copy statement.
- Almost all benchmarks have been edited to ensure the computation
  result matches the mathematical specification of the operation.
- A major effort on documentation and harmonization of problem sizes
  and data allocations schemes.

* Important Note:
-----------------

PolyBench/C 3.2 kernels had numerous implementation errors making
their outputs to not match what is expected from the mathematical
specification of the operation. Many of them did not influence the
program behavior (e.g., the number and type of operations, data
dependences, and overall control-flow was similar to the corrected
implementation), however, some had non-negligible impact. These are
described below.

 - adi: There was an off-by-one error, which made back substitution
   part of a pass in ADI to not depend on the forward pass, making the
   program fully tilable.
- syrk: A typo on the loop bounds made the iteration space rectangular
   instead of triangular. This has led to additional dependences and
   two times more operations than intended.
- trmm: A typo on the loop bounds led to the wrong half of the matrix
   being used in the computation. This led to additional dependences,
   making it harder to parallelize this kernel.
- lu: An innermost loop was missing for the operation to be valid on
   general matrices. This cause the kernel to perform about half the
   work compared to a general implementation of LU decomposition. The
   new implementation is the generic LU decomposition.

In addition, some of the kernels used "high-footprint" memory allocation for
easier parallelization, where variables used in accumulation were fully
expanded. These variables were changed to only use a scalar.


-------------
* New in 3.2:
-------------

- Rename the package to PolyBench/C, to prepare for the upcoming
  PolyBench/Fortran and PolyBench/GPU.
- Fixed a typo in polybench.h, causing compilation problems for 5D arrays.
- Fixed minor typos in correlation, atax, cholesky, fdtd-2d.
- Added an option to build the test suite with constant loop bounds
  (default is parametric loop bounds)

-------------
* New in 3.1:
-------------

- Fixed a typo in polybench.h, causing compilation problems for 3D arrays.
- Set by default heap arrays, stack arrays are now optional.

-------------
* New in 3.0:
-------------

- Multiple dataset sizes are predefined. Each file comes now with a .h
  header file defining the dataset.
- Support of heap-allocated arrays. It uses a single malloc for the
  entire array region, the data allocated is cast into a C99
  multidimensional array.
- One benchmark is out: gauss_filter
- One benchmark is in: floyd-warshall
- PAPI support has been greatly improved; it also can report the
  counters on a specific core to be set by the user.



----------------
* Mailing lists:
----------------

** polybench-announces@lists.sourceforge.net:
---------------------------------------------

Announces about releases of PolyBench.

** polybench-discussion@lists.sourceforge.net:
----------------------------------------------

General discussions reg. PolyBench.



-----------------------
* Available benchmarks:
-----------------------

See utilities/benchmark_list for paths to each files.
See doc/polybench.pdf for detailed description of the algorithms.



------------------------------
* Sample compilation commands:
------------------------------

** To compile a benchmark without any monitoring:
-------------------------------------------------

$> gcc -I utilities -I linear-algebra/kernels/atax utilities/polybench.c linear-algebra/kernels/atax/atax.c -o atax_base


** To compile a benchmark with execution time reporting:
--------------------------------------------------------

$> gcc -O3 -I utilities -I linear-algebra/kernels/atax utilities/polybench.c linear-algebra/kernels/atax/atax.c -DPOLYBENCH_TIME -o atax_time


** To generate the reference output of a benchmark:
---------------------------------------------------

$> gcc -O0 -I utilities -I linear-algebra/kernels/atax utilities/polybench.c linear-algebra/kernels/atax/atax.c -DPOLYBENCH_DUMP_ARRAYS -o atax_ref
$> ./atax_ref 2>atax_ref.out



-------------------------
* Some available options:
-------------------------

They are all passed as macro definitions during compilation time (e.g,
-Dname_of_the_option).

** Typical options:
-------------------

- POLYBENCH_TIME: output execution time (gettimeofday) [default: off]

- MINI_DATASET, SMALL_DATASET, MEDIUM_DATASET, LARGE_DATASET,
  EXTRALARGE_DATASET: set the dataset size to be used
  [default: STANDARD_DATASET]

- POLYBENCH_DUMP_ARRAYS: dump all live-out arrays on stderr [default: off]

- POLYBENCH_STACK_ARRAYS: use stack allocation instead of malloc [default: off]


** Options that may lead to better performance:
-----------------------------------------------

- POLYBENCH_USE_RESTRICT: Use restrict keyword to allow compilers to
  assume absence of aliasing. [default: off]

- POLYBENCH_USE_SCALAR_LB: Use scalar loop bounds instead of parametric ones.
  [default: off]

- POLYBENCH_PADDING_FACTOR: Pad all dimensions of all arrays by this
  value [default: 0]

- POLYBENCH_INTER_ARRAY_PADDING_FACTOR: Offset the starting address of
  polybench arrays allocated on the heap (default) by a multiple of
  this value [default: 0]

- POLYBENCH_USE_C99_PROTO: Use standard C99 prototype for the functions.
  [default: off]


** Timing/profiling options:
----------------------------

- POLYBENCH_PAPI: turn on papi timing (see below).

- POLYBENCH_CACHE_SIZE_KB: cache size to flush, in kB [default: 33MB]

- POLYBENCH_NO_FLUSH_CACHE: don't flush the cache before calling the
  timer [default: flush the cache]

- POLYBENCH_CYCLE_ACCURATE_TIMER: Use Time Stamp Counter to monitor
  the execution time of the kernel [default: off]

- POLYBENCH_LINUX_FIFO_SCHEDULER: use FIFO real-time scheduler for the
  kernel execution, the program must be run as root, under linux only,
  and compiled with -lc [default: off]



---------------
* PAPI support:
---------------

** To compile a benchmark with PAPI support:
--------------------------------------------

$> gcc -O3 -I utilities -I linear-algebra/kernels/atax utilities/polybench.c linear-algebra/kernels/atax/atax.c -DPOLYBENCH_PAPI -lpapi -o atax_papi


** To specify which counter(s) to monitor:
------------------------------------------

Edit utilities/papi_counters.list, and add 1 line per event to
monitor. Each line (including the last one) must finish with a ',' and
both native and standard events are supported.

The whole kernel is run one time per counter (no multiplexing) and
there is no sampling being used for the counter value.



------------------------------
* Accurate performance timing:
------------------------------

With kernels that have an execution time in the orders of a few tens
of milliseconds, it is critical to validate any performance number by
repeating several times the experiment. A companion script is
available to perform reasonable performance measurement of a PolyBench.

$> gcc -O3 -I utilities -I linear-algebra/kernels/atax utilities/polybench.c linear-algebra/kernels/atax/atax.c -DPOLYBENCH_TIME -o atax_time
$> ./utilities/time_benchmark.sh ./atax_time

This script will run five times the benchmark (that must be a
PolyBench compiled with -DPOLYBENCH_TIME), eliminate the two extremal
times, and check that the deviation of the three remaining does not
exceed a given threshold, set to 5%.

It is also possible to use POLYBENCH_CYCLE_ACCURATE_TIMER to use the
Time Stamp Counter instead of gettimeofday() to monitor the number of
elapsed cycles.



----------------------------------------
* Generating macro-free benchmark suite:
----------------------------------------

(from the root of the archive:)
$> PARGS="-I utilities -DPOLYBENCH_TIME";
$> for i in `cat utilities/benchmark_list`; do perl utilities/create_cpped_version.pl $i "$PARGS"; done

This create for each benchmark file 'xxx.c' a new file
'xxx.preproc.c'. The PARGS variable in the above example can be set to
the desired configuration, for instance to create a full C99 version
(parametric arrays):

$> PARGS="-I utilities -DPOLYBENCH_USE_C99_PROTO";
$> for i in `cat utilities/benchmark_list`; do perl utilities/create_cpped_version.pl $i "$PARGS"; done



------------------
* Utility scripts:
------------------
create_cpped_version.pl: Used in the above for generating macro free version.

makefile-gen.pl: generates make files in each directory. Options are globally
                 configurable through config.mk at polybench root.
  header-gen.pl: refers to 'polybench.spec' file and generates header in
                 each directory. Allows default problem sizes and datatype to
                 be configured without going into each header file.

    run-all.pl: compiles and runs each kernel.
      clean.pl: runs make clean in each directory and then removes Makefile.
