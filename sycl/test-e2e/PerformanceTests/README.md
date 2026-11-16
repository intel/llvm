# SYCL performance tests

The tests in this directory measure the run time or compile time of SYCL
features. They are used in two ways:

* as ordinary SYCL E2E tests. `lit.local.cfg` requires the
  `enable-perf-tests` feature, so lit runs them only when asked to:

  ```
  llvm-lit --param enable-perf-tests=True --param sycl_devices="level_zero:gpu" \
    <e2e build dir>/PerformanceTests
  ```

  Under lit, every test runs its full measurement and prints the results.
* as the `SYCL Performance Tests` suite of the benchmark framework
  (`devops/scripts/benchmarks/benches/sycl_perf.py`, preset `SYCLPerf`,
  also part of `Full`). The suite reads the RUN lines of each test, builds and
  runs the tests itself (without lit), and publishes the results on the
  performance dashboard.

## Result format

Every measured case prints one line to stdout:

```
PERF_RESULT {"name":"<case>","variant":"<variant>","unit":"<unit>","value":<median>,"stddev":<sample stddev>,"min":<min>,"reps":<N>,"lower_is_better":true}
```

All keys are always present. `value`, `stddev` and `min` are computed over the
timed repetitions (warm-ups are excluded). Other output is ignored by the
suite.

## Runtime tests: `perf_harness.hpp`

A header-only, standard-library-only C++17 harness:

```c++
#include "../perf_harness.hpp"

int main(int argc, char **argv) {
  perf::add_case("my op, 100 calls", "ns", [&](perf::State &S) {
    S.items(100);                 // report time per call
    S.measure([&] {               // time one repetition
      for (int I = 0; I < 100; ++I)
        my_op();
    });
    perf::check(result_ok(), "result"); // throws on failure
  }, perf::Options{/*warmup=*/1, /*reps=*/10});
  return perf::main(argc, argv);
}
```

* The case function is called once per warm-up and once per timed repetition
  and must call exactly one of `S.measure(F)` (time `F` with
  `std::chrono::steady_clock`; the unit must be `s`, `ms`, `us` or `ns`) or
  `S.manual(V)` (a value the test measured itself, in the case's unit).
  `S.items(K)` divides the recorded value by `K`. `S.is_warmup()` and
  `S.iteration()` (0-based, counting warm-ups) are available for per-repetition
  setup.
* Defaults are 1 warm-up and 10 repetitions; `perf::Options` overrides them
  per case and the command line overrides both.
* A case fails if it throws, e.g. from `perf::check()`. Failed cases print no
  result, and the process exits nonzero. Keep a cheap correctness check where
  it is meaningful.
* Command line handled by `perf::main()`:
  * `--list`: print the case names, one per line, and exit;
  * `--case=<name>`: run only this case (exact match, repeatable);
  * `--reps=N`, `--warmup=N`;
  * `--variant=<text>`: copied to the `variant` key, used to tell build
    variants of the same source apart;
  * `--<key>=<value>`: test-specific, read with `perf::arg("key")` (throws if
    missing) or `perf::arg("key", "default")` inside case functions.

  Any other argument is an error.

## Compile-time tests: `perf_compile.py`

```
python3 perf_compile.py --name "<name>" [--reps N] [--warmup N] -- <compiler command...>
```

Runs the command `warmup + reps` times (defaults 1 and 3) and prints two
results with unit `s`: `<name>` (wall time) and `<name>, CPU time` (user + sys
time of the whole compiler process tree). It fails, showing the compiler
output, if any compile fails.

## RUN lines

The suite understands only this subset of lit syntax; anything else makes it
fail. Only use:

```
// RUN: %{build} <flags...> -o %t<sfx>.out
// RUN: %{run} %t<sfx>.out [args...]
// RUN: python3 %S/../perf_compile.py --name "<name>" [--reps N] [--warmup N] -- %{build} <flags...> [-o %t<sfx>.out]
```

* `<sfx>` matches `[A-Za-z0-9_.-]*`. Run arguments may use `%t`, `%/t`,
  `--variant="<text>"` and `--key=value`; quote values containing spaces with
  double quotes.
* `perf_compile.py` lines are not prefixed with `%{run-aux}`; adjust the
  relative path to the test's location.
* `%if <feature> %{ ... %}` segments and trailing `\` continuations are
  allowed. The suite drops `%if` segments.
* Other directives: `REQUIRES`, `UNSUPPORTED`, `XFAIL` and their `*-TRACKER`
  lines.

Each test must also have exactly one line

```
// PERF-DESCRIPTION: <one sentence>
```

which is shown as the description on the dashboard.

## Names

Case names, variants and `--name` values are dashboard history keys: renaming
one starts a new history. Keep them short, descriptive and stable, e.g.
`has(fp64)`, `range<2> 1888x788, 1000 submits`,
`reduce_over_group sub-group int32`, `sycl.hpp, device syntax-only`. They may
only contain `[A-Za-z0-9 _.,:;<>()=+/*&#-]` and be at most 120 characters
long; the harness and `perf_compile.py` reject anything else.
