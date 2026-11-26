# Runtime performance tests

Each test in this folder measures how long a SYCL operation takes at run time,
using the header-only harness `../perf_harness.hpp` (C++17, standard library
only). On the dashboard, a test appears as `sycl_perf RunTime/<file name>` and
each of its measured operations as `sycl_perf RunTime/<file name> <test name>`.

| Test | Measures |
|---|---|
| `aspect_queries.cpp` | One `sycl::device::has()` call, for every aspect (ns). |
| `parallel_for_range_roundup.cpp` | 1000 submits of a 2D `parallel_for`, per range-rounding mode (ms). |
| `reduce_over_sub_group.cpp` | 100000 sub-group `reduce_over_group` calls, per type (ms). |
| `auto-pch.cpp` | `kernel_compiler` build time, per extra header and PCH mode (ms). |

## Harness

A test creates one `perf::Benchmark` in its own `main()`, does its setup with
plain code, calls `add_perf_test()` for every operation to measure, then
verifies and cleans up with plain code. There is no global state and no
harness-provided `main()`.

```c++
#include "../perf_harness.hpp"

int main(int argc, char **argv) {
  perf::Benchmark Bench(argc, argv, "My benchmark",
                        "What this test measures.", perf::Unit::us);
  sycl::queue Q;                                    // setup, not timed
  int *Out = sycl::malloc_shared<int>(1, Q);
  Bench.add_perf_test("single_task + wait", [&] {   // measured immediately
    Q.single_task([=] { *Out = 42; }).wait();
  });
  perf::check(*Out == 42, "kernel wrote the result"); // throws on failure
  sycl::free(Out, Q);                               // cleanup
  return 0;
}
```

### `perf::Benchmark`

`perf::Benchmark <id>(argc, argv, "<name>", "<description>", perf::Unit::<u>);`

* `<name>` is the benchmark's display name and `<description>` its dashboard
  description; `<u>` is `s`, `ms`, `us` or `ns` and applies to every test of
  the benchmark.
* The suite reads these arguments from the source without building the test,
  so the declaration must use exactly this form, with plain string literals
  (adjacent literals are fine, escape sequences are not). Use exactly one
  `perf::Benchmark` per file.
* The constructor parses the command line (see below) and prints a
  `PERF_INFO` line.
* `Bench.arg("key")` returns the value of a `--key=value` argument (throws if
  it is missing); `Bench.arg("key", "default")` returns a default instead.

### `add_perf_test(name, fn[, opts])`

Measures `fn` immediately and prints one `PERF_RESULT` line:

1. **Calibration** (counts as the first warm-up): `fn` is run in batches of
   growing size until one batch takes at least `min_sample_time`. The final
   batch size `K` (`batch` in the output) is used for every sample, so very
   short operations are timed accurately.
2. **Warm-up**: `warmup - 1` further untimed batches.
3. **Sampling**: every sample is the time of one batch divided by `K`, in the
   benchmark's unit. Sampling stops when at least `reps` samples and
   `min_time` seconds of sampling are reached, or after `max_reps` samples.
4. **Result**: median (`value`), sample standard deviation, minimum and median
   absolute deviation (`mad`) of the samples.

`fn` may be called many times, so it must be repeatable; keep setup and
verification outside it. Exceptions thrown by `fn` propagate and fail the
test.

`perf::Opts` overrides the defaults per test (all setters are optional and
chainable):

| Setter | Default | Meaning |
|---|---|---|
| `.warmup(N)` | 1 | Untimed batches, including calibration. |
| `.reps(N)` | 10 | Minimum number of samples. |
| `.min_time(s)` | 0.1 | Minimum sampling time in seconds; use `0` for slow operations. |
| `.max_reps(N)` | 1000 | Maximum number of samples. |
| `.min_sample_time(s)` | 0.001 | Calibration target per sample, in seconds. |

For example, `perf::Opts().warmup(1).reps(4).min_time(0)` runs a slow
operation exactly five times and reports the last four.

### Helpers

* `perf::check(cond, "message")` throws (and so fails the test) if `cond` is
  false. Keep a cheap correctness check where it is meaningful.
* `perf::do_not_optimize(value)` keeps a computed value alive, so the compiler
  cannot drop the code that produced it.

### Command line

| Option | Meaning |
|---|---|
| `--list` | Print the test names, one per line, without measuring. |
| `--case=<name>` | Measure only this test (exact name; repeatable). Unknown names are an error. |
| `--reps=N` | Exactly `N` samples (ignores `min_time`). |
| `--warmup=N` | Number of warm-up batches (at least the calibration runs). |
| `--min-time=<seconds>` | Override `min_time`. |
| `--variant=<text>` | Copied to the `variant` key, to tell build variants of one source apart. |
| `--<key>=<value>` | Test-specific, read with `Bench.arg("key")`. |

Any other argument, or a malformed value, exits with status 2.

## RUN lines

Only these forms are allowed (plus the shared forms listed in
`../README.md`):

```
// RUN: %{build} <flags...> -o %t<sfx>.out
// RUN: %{run} %t<sfx>.out [--variant="<text>"] [--<key>=<value>...]
```

Flags and arguments may use `%t`, `%/t` and shell quoting, e.g.
`'-DPCH_DIR="%/t.dir"'`. If a test is built and run several times, e.g. with
different compiler flags, give every `%{run}` line a distinct
`--variant="<text>"`; the variant becomes part of the dashboard name.

## Adding a runtime test

1. Create `RunTime/<name>.cpp`. The file name is part of the dashboard name,
   so choose it carefully.
2. Add the RUN lines, e.g.

   ```
   // RUN: %{build} -o %t.out
   // RUN: %{run} %t.out
   ```

   plus any `REQUIRES`, `UNSUPPORTED` or `XFAIL` lines the test needs.
3. Include `"../perf_harness.hpp"` and the SYCL headers the test uses (prefer
   `<sycl/detail/core.hpp>` and fine-grained headers).
4. In `main(int argc, char **argv)`, declare the `perf::Benchmark` with a
   display name, a one-sentence description and the unit of all its results.
5. Do the setup (queues, allocations, input data) with plain code.
6. Call `Bench.add_perf_test("<test name>", [&] { ... })` for each operation
   to measure; the lambda contains only the code to time. Pass `perf::Opts`
   for slow operations (e.g. `.reps(5).min_time(0)`).
7. Verify the results with `perf::check()` where it is cheap, clean up, and
   return 0.
8. Run it with lit (see `../README.md`), and check that the `PERF_RESULT`
   lines look sensible. Then check that the dashboard suite accepts it, e.g.
   `main.py <workdir> --sycl <build> --preset SYCLPerf --dry-run --list`.

Once results are on the dashboard, do not rename the file, the test names or
the variants: each rename starts a new history.
