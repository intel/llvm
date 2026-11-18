# SYCL performance tests

The tests in this directory measure the run time or the compile time of SYCL
features. They are used in two ways:

* as ordinary SYCL E2E tests. `lit.local.cfg` requires the
  `enable-perf-tests` feature, so lit runs them only when asked to:

  ```
  llvm-lit --param enable-perf-tests=True --param sycl_devices="level_zero:gpu" \
    <e2e build dir>/PerformanceTests
  ```

  Under lit, every test runs its full measurement and prints its results
  (add `-a` to see them, and `-D output_limit=<bytes>` for tests with many
  results).
* as the `SYCL Performance Tests` suite of the benchmark framework
  (`devops/scripts/benchmarks/benches/sycl_perf.py`; preset `SYCLPerf`, also
  part of `Full`). The suite reads the RUN lines of every test, builds and
  runs the tests itself (without lit) and publishes the results on the
  performance dashboard. To run it locally:

  ```
  devops/scripts/benchmarks/main.py <workdir> --sycl <DPC++ build or install> \
    --preset SYCLPerf --output-html local
  ```

## Layout

| Path | Contents |
|---|---|
| [`RunTime/`](RunTime/README.md) | Runtime tests, written with `perf_harness.hpp`. Dashboard name: `sycl_perf RunTime/<test>`. |
| [`CompileTime/`](CompileTime/README.md) | Compile-time tests, measured with `perf_compile.py`. Dashboard name: `sycl_compile <benchmark name>`, grouped as `CompileTime/<test>`. |
| `perf_harness.hpp` | The C++ harness used by the runtime tests. |
| `perf_compile.py` | The compile timer used by the compile-time tests. |
| `lit.local.cfg` | Requires `enable-perf-tests` for the whole directory. |

See the README in each folder for the options of its harness or script and
for how to add a new test.

## Result format

Both kinds of tests print one line per measured result; all other output is
ignored by the suite:

```
PERF_RESULT {"name":"<test>","variant":"<variant>","unit":"<unit>","value":<median>,"stddev":<sample stddev>,"min":<min>,"reps":<N>,"lower_is_better":true,...}
```

The keys up to `lower_is_better` are always present, in this order; later keys
(e.g. `batch`, `mad`) are informational. Runtime tests also print one
`PERF_INFO` line with the benchmark name, description, unit and variant.

## Rules shared by both folders

* **RUN lines.** The suite understands only a small subset of lit syntax,
  listed in each folder's README; anything else makes the suite fail with the
  offending file and line. In addition to those forms, both folders may use
  `// RUN: echo <text>` (skipped by the suite), `%if <feature> %{ ... %}`
  segments (dropped by the suite), trailing `\` continuations, and the
  `REQUIRES`, `UNSUPPORTED` and `XFAIL` directives with their `*-TRACKER`
  lines. `%t<sfx>` suffixes match `[A-Za-z0-9_.-]*`.
* **Names are dashboard history keys.** Test names, variants and
  `--benchmark-name` values identify a result's history; renaming one starts a
  new history. Keep them short, descriptive and stable. They may only contain
  `[A-Za-z0-9 _.,:;<>()=+/*&#-]` and be at most 120 characters long; the
  harness and `perf_compile.py` reject anything else.
* **No tuning environment.** Tests must not set `SYCL_*`, `UR_*` or `ZE_*`
  tuning variables; the suite also removes inherited compiler-related
  variables (`CPATH`, `SYCL_*`, ...) before compiling.
