# Compile-time performance tests

Each test in this folder is a source file whose compilation is timed by
`../perf_compile.py`, called from the test's RUN lines. On the dashboard, every
timed compile appears as `sycl_compile <benchmark name>` (wall time) and
`sycl_compile <benchmark name>, CPU time`, grouped as `CompileTime/<file name>`.

| Test | Measures |
|---|---|
| `core_hpp.cpp` | An empty program including `<sycl/detail/core.hpp>`: device syntax-only check and full build. |
| `sycl_hpp.cpp` | An empty program including `<sycl/sycl.hpp>`: device syntax-only check and full build. |
| `single_task_overhead.cpp` | Device compile of 400 template-heavy `single_task`s, without and with `-DSUBMIT`. |

## `perf_compile.py`

```
python3 perf_compile.py --benchmark-name "<name>" [--reps N] [--warmup N] -- <compiler command...>
```

| Option | Default | Meaning |
|---|---|---|
| `--benchmark-name "<name>"` | (required) | Name of the result; also its dashboard history key. |
| `--reps N` | 3 | Timed compiles. |
| `--warmup N` | 1 | Untimed compiles before the timed ones (fill the file and disk caches). |
| `-- <command...>` | | The compiler command to time, run without a shell. |

For every timed compile it measures the wall time and the CPU time (user plus
system time of the whole compiler process tree, which is less sensitive to
machine load). It then prints two `PERF_RESULT` lines with unit `s` and the
median of each: `<name>` and `<name>, CPU time`. If any compile fails, it
shows the compiler output and exits nonzero, which fails the test.

The dashboard suite runs the same script, with `--warmup 0 --reps 1`, and does
its own iterations; it also does one untimed compile first.

## RUN lines

Only this form is allowed (plus the shared forms listed in `../README.md`):

```
// RUN: python3 %S/../perf_compile.py --benchmark-name "<name>" [--reps N] [--warmup N] -- %{build} <flags...> [-o %t<sfx>.out]
```

* `%{build}` is the normal E2E compile command for the test's source; append
  the flags of the variant to measure, e.g. `-fsycl-device-only -fsyntax-only`
  for a device syntax check, or `-o %t.out` for a full build.
* Do not prefix the line with `%{run-aux}`: compile-time measurements belong
  to the build phase of the test, like any other compilation.

Every file also needs exactly one line

```
// BENCHMARK-DESCRIPTION: <one sentence>
```

which is shown as the description of its group on the dashboard.

## Adding a compile-time test

1. Create `CompileTime/<name>.cpp` with the code whose compilation you want to
   measure. The file name becomes the dashboard group, so choose it carefully.
2. Add one `perf_compile.py` RUN line per variant to measure, each with a
   distinct `--benchmark-name`, e.g.

   ```
   // RUN: python3 %S/../perf_compile.py --benchmark-name "my_header.hpp, device syntax-only" -- %{build} -fsycl-device-only -fsyntax-only
   // RUN: python3 %S/../perf_compile.py --benchmark-name "my_header.hpp, full build" -- %{build} -o %t.out
   ```

   Start the name with what is compiled and end it with the variant.
3. Add `// BENCHMARK-DESCRIPTION: <one sentence>` and any `REQUIRES`,
   `UNSUPPORTED` or `XFAIL` lines the test needs.
4. Run it with lit (see `../README.md`) and check that every variant prints its
   two `PERF_RESULT` lines. Then check that the dashboard suite accepts it,
   e.g. `main.py <workdir> --sycl <build> --preset SYCLPerf --dry-run --list`.

Once results are on the dashboard, do not change the `--benchmark-name`
values: each rename starts a new history.
