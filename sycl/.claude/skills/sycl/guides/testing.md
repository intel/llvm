# SYCL tests: add, extend, fix, review

Covers `sycl/test` (LIT), `sycl/test-e2e` (E2E), `sycl/unittests` (gtest). Siblings and
headers show the mechanics; this guide covers tier choice, process and traps only.

## Tier

| Claim | Tier |
|---|---|
| Runtime results that need device execution: kernel output, data movement, backend behaviour, interop, a `__SYCL_DEVICE_ONLY__` branch's behaviour (its emitted IR is LIT) | **E2E** `sycl/test-e2e/` — only tier that runs device code |
| Runtime → backend behaviour: UR calls, flags, wait-lists, caching, lifetime, injected UR errors, `errc`, env/config, threads | **Unit** `sycl/unittests/` — gtest + `UrMock`, no hardware, no kernel results |
| Compile-time: `static_assert`, traits, diagnostics, device IR, warnings, ABI, feature macros | **LIT** `sycl/test/` — never runs device code |

## Process

1. **Search first.** One grep of the tier's tree for the API/UR entry/operator. Coverage
   exists → say so and stop, or add one case per new code path to that file.
2. **Write.** Copy the nearest sibling's shape. 1–3 line comment: what is checked and why,
   issue link if any. No license header. Shared rules: `../SKILL.md`.
3. **Prove it can fail.** Run, flip an expected value (or break the callback / CHECK), confirm
   failure, restore.
4. **Run** (`<build>` = DPC++ build dir, from repo root): unit `ninja -C <build> check-sycl-unittests`,
   or one suite `check-sycl-<Target>` (e.g. `check-sycl-QueueTests`); not the raw binary, the target
   sets the env (fresh `libsycl`, mock OpenCL on `LD_LIBRARY_PATH`, `SYCL_CONFIG_FILE_NAME`) · LIT `<build>/bin/llvm-lit -v sycl/test/<path>.cpp` ·
   E2E `<build>/bin/llvm-lit -v --param sycl_devices="level_zero:gpu" sycl/test-e2e/<path>.cpp`.
5. **Report**: tier and why; command, result, mutation check fired; required gates.

## Traps

**E2E**
- Use `<sycl/detail/core.hpp>` + fine-grained headers, instead of <sycl/sycl.hpp>
- `XFAIL:` → next line `// XFAIL-TRACKER: <GitHub issue URL | PROJ-123>`. `UNSUPPORTED:` (incl.
  `true`) → `// UNSUPPORTED-TRACKER: <id>` or `// UNSUPPORTED-INTENDED: <reason>`. Flaky ⇒ UNSUPPORTED.
- `%{build}` sets `-fsycl-targets` and `-Werror`; `%{run}` runs per device via
  `ONEAPI_DEVICE_SELECTOR`: `sycl::queue Q;`, never a hard-coded selector. Prefer `REQUIRES: aspect-<x>`
  or runtime `has(aspect::x)` over `REQUIRES: gpu`. Compile gates: `target-*`; runtime XFAIL: `run-mode`.
- Unique `%t<name>.out` per build line; float tolerance (`ulp_utils.hpp`, `DeviceLib/math_utils.hpp`);
  `wait()` before reading USM; `assert`/nonzero return, not `exit(1)`.
- Leak check: `%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK`.
- Build and run are separate, maybe different machines/OSes: no absolute paths. Compiler may be
  `clang-cl` or `clang++`: flags via substitutions (`%O0`, `%debug_option`, `%fPIC`, `%shared_lib`,
  `%if cl_options %{...%}`), no raw GCC flags. No OS-specific shell in RUN lines; if unavoidable,
  `%if linux`/`%if windows`, non-binary run-stage steps under `%{run-aux}`. Current substitutions
  and features: `sycl/test-e2e/lit.cfg.py` and `format.py`, not memory.
- Graph: body in `Graph/Inputs/<name>.cpp` (`graph_common.hpp`); wrappers in `Explicit/` and
  `RecordReplay/` define `GRAPH_E2E_EXPLICIT` / `GRAPH_E2E_RECORD_REPLAY` and include it.

**Unit**
- `sycl::unittest::UrMock<> Mock;` first, one per test, before any runtime object that reaches UR:
  it installs the mock adapter and default platform/device. Callbacks only to override/inspect calls.
- Mock defaults (`urDeviceGetInfo`, `urDeviceGet`, ...) are `replace` callbacks: your bare `replace`
  discards them. Fixed-size patch → `set_after_callback`; strings/size changes → `replace` and forward
  the rest to `sycl::unittest::MockAdapter::mock_urDeviceGetInfo(pParams)`. Queries are two-phase:
  set `**P.ppPropSizeRet` when `*P.ppPropValue` is null, copy otherwise.
- `~UrMock` resets callbacks, not statics: reset counters per test or use a per-test struct.
- `OBJECT` linking exposes `<detail/*.hpp>` internals. Kernels: `<helpers/TestKernel.hpp>`, or
  `MOCK_INTEGRATION_HEADER(K)` + `MockDeviceImage[Array]` (`SYCL2020/KernelBundle.cpp`).

**LIT**
- IR: `-fsycl-device-only -S -emit-llvm`, output `-o -` or `%t` (literal paths race).
- Check one property with `{{.*}}`/`[[VAR:...]]` on a `SYCL_EXTERNAL` function; builtins are mangled
  (`@{{.*}}__spirv_AtomicLoad{{.*}}(`). Full-body checks: `llvm/utils/update_cc_test_checks.py --clang <build>/bin/clang++`.
- `-Xclang -verify` + `-verify-ignore-unexpected=note,warning`; positive files `// expected-no-diagnostics`;
  header diagnostics `@*:*`. `%fsycl-host-only` has no `-fsycl`: device diagnostics won't fire.
- `REQUIRES: linux` + `UNSUPPORTED: libcxx` only when the test depends on libstdc++ layout/mangling
  (`abi/`, `gdb/`) or a host toolchain (`-fsycl-host-compiler=g++`).

## Review

Flag: wrong tier; vacuous (cannot fail, no mutation evidence); redundant with a sibling; any trap
above; noise (license header, verbose or missing comments, `cl::sycl::`, unused includes, absolute
paths, Intel-internal links, unrequested JIRA ids, unformatted). Else approve with tier justification.
