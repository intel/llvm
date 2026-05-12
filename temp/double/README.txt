# SYCL source-kernel repro for `double` in a by-value struct

This directory contains a standalone C++ SYCL repro for the failure seen in the `device_repr` example.

What it does:
- Builds a source kernel where `MyCoolStruct` contains `unsigned long long`.
- Builds the same source pattern again with `double` in the same struct field.
- Uses the same SYCL source-kernel flow as the Rust bindings: `create_kernel_bundle_from_source` -> `build` -> `ext_oneapi_get_kernel`.

Expected behavior on the affected toolchain:
- The `unsigned long long` variant resolves the kernel by name.
- The `double` variant builds, but `ext_oneapi_has_kernel` is false and `ext_oneapi_get_kernel` fails.

Run it with:

```bash
./run_source_kernel_double_repro.sh
```

If oneAPI is not already on `PATH`, the script defaults to `/home/test-user/oneapi_2025.3.2.21` and sources `setvars.sh`. Override that with `ONEAPI_ROOT=/path/to/oneapi` if needed.