# Offload Adapter

The Offload adapter is an experimental adapter built on top of LLVM's liboffload
runtime.

The main purpose of the adapter is to drive development of liboffload, and
identify gaps that need addressed for full SYCL support.

The long-term goal of the project is to replace Unified Runtime with liboffload.

The adapter should be used with an upstream build of LLVM with the `offload`
runtime enabled. The in-tree version is unlikely to be usable, as there are
frequent breaking API changes.

## Building upstream LLVM with Offload enabled
The `liboffload` library will be built if the `offload` runtime is enabled. An
example CMake configuration is:
```sh
cmake -S llvm -B build -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$PWD/build/install
  -DLLVM_ENABLE_PROJECTS='clang;clang-tools-extra;lldb;lld;' \
  -DLLVM_ENABLE_RUNTIMES='offload;openmp'
```

Because Offload's API header is generated with tablegen at build time, the
runtime must be installed rather than just built so it gets copied into the
correct location. From either the top-level build or runtime build directory:
```sh
$ ninja install
```

## Building the adapter
Build UR/DPC++ with the following additional flags:
```
  -DUR_BUILD_ADAPTER_OFFLOAD=ON
  -DUR_OFFLOAD_INSTALL_DIR=<path to upstream llvm install>
```

## Testing the adapter
The adapter can be tested with UR CTS via the
`check-unified-runtime-conformance-offload` target.

It is also possible to run the SYCL E2E tests by configuring with
```sh
  -DSYCL_TEST_E2E_TARGETS=offload:gpu
  -DOFFLOAD_BUILD_TARGET=<build-target>
```
where `<build-target>` is one of `target-nvidia` or `target-amdgpu`, depending
on the available device.
