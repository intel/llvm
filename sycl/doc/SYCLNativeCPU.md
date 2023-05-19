# SYCL Native CPU

The SYCL Native CPU flow aims at treating the host CPU as a "first class citizen", providing a SYCL implementation that targets CPUs of various different architectures, with no other dependencies than DPC++ itself, while bringing performances comparable to state-of-the-art CPU backends.

# Compiler and runtime options

The SYCL Native CPU flow is enabled by setting `native_cpu` as a `sycl-target` (please note that currently doing so overrides any other SYCL target specified in the compiler invocation):

```
clang++ -fsycl -fsycl-targets=native_cpu <input> -o <output>
```

This will perform automatically all the compilation stages. It is also possible to manually perform all the necessary compiler invocations, this is more verbose but allows the user to use an arbitrary host compiler for the second compilation stage:

```
#device compiler
clang++ -fsycl-device-only -fsycl-targets=native_cpu -Xclang -fsycl-int-header=<integration-header> \
  -D __SYCL_NATIVE_CPU__ \
  -Xclang -fsycl-int-footer=<integration-footer> <input> -o <device-ir>
#host compiler
clang++ -fsycl-is-host -include <integration-header> \
  -D __SYCL_NATIVE_CPU__ \
  -include <integration-footer> \
  <intput> -c -o <host-o>
#compile device IR
clang++ <device-ir> -o <device-o>
#link
clang++ -L<sycl-lib-path> -lsycl <device-o> <host-o> -o <output>
```
In order to execute kernels compiled for `native-cpu`, we provide a PI Plugin. The plugin needs to be enabled when configuring DPC++ (e.g. `python buildbot/configure.py --enable-plugin native_cpu`) and needs to be selected at runtime by setting the environment variable `ONEAPI_DEVICE_SELECTOR=native_cpu:cpu`. 

# Supported features and limitations

The SYCL Native CPU flow is still WIP, not optimized and several core SYCL features are currently unsupported. Currently `barrier` and all the math builtins are not supported, and attempting to use those will most likely fail with an `undefined reference` error at link time. Examples of supported applications can be found in the [runtime tests](sycl/test/native_cpu).
To execute `e2e` tests on the Native CPU, configure the test suite with:

```bash
# make sure that DPC++ is in your $PATH and your environment is configured for DPC++

cd sycl/test-e2e
cmake \
  -G Ninja \
  -B build -S . \
 -DCMAKE_CXX_COMPILER=clang++ \
 -DSYCL_TEST_E2E_TARGETS="native_cpu:cpu" 

```



