# SYCL Native CPU

The SYCL Native CPU flow aims at treating the host CPU as a "first class citizen", providing a SYCL implementation that targets CPUs of various different architectures, with no other dependencies than DPC++ itself, while bringing performances comparable to state-of-the-art CPU backends.

# Compiler and runtime options

The SYCL Native CPU flow is enabled by the `-fsycl-native-cpu` compiler option (please note that currently `-fsycl-native-cpu` overrides any other SYCL target specified in the compiler invocation):

```
clang++ -fsycl -fsycl-native-cpu <input> -o <output>
```

This will perform automatically all the compilation stages. It is also possible to manually perform all the necessary compiler invocations, this is more verbose but allows the user to use an arbitrary host compiler for the second compilation stage:

```
#device compiler
clang++ -fsycl-device-only -fsycl-native-cpu -Xclang -fsycl-int-header=<integration-header> \
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

Our implementation currently piggybacks on the original (library-only) SYCL Host Device, therefore in order to run an application compiled with `-fsycl-native-cpu`, you need to set the environment variable `ONEAPI_DEVICE_SELECTOR=host:*` to make sure that the SYCL runtime chooses the host device to execute the application.

# Supported features and limitations

The SYCL Native CPU flow is still WIP, not optimized and several core SYCL features are currently unsupported. Currently only `parallel_for`s over `sycl::range` are supported, attempting to use `local_size`, `local_id`, `barrier` and any math builtin will most likely fail with an `undefined reference` error at link time. Examples of supported applications can be found in the [runtime tests](sycl/test/native_cpu).


