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
In order to execute kernels compiled for `native-cpu`, we provide a PI Plugin. The plugin needs to be enabled when configuring DPC++ (e.g. `python buildbot/configure.py --native_cpu`) and needs to be selected at runtime by setting the environment variable `ONEAPI_DEVICE_SELECTOR=native_cpu:cpu`. 

# Supported features and current limitations

The SYCL Native CPU flow is still WIP, not optimized and several core SYCL features are currently unsupported. Currently `barrier` and several math builtins are not supported, and attempting to use those will most likely fail with an `undefined reference` error at link time. Examples of supported applications can be found in the [runtime tests](https://github.com/intel/llvm/blob/sycl/sycl/test/native_cpu).


To execute the `e2e` tests on the Native CPU, configure the test suite with:

```bash
# make sure that DPC++ is in your $PATH and your environment is configured for DPC++

cd sycl/test-e2e
cmake \
  -G Ninja \
  -B build -S . \
 -DCMAKE_CXX_COMPILER=clang++ \
 -DSYCL_TEST_E2E_TARGETS="native_cpu:cpu" 

```

Note that a number of `e2e` tests are currently still failing.

# Running example

The following section gives a brief overview of how a simple SYCL application is compiled for the Native CPU target. Consider the following SYCL sample, which performs vector addition using USM:

```c++
  cl::sycl::queue deviceQueue;
  cl::sycl::range<1> numOfItems{N};
  auto a_ptr = sycl::malloc_device<int>(N, deviceQueue);
  auto b_ptr = sycl::malloc_device<int>(N, deviceQueue);
  auto c_ptr = sycl::malloc_device<int>(N, deviceQueue);

  // copy mem to device, omitted
  deviceQueue
      .submit([&](cl::sycl::handler &cgh) {
        auto kern = [=](cl::sycl::id<1> wiID) {
          c_ptr[wiID] = a_ptr[wiID] + b_ptr[wiID];
        };
        cgh.parallel_for<class Sample>(numOfItems, kern);
      })
      .wait();
  deviceQueue.memcpy(C.data(), c_ptr, N * sizeof(int));

```

The extracted device code produces the following LLVM-IR:

```llvm
define weak_odr dso_local spir_kernel void @_Z6Sample(ptr noundef align 4 %_arg_c_ptr, ptr noundef align 4 %_arg_a_ptr, ptr noundef align 4 %_arg_b_ptr) local_unnamed_addr #1 comdat !srcloc !74 !kernel_arg_buffer_location !75 !kernel_arg_type !76 !sycl_fixed_targets !49 !sycl_kernel_omit_args !77 {
entry:
  %0 = load i64, ptr @__spirv_BuiltInGlobalInvocationId, align 32, !noalias !78
  %arrayidx.i = getelementptr inbounds i32, ptr %_arg_a_ptr, i64 %0
  %1 = load i32, ptr %arrayidx.i, align 4, !tbaa !72
  %arrayidx4.i = getelementptr inbounds i32, ptr %_arg_b_ptr, i64 %0
  %2 = load i32, ptr %arrayidx4.i, align 4, !tbaa !72
  %add.i = add nsw i32 %1, %2
  %cmp.i8.i = icmp ult i64 %0, 2147483648
  tail call void @llvm.assume(i1 %cmp.i8.i)
  %arrayidx6.i = getelementptr inbounds i32, ptr %_arg_c_ptr, i64 %0
  store i32 %add.i, ptr %arrayidx6.i, align 4, !tbaa !72
  ret void
}
```

For the Native CPU target, the device compiler is in charge of materializing the SPIRV builtins (such as `@__spirv_BuiltInGlobalInvocationId`), so that they can be correctly updated by the runtime when executing the kernel. This is performed by the [PrepareSYCLNativeCPU pass](https://github.com/intel/llvm/blob/sycl/llvm/lib/SYCLLowerIR/PrepareSYCLNativeCPU.cpp).
The PrepareSYCLNativeCPUPass also emits a `subhandler` function, which receives the kernel arguments from the SYCL runtime (packed in a vector), unpacks them, and forwards only the used ones to the actual kernel. 


## PrepareSYCLNativeCPU Pass

This pass will add a pointer to a `nativecpu_state` struct as kernel argument to all the kernel functions, and it will replace all the uses of SPIRV builtins with the return value of appropriately defined functions, which will read the requested information from the `__nativecpu_state` struct. The `__nativecpu_state` struct and the builtin functions are defined in [native_cpu.hpp](https://github.com/intel/llvm/blob/sycl/sycl/include/sycl/detail/native_cpu.hpp).


The resulting IR is:

```llvm
define weak dso_local void @_Z6Sample.NativeCPUKernel(ptr noundef align 4 %0, ptr noundef align 4 %1, ptr noundef align 4 %2, ptr %3) local_unnamed_addr #3 !srcloc !74 !kernel_arg_buffer_location !75 !kernel_arg_type !76 !sycl_fixed_targets !49 !sycl_kernel_omit_args !77 {
entry:
  %ncpu_builtin = call ptr @_Z13get_global_idmP15nativecpu_state(ptr %3)
  %4 = load i64, ptr %ncpu_builtin, align 32, !noalias !78
  %arrayidx.i = getelementptr inbounds i32, ptr %1, i64 %4
  %5 = load i32, ptr %arrayidx.i, align 4, !tbaa !72
  %arrayidx4.i = getelementptr inbounds i32, ptr %2, i64 %4
  %6 = load i32, ptr %arrayidx4.i, align 4, !tbaa !72
  %add.i = add nsw i32 %5, %6
  %cmp.i8.i = icmp ult i64 %4, 2147483648
  tail call void @llvm.assume(i1 %cmp.i8.i)
  %arrayidx6.i = getelementptr inbounds i32, ptr %0, i64 %4
  store i32 %add.i, ptr %arrayidx6.i, align 4, !tbaa !72
  ret void
}
```
This pass will also set the correct calling convention for the target, and handle calling convention-related function attributes, allowing to call the kernel from the runtime.

The `subhandler` for the Native CPU kernel looks like: 

```llvm
define weak void @_Z6Sample(ptr %0, ptr %1) #4 {
entry:
  %2 = getelementptr %0, ptr %0, i64 0
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr %0, ptr %0, i64 3
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr %0, ptr %0, i64 4
  %7 = load ptr, ptr %6, align 8
  %8 = getelementptr %0, ptr %0, i64 7
  %9 = load ptr, ptr %8, align 8
  call void @_ZTS10SimpleVaddIiE.NativeCPUKernel(ptr %3, ptr %5, ptr %7, ptr %9, ptr %1)
  ret void
}
```
As you can see, the `subhandler` steals the kernel's function name, and receives two pointer arguments: the first one points to the kernel arguments from the SYCL runtime, and the second one to the `__nativecpu_state` struct.

## Kernel registration

In order to register the Native CPU kernels to the SYCL runtime, we applied a small change to the `clang-offload-wrapper` tool: normally, the `clang-offload-wrapper` bundles the offload binary in an LLVM-IR module. Instead of bundling the device code, for the Native CPU target we insert an array of function pointers to the `subhandler`s, and the `pi_device_binary_struct::BinaryStart` and `pi_device_binary_struct::BinaryEnd` fields, which normally point to the begin and end addresses of the offload binary, now point to the begin and end of the array.

```
 -------------------------------------------------------
 | "_Z6Sample"   | other entries  |  "__nativecpu_end" |
 | &_Z6Sample    |                |  nullptr           |
 -------------------------------------------------------
        ^                                   ^    
        |                                   |
    BinaryStart                         BinaryEnd  
```

Each entry in the array contains the kernel name as a string, and a pointer to the `sunhandler` function declaration. Since the subhandler's signature has always the same arguments (two pointers in LLVM-IR), the `clang-offload-wrapper` can emit the function declarations given just the function names contained in the `.table` file emitted by `sycl-post-link`. The symbols are then resolved by the system's linker, which receives both the output from the offload wrapper and the lowered device module.

## Kernel lowering and execution

The information produced by the device compiler is then employed to correctly lower the kernel LLVM-IR module to the target ISA (this is performed by the driver when `-fsycl-targets=native_cpu` is set). The object file containing the kernel code is linked with the host object file (and libsycl and any other needed library) and the final executable is ran using the Native CPU PI Plug-in, defined in [pi_native_cpu.cpp](https://github.com/intel/llvm/blob/sycl/sycl/plugins/native_cpu/pi_native_cpu.cpp).

## Ongoing work

* Complete support for remaining SYCL features, including but not limited to
  * kernels with barriers
  * math and other builtins
  * work group local memory
* Vectorization (e.g. Whole Function Vectorization)
* Subgroup support
* Performance optimizations
* Support for multiple SYCL targets alongside native_cpu

### Please note that Windows support is temporarily disabled due to some implementation details, it will be reinstantiated soon.
