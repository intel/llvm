# Intel "Explicit SIMD" SYCL extension

OneAPI provides the "Explicit SIMD" SYCL extension (or simply "ESIMD") for
lower-level Intel GPU programming. It provides APIs closely matching Intel GPU ISA
yet allows to write explicitly vectorized device code. This helps programmer to
have more control over the generated code and depend less on compiler
optimizations. The [specification](sycl_ext_intel_esimd.md),
[API reference](https://intel.github.io/llvm-docs/doxygen/group__sycl__esimd.html), and
[working code examples](https://github.com/intel/llvm/blob/sycl/sycl/test-e2e/ESIMD/) are available on the Intel DPC++ project's github.

**_NOTE:_** _Some parts of this extension is under active development and APIs in the
`sycl::ext::intel::experimental::esimd` package are subject to change. There are
currently a number of [restrictions](#restrictions) specified below._

ESIMD kernels and functions always require the subgroup size of one, which means
compiler never does vectorization across work-items in a subgroup. Instead,
vectorization is expressed explicitly in the code by the programmer. Here is a
trivial example which adds elements of two arrays and writes the results to the
third:

```cpp
    float *A = malloc_shared<float>(Size, q);
    float *B = malloc_shared<float>(Size, q);
    float *C = malloc_shared<float>(Size, q);

    for (unsigned i = 0; i != Size; i++) {
      A[i] = B[i] = i;
    }

    q.submit([&](handler &cgh) {
      cgh.parallel_for<class Test>(
        Size / VL, [=](id<1> i)[[intel::sycl_explicit_simd]] {
        auto offset = i * VL;
        // pointer arithmetic, so offset is in elements:
        simd<float, VL> va(A + offset);
        simd<float, VL> vb(B + offset);
        simd<float, VL> vc = va + vb;
        vc.copy_to(C + offset);
      });
    }).wait_and_throw();
```

In this example the lambda function passed to the `parallel_for` is marked with
a special attribute - `[[intel::sycl_explicit_simd]]`. This tells the compiler that
this kernel is a ESIMD one and ESIMD APIs can be used inside it. Here the `simd`
objects and `copy_to` intrinsics are used which are available only in the ESIMD extension.
Full runnable code sample can be found on the
[github repo](https://github.com/intel/llvm/blob/sycl/sycl/test-e2e/ESIMD/vadd_usm.cp).

### Compiling and running ESIMD code

Code that uses the ESIMD extension can be compiled and run using the same commands
as standard SYCL:

To compile using the open-source Intel DPC++/C++ compiler:
```bash
 clang++ -fsycl vadd_usm.cpp
```

To compile using Intel(R) oneAPI DPC++/C++ Compiler:
```bash
 icpx -fsycl vadd_usm.cpp
 #or
 clang++ -fsycl vadd_usm.cpp
```
To run on an Intel GPU device through the Level Zero backend:
```bash
 ONEAPI_DEVICE_SELECTOR=level_zero:gpu ./a.out
```

The resulting executable (`a.out`) can only be run on Intel GPU hardware, such as
Intel(R) UHD Graphics 600 or later. The DPC++ runtime automatically recognizes ESIMD
kernels and dispatches their execution, so no additional setup is needed. Both Linux
and Windows platforms are supported, including OpenCL and Level Zero backends.

Regular SYCL and ESIMD kernels can co-exist in the same translation unit and in
the same application.

### SYCL and ESIMD interoperability

SYCL kernels can call ESIMD functions using the special `invoke_simd` API.
More details are available in [invoke_simd spec](../sycl_ext_oneapi_invoke_simd.asciidoc)
Test cases are available [here](../../../../test-e2e/InvokeSimd/)

```cpp
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/oneapi/experimental/invoke_simd.hpp>
#include <sycl/sycl.hpp>

constexpr int N = 8;

namespace seoe = sycl::ext::oneapi::experimental::simd;
namespace esimd = sycl::ext::intel::simd;

// ESIMD function
[[intel::device_indirectly_callable]] SYCL_EXTERNAL seoe::simd<float, N> __regcall esimd_scale(seoe::simd<float, N> x, float n) SYCL_ESIMD_FUNCTION {
  return esimd::simd<float, N>(x) * n;
}
...
auto ndr = nd_range<1>{range<1>{global_size}, range<1>{N * num_sub_groups}};
q.parallel_for(ndr, sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(N)]] {
  sycl::sub_group sg = it.get_sub_group();
  float x = ...;
  float n = ...;

  // Invoke SIMD function:
  // `x` values from each work-item are grouped into a simd<float, N>.
  // `n` is passed as a uniform scalar.
  // The vector result simd<float, N> is split into N scalar elements,
  // then assigned to each `y` of each corresponding N work-items.
  float y = seoe::invoke_simd(sg, esimd_scale, x, seoe::uniform(n));
});
```

Currently, compilation of programs with `invoke_simd` calls requires a few additional compilation options. Also, running such programs may require setting additional parameters for the GPU driver:
```bash
# compile: pass -fsycl-allow-func-ptr because by default the function pointers
# are not allowed in SYCL/ESIMD programs;
# also pass -fno-sycl-device-code-split-esimd to keep invoke_simd() caller
# and callee in the same module.
clang++ -fsycl -fno-sycl-device-code-split-esimd -Xclang -fsycl-allow-func-ptr -o invoke_simd
# run the program:
IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 invoke_simd
```

### ESIMD_EMULATOR backend

Under Linux environment, the same resulting executable file can be run
on CPU under emulation mode without Intel GPU. For details, check
[ESIMD_EMULATOR back-end] (esimd_emulator.md)

### Restrictions

This section contains lists of the main restrictions that apply when using the ESIMD
extension.
> **Note**: Some restrictions are not enforced by the compiler, which may lead to
> undefined program behavior if violated.

#### Features not supported with the ESIMD extension:
- The [C and C++ Standard libraries support](../supported/C-CXX-StandardLibrary.rst)
- The [Device library extensions](../../../design/DeviceLibExtensions.rst)

#### Unsupported standard SYCL APIs:
- Local accessors are not implemented yet. Local memory can be allocated and accessed via the explicit device-side API;
- 2D and 3D accessors;
- Constant accessors;
- `sycl::accessor::get_pointer()`. All memory accesses through an accessor are
done via explicit APIs; e.g. `sycl::ext::intel::experimental::esimd::block_store(acc, offset)`;
- Accessors with offsets and/or access range specified;
- `sycl::image`, `sycl::sampler`, `sycl::stream` classes;

#### Other restrictions:

- Only Intel GPU device is supported.
- Interoperability between regular SYCL and ESIMD kernels is only supported one way.
  Regular SYCL kernels can call ESIMD functions, but not vice-versa. Invocation of SYCL code from ESIMD is not supported yet.
