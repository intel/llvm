# Intel "Explicit SIMD" SYCL extension

OneAPI provides the "Explicit SIMD" SYCL extension (or simply "ESIMD") for
lower-level Intel GPU programming. It provides APIs closely matching Intel GPU ISA
yet allows to write explicitly vectorized device code. This helps programmer to
have more control over the generated code and depend less on compiler
optimizations.

ESIMD kernels and functions always require the subgroup size of one, which means
compiler never does vectorization across work-items in a subgroup. Instead,
vectorization is expressed explicitly in the code by the programmer.

**IMPORTANT NOTE: _Some parts of this extension are under active development. The APIs in the
`sycl::ext::intel::experimental::esimd` namespace are subject to change or removal._**

Please see the additional resources on the Intel DPC++ project's github:

1) [ESIMD Extension Specification](./sycl_ext_intel_esimd.md)
1) [ESIMD API/doxygen reference](https://intel.github.io/llvm-docs/doxygen/group__sycl__esimd.html)
1) [Examples](./examples/README.md)
1) [ESIMD end-to-end LIT tests](https://github.com/intel/llvm/blob/sycl/sycl/test-e2e/ESIMD/)
1) [Implementation and API Restrictions](./sycl_ext_intel_esimd.md#implementation-restrictions)

---

Here is a trivial example which adds elements of two arrays and writes the results to the
third:

```cpp
    float *A = malloc_shared<float>(Size, q);
    float *B = malloc_shared<float>(Size, q);
    float *C = malloc_shared<float>(Size, q);

    for (unsigned i = 0; i != Size; i++)
      A[i] = B[i] = i;

    q.parallel_for(Size / VL, [=](id<1> i)[[intel::sycl_explicit_simd]] {
      auto offset = i * VL;
      // pointer arithmetic, so offset is in elements:
      simd<float, VL> va(A + offset);
      simd<float, VL> vb(B + offset);
      simd<float, VL> vc = va + vb;
      vc.copy_to(C + offset);
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

More examples are available [here](./examples/)

More details are available in [invoke_simd spec](../../experimental/sycl_ext_oneapi_invoke_simd.asciidoc)

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
IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 ./invoke_simd
```
