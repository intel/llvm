# Intel "Explicit SIMD" SYCL extension

OneAPI provides the "Explicit SIMD" SYCL extension (or simply "ESIMD") for
lower-level Intel GPU programming. It provides APIs closely matching Intel GPU ISA
yet allows to write explicitly vectorized device code. This helps programmer to
have more control over the generated code and depend less on compiler
optimizations. The [specification](SYCL_EXT_INTEL_ESIMD.md),
[API reference](https://intel.github.io/llvm-docs/doxygen/group__sycl__esimd.html), and
[working code examples](https://github.com/intel/llvm-test-suite/tree/intel/SYCL/ESIMD) are available on the Intel DPC++ project's github.

**_NOTE:_** _Some parts of this extension is under active development and APIs in the
`sycl::ext::intel::experimental::esimd` package are subject to change. There are
currently a number of restrictions specified below._

ESIMD kernels and functions always require the subgroup size of one, which means
compiler never does vectorization across workitems in a subgroup. Instead,
vectorization is experessed explicitly in the code by the programmer. Here is a
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
objects and `copy_to` intrinsics are used which are avaiable only in the ESIMD extension.
Full runnable code sample can be found on the
[github repo](https://github.com/intel/llvm-test-suite/blob/intel/SYCL/ESIMD/vadd_usm.cpp).

#### Compiling and running ESIMD code.

Compiling and running code that uses the ESIMD extension is the same as compiler and
running code that uses the standard SYCL:

To compile using the open-source Intel DPC++ compiler:
> `$ clang++ -fsycl vadd_usm.cpp`

To compile using Intel(R) OneAPI Toolkit:
> `$ dpcpp vadd_usm.cpp`

To run on an Intel GPU device through the Level Zero backend:
> `$ SYCL_DEVICE_FILTER=level_zero:gpu ./a.out`

The resulting executable (`a.out`) can only be run on Intel GPU hardware, such as
Intel(R) UHD Graphics 600 or later. The DPC++ runtime automatically recognizes ESIMD
kernels and dispatches their execution, so no additional setup is needed. Both Linux
and Windows platforms are supported, including OpenCL and Level Zero backends.

Regular SYCL and ESIMD kernels can co-exist in the same translation unit and in
the same application, however interoperability (e.g. invocation of ESIMD
functions from a standard SYCL code) between them is not yet supported.

#### Restrictions

This section contains lists of the main restrictions that apply when using the ESIMD
extension.
> **Note**: Some restrictions are not enforced by the compiler, which may lead to
> undefined program behavior if violated.



##### Features not supported with ESIMD extension:
- The [C and C++ Standard libraries support](../supported/C-CXX-StandardLibrary.rst)
- The [Device library extensions](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/C-CXX-StandardLibrary/DeviceLibExtensions.rst)
- Host device

##### Unsupported standard SYCL APIs:
- Local accessors. Local memory is allocated and accessed via explicit
device-side API
- 2D and 3D accessors
- Constant accessors
- `sycl::accessor::get_pointer()`. All memory accesses through an accessor are
done via explicit APIs; e.g. `sycl::ext::intel::experimental::esimd::block_store(acc, offset)`
- Accessors with offsets and/or access range specified
- `sycl::sampler` and `sycl::stream` classes

##### Other restrictions:
- Only Intel GPU device is supported
- Interoperability between regular SYCL and ESIMD kernels is not yet supported.
  I.e., it's not possible to invoke an ESIMD kernel from SYCL kernel and vice-versa.
