# Intel "Explicit SIMD" SYCL extension

OneAPI provides the "Explicit SIMD" SYCL extension (or simply "ESIMD") for
lower-level Intel GPU programming. It provides APIs closely matching Intel GPU ISA
yet allows to write explicitly vectorized device code. This helps programmer to
have more control over the generated code and depend less on compiler
optimizations. The [specification](SYCL_EXT_INTEL_ESIMD.md),
[documented ESIMD APIs headers](../../../../include/sycl/ext/intel/experimental/esimd) and
[working code examples](https://github.com/intel/llvm-test-suite/tree/intel/SYCL/ESIMD) are available on the Intel DPC++ project's github.

**_NOTE:_** _This extension is under active development and lots of APIs are
subject to change. There are currently a number of restrictions specified
below._

ESIMD kernels and functions always require the subgroup size of one, which means
compiler never does vectorization across workitems in a subgroup. Instead,
vectorization is experessed explicitly in the code by the programmer. Here is a
trivial example which adds elements of two arrays and writes the results to the
third:

```cpp
  float *A = static_cast<float *>(malloc_shared(Size * sizeof(float), dev, ctxt));
  float *B = static_cast<float *>(malloc_shared(Size * sizeof(float), dev, ctxt));
  float *C = static_cast<float *>(malloc_shared(Size * sizeof(float), dev, ctxt));

  for (unsigned i = 0; i < Size; ++i) {
    A[i] = B[i] = i;
  }

  // We need that many workitems. Each processes VL elements of data.
  cl::sycl::range<1> GlobalRange{Size / VL};
  // Number of workitems in each workgroup.
  cl::sycl::range<1> LocalRange{GroupSize};

  cl::sycl::nd_range<1> Range(GlobalRange, LocalRange);

  auto e = q.submit([&](handler &cgh) {
    cgh.parallel_for<class Test>(Range, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL {
      using namespace sycl::ext::intel::experimental::esimd;

      int i = ndi.get_global_id(0);
      simd<float, VL> va;
      va.copy_from(A + i * VL);
      simd<float, VL> vb;
      vb.copy_from(B + i * VL);
      simd<float, VL> vc = va + vb;
      vc.copy_to(C + i * VL);
    });
  });
```

In this example the lambda function passed to the `parallel_for` is marked with
a special attribute - `SYCL_ESIMD_KERNEL`. This tells the compiler that this
kernel is a ESIMD one and ESIMD APIs can be used inside it. Here the `simd`
objects and `copy_from`/`copy_to` intrinsics are used which are avaiable
only in the ESIMD extension.
Full runnable code sample can be found on the
[github repo](https://github.com/intel/llvm-test-suite/blob/intel/SYCL/ESIMD/vadd_usm.cpp).

#### Compiling and running ESIMD code.

A code, which uses the ESIMD extension can be compiled and run using the same
options as the regular SYCL code, e.g.:

> `$ clang++ -fsycl vadd_usm.cpp`
> `$ SYCL_DEVICE_FILTER=level_zero:gpu ./a.out`

The resulting executable can only be run on Intel GPU hardware, such as
Intel HD Graphics 600 or later. Both Linux and Windows platforms are supported,
including OpenCL and LevelZero backends.

Regular SYCL and ESIMD kernels can co-exist in the same translation unit and in
the same application, however interoperability (e.g. invocation of ESIMD
functions from a standard SYCL code) between them is not yet supported.

#### Restrictions

Here is a list of main restrictions imposed on using ESIMD extension. Note that
some of them are not enforced by the compiler, which may lead to undefined
program behavior if violated.

##### Features not supported with ESIMD extension:
- Ahead-of-time compilation
- The [C and C++ Standard libraries support](../supported/C-CXX-StandardLibrary.rst)
- The [Device library extensions](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/C-CXX-StandardLibrary/DeviceLibExtensions.rst)
- Host device (in some cases)

##### Unsupported standard SYCL APIs:
- Local accessors
- Most of image APIs
- Memory access through a raw pointer returned by `sycl::accessor::get_pointer()`

##### Other restrictions:
- Only Intel GPU device is supported
- Interoperability between regular SYCL and ESIMD kernels is not yet supported.
  I.e., it's not possible to invoke an ESIMD kernel from SYCL kernel and vice-versa.
