# Specialization constants.

Specialization constant is basically a variable in a SYCL program set by host
code and used in device code which appears to be constant for the online (JIT)
compiler of the device code. Things like optimal tile size in a tiled matrix
multiplication kernel may depend on the hardware and can be expressed via a
specialization constant for better code generation.

This version of oneAPI provides experimental implementation of specialization
constants based on the
[proposal](https://github.com/codeplaysoftware/standards-proposals/blob/master/spec-constant/index.md)
from Codeplay.

**NOTE:** This extension is now deprecated.  Use the core SYCL specialization
constant APIs defined in the
[SYCL 2020 specification](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html)
instead.

A specialization constant is identified by a C++ type name, similarly to a
kernel, its value is set via `program::set_spec_constant` class API and is
"frozen" once the program is built. The following example shows how
different values of a specialization constant can be used within the same
kernel:

```cpp
  for (int i = 0; i < n_sc_sets; i++) {
    cl::sycl::program program(q.get_context());
    const int *sc_set = &sc_vals[i][0];
    cl::sycl::ext::oneapi::experimental::spec_constant<int32_t, SC0> sc0 =
        program.set_spec_constant<SC0>(sc_set[0]);
    cl::sycl::ext::oneapi::experimental::spec_constant<int32_t, SC1> sc1 =
        program.set_spec_constant<SC1>(sc_set[1]);

    program.build_with_kernel_type<KernelAAA>();

    try {
      cl::sycl::buffer<int, 1> buf(vec.data(), vec.size());

      q.submit([&](cl::sycl::handler &cgh) {
        auto acc = buf.get_access<cl::sycl::access::mode::write>(cgh);
        cgh.single_task<KernelAAA>(
            program.get_kernel<KernelAAA>(),
            [=]() {
              acc[i] = sc0.get() + sc1.get();
            });
      });
    } catch (cl::sycl::exception &e) {
      std::cout << "*** Exception caught: " << e.what() << "\n";
      return 1;
    }
    ...
  }
```
Here the values of specialization constants `SC0` and `SC1` are changed on
every loop iteration. All what's needed is re-creating a `program` class
instance, setting new values and rebuilding it via
`program::build_with_kernel_type`. JIT compiler will effectively replace
`sc0.get()` and  `sc1.get()` within thhe device code with the corresponding
constant values (`sc_vals[i][0]` and `sc_vals[i][1]`). Full runnable example
can be found on
[github](https://github.com/intel/llvm-test-suite/blob/intel/SYCL/SpecConstants/1.2.1/spec_const_redefine.cpp).

Specialization constants can be used in programs compiled Ahead-Of-Time, in this
case a specialization constant takes default value for its type (as specified by
[C++ standard](https://en.cppreference.com/w/cpp/language/value_initialization)).

#### Limitations
- The implementation does not support the `template <unsigned NID> struct spec_constant_id`
  API design for interoperability with OpenCL - to set specializataion constants
  in SYCL programs originating from external SPIRV modules and wrapped by OpenCL
  program objects. In SPIRV/OpenCL specialization constants are identified by an
  integer number, and the `spec_constant_id` class models that.
- Only primitive numeric types are supported.

