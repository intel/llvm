RFC: SYCL support for unnamed SYCL kernel functions

The SYCL 2020 specification requires implementations to support the use of a lambda expression as
the entry point to a SYCL kernel.
Kernel invocation requires that host and device compilers agree on kernel names.
Kernel names may be explicitly chosen by programmers by passing a type name as a template argument
to a SYCL kernel invocation function like `sycl::handler::single_task`.
In the following example, a kernel name is deterministically generated based on the `my_kernel_name`
type.

```c++
#include <sycl/sycl.hpp>
class my_kernel_name;
int main() {
  sycl::queue q(sycl::cpu_selector_v);
  q.submit([](sycl::handler &cgh) {
    cgh.single_task<my_kernel_name>([]{});
  });
  q.wait();
}
```

If an explicit kernel name is not provided, then the implementation is required to implicitly
generate a kernel name based on the kernel entry point and surrounding context.
The above example can more simply be written as follows.
Note that the declaration of `my_kernel_name` and all uses of it have been removed.

```c++
#include <sycl/sycl.hpp>
int main() {
  sycl::queue q(sycl::cpu_selector_v);
  q.submit([](sycl::handler &cgh) {
    cgh.single_task([]{});
  });
  q.wait();
}
```

When the kernel entry point is a lambda expression, it is technically challenging for host
and device compilers to independently generate a matching kernel name.
The existing C++ name mangling schemes specify how names are generated for the closure type
of a lambda expression, but these names are not generally intended to be stable; particularly
not in the presence of conditionally included code.
The SYCL 2020 specification therefore allows conforming implementations to support a reduced
feature set that does not include support for unnamed SYCL kernel functions.
This is specified in
[Appendix B.2, "Reduced feature set"](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:feature-sets.reduced).
Per
[section 5.6, "Preprocessor directives and macros"](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#_preprocessor_directives_and_macros),
implementations that do not support unnamed SYCL kernel functions are required to predefine
the `SYCL_FEATURE_SET_REDUCED` macro with the value `1` and to not define `SYCL_FEATURE_SET_FULL`.
SYCL programs that use unnamed SYCL kernel functions are therefore more limited in their
portability.

Support for unnamed SYCL kernel functions is more easily achieved when the host compiler
is SYCL-aware since this allows the host and device compilers to use more surrounding context
to generate a stable kernel name than is incorporated in the existing C++ name mangling schemes
used for lambdas.
Clang already supports a `__builtin_sycl_unique_stable_name` builtin function to assist with
production of a stable kernel name.
Modulo defects in its design and implementation, this suffices to provide full support of
unnamed SYCL kernel functions when Clang is used as both the host and device compiler.

Support for SYCL-unaware host compilers could be provided by preprocessing a SYCL translation
unit, instrumenting the resulting preprocessed output with a SYCL-aware tool to insert stable
kernel names in calls to SYCL kernel invocation functions, and then passing the result to
the SYCL-unaware host compiler.
This approach has several downsides including:
- Loss of preprocessing information during host compilation. This may affect when diagnostics
  are issued and their presentation form.
- Loss of precise source location information. This may affect diagnostic presentation form
  and the quality of debugging information.
- Performing the instrumentation would be technically challenging as it would require the
  ability to consume preprocessed output intended for the host compiler while also being able
  to parse the code sufficiently well to deterministically generate kernel names that match
  those produced by the device compiler or to correlate kernel invocations with a table of
  kernel names produced by the device compiler.

At present, the `__builtin_sycl_unique_stable_name` builtin function does not generate
matching names during host and device compilation for the following example.
This is presumed to be either a defect in its implementation or a limitation of its design.
Running this program results in a run-time error due to a failure to resolve kernel names.
It is not entirely clear whether the SYCL specification requires a conforming implementation
to support this example.
An issue has been filed against the SYCL specification to request clarification.
See [issue #434, "Correspondence of unnamed lambdas as kernels across host and device compilation"](https://github.com/KhronosGroup/SYCL-Docs/issues/454).
```c++
#include <sycl/sycl.hpp>
int main() {
  sycl::queue q(sycl::cpu_selector_v);
  q.submit([](sycl::handler &cgh) {
#if !defined(__SYCL_DEVICE_ONLY__)
    // A host-only lambda to misalign discriminators.
    []{}();
#endif
    cgh.single_task([]{});
  });
  q.wait();
}
```

Proposed:
- Support for unnamed SYCL kernel functions will only be provided when Clang is used as
  both the host and device compiler.
- Support for unnamed SYCL kernel functions will not be provided if the `-fsycl-host-compiler`
  option is used. Such support could be provided in the future, either using the preprocessed
  output approach described above or another mechanism, based on demand.
- The `SYCL_FEATURE_SET_FULL` and `SYCL_FEATURE_SET_REDUCED` predefined macros will be defined
  as specified in the SYCL specification based on whether support for unnamed SYCL kernel
  functions is enabled.
- The `__builtin_sycl_unique_stable_name` builtin will be enhanced to address known
  deficiencies like for the example above.
- The `__builtin_sycl_unique_stable_name` builtin will be enhanced to generate names that use
  the `$` or `.` character reserved by the Itanium C++ ABI for private use.
  (see [section 5.1.1, "General", of the Itanium C++ ABI](http://itanium-cxx-abi.github.io/cxx-abi/abi.html#mangling)).
