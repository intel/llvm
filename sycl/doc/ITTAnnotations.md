# ITT annotations support

This extension enables a set of functions implementing
the Instrumentation and Tracing Technology (ITT) functionality
in SYCL device code.

There are three sets of functions defined by this extension,
and they serve different purposes.

## User APIs

The user code calling these functions must include the corresponding header
file(s) provided by `ittnotify` project (TBD: reference ITT repo here).

These functions are named using `__itt_notify_` prefix.

## Stub APIs

These functions are not defined in any header file, and their declarations
follow exactly the declarations of the corresponding user APIs, except that
they have an extra `_stub` suffix in their names.

These functions implement the ITT functionality in a way that allows
the tools, such as Intel(R) Inspector, to recognize the ITT annotations
and run their analysis methods based on that.

For SYCL device code these functions are implemented as `noinline` and
`optnone` functions so that the corresponding calls may be distinguished
in the execution trace. This is just one way for implementing them,
and the actual implementation may change in future.

## Compiler wrapper APIs

These functions are not defined in any header file, and they are supposed
to be called from the compiler generated code. These thin wrappers
just provide a convenient way for compilers to produce ITT annotations
without generating too much code in the compilers' IR.

These functions have `_wrapper` suffix in their names.

**Example**

```c++
DEVICE_EXTERN_C void __itt_offload_wi_start_stub(
 size_t[3], size_t, uint32_t);

DEVICE_EXTERN_C void __itt_offload_wi_start_wrapper() {
  if (__spirv_SpecConstant(0xFF747469, 0)) {
    size_t GroupID[3] = ...;
    size_t WIId = ...;
    uint32_t WGSize = ...;
    __itt_offload_wi_start_stub(GroupID, WIId, WGSize);
  }
}
```

A compiler may generate a simple call to `__itt_offload_wi_start_wrapper`
to annotate a kernel entry point. Compare this to the code inside the wrapper
function, which a compiler would have to generate if there were no such
a wrapper.

## Conditional compilation

Data Parallel C++ compiler automatically instruments user code through
SPIRITTAnnotations LLVM pass, which is enabled for targets, that natively
support specialization constants (i.e., SPIR-V targets). Annotations are
generated for barriers, atomics, work item start and finish.
To minimize the effect of ITT annotations on the performance of the device code,
the implementation is guarded with a specialization constant check. This allows
users and tools to have one version of the annotated code that may be built
with and without ITT annotations "enabled". When the ITT annotations are not
enabled, we expect that the overall effect of the annotations will be minimized
by the dead code elimination optimization(s) made by the device compilers.

For this purpose we reserve a 1-byte specialization constant numbered
`4285822057` (`0xFF747469`). The users/tools/runtimes should set this
specialization constant to non-zero value to enable the ITT annotations
in SYCL device code.

The specialization constant value is controlled by
INTEL_ENABLE_OFFLOAD_ANNOTATIONS environment variable. Tools, that support ITT
annotations must set this environment variable to any value.
