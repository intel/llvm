### TODO list for the SYCL Explicit SIMD extension.

A place for TODO items/feedback regarding the extension in addition to the
github issues mechanism.

#### Front-End

1. Fix a generic (unrelated to ESIMD) issue [1811](https://github.com/intel/llvm/issues/1811) with lambda/functor function
   detection. Needed to improve diagnostics, fix attribute propagation.  
   ETA: ???
2. Fix kernel body function detection. (unrelated to ESIMD)
  clang/lib/Sema/SemaSYCL.cpp:296 function isSYCLKernelBodyFunction
  The test in the function should involve checking the caller and matching function
  types of the caller's parameter and the type of 'this' of this function. But the
  information about the original caller (e.g. kernel_parallel_for) is
  unavailable at this point - kernel creation infrastructure must be enhanced.
  For now the check is only if FD is '()' operator. Works OK for today's
  handler::kernel_parallel_for/... implementations as no other '()' operators
  are invoked except the kernel body.

