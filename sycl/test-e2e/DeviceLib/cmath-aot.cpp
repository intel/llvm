// REQUIRES: opencl-aot, cpu
// UNSUPPORTED: windows

// RUN: %clangxx -fsycl -fsycl-targets=spir64_x86_64 %S/cmath_test.cpp -o %t.cmath.out
// RUN: %CPU_RUN_PLACEHOLDER %t.cmath.out

// RUN: %clangxx -fsycl -fsycl-targets=spir64_x86_64 %S/cmath_fp64_test.cpp -o %t.cmath.fp64.out
// RUN: %CPU_RUN_PLACEHOLDER %t.cmath.fp64.out

// RUN: %clangxx -fsycl -fsycl-targets=spir64_x86_64 %S/std_complex_math_test.cpp -o %t.complex.out
// RUN: %CPU_RUN_PLACEHOLDER %t.complex.out

// RUN: %clangxx -fsycl -fsycl-targets=spir64_x86_64 %S/std_complex_math_fp64_test.cpp -o %t.complex.fp64.out
// RUN: %CPU_RUN_PLACEHOLDER %t.complex.fp64.out
