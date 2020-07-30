// REQUIRES: opencl-aot, cpu
// UNSUPPORTED: windows

// RUN: %clangxx -fsycl -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice %S/cmath_test.cpp %sycl_libs_dir/libsycl-cmath.o %sycl_libs_dir/libsycl-fallback-cmath.o -o %t.cmath.out
// RUN: %CPU_RUN_PLACEHOLDER %t.cmath.out

// RUN: %clangxx -fsycl -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice %S/cmath_fp64_test.cpp %sycl_libs_dir/libsycl-cmath-fp64.o %sycl_libs_dir/libsycl-fallback-cmath-fp64.o -o %t.cmath.fp64.out
// RUN: %CPU_RUN_PLACEHOLDER %t.cmath.fp64.out

// RUN: %clangxx -fsycl -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice %S/std_complex_math_test.cpp %sycl_libs_dir/libsycl-complex.o %sycl_libs_dir/libsycl-cmath.o %sycl_libs_dir/libsycl-fallback-complex.o %sycl_libs_dir/libsycl-fallback-cmath.o -o %t.complex.out
// RUN: %CPU_RUN_PLACEHOLDER %t.complex.out

// RUN: %clangxx -fsycl -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice %S/std_complex_math_fp64_test.cpp %sycl_libs_dir/libsycl-complex-fp64.o %sycl_libs_dir/libsycl-cmath-fp64.o %sycl_libs_dir/libsycl-fallback-complex-fp64.o %sycl_libs_dir/libsycl-fallback-cmath-fp64.o -o %t.complex.fp64.out
// RUN: %CPU_RUN_PLACEHOLDER %t.complex.fp64.out
