// RUN: %clangxx -fsycl -fsycl-link %S/std_complex_math_test.cpp -o %t_device.o
// RUN: %clangxx -fsycl-device-only -Xclang -fsycl-int-header=std_complex_math_test_ihdr.h %S/std_complex_math_test.cpp -Wno-sycl-strict
// >> host compilation...
// RUN: %clangxx -std=c++17 %include_option std_complex_math_test_ihdr.h -c %S/std_complex_math_test.cpp -o %t_host.o %sycl_options -Wno-sycl-strict
// RUN: %clangxx %t_host.o %t_device.o -o %t.out %sycl_options
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// RUN: %clangxx -fsycl -fsycl-link  %S/std_complex_math_fp64_test.cpp -o %t_fp64_device.o
// RUN: %clangxx -fsycl-device-only -Xclang -fsycl-int-header=std_complex_math_fp64_test_ihdr.h %S/std_complex_math_fp64_test.cpp -Wno-sycl-strict
// >> host compilation...
// RUN: %clangxx -std=c++17 %include_option std_complex_math_fp64_test_ihdr.h -c %S/std_complex_math_fp64_test.cpp -o %t_fp64_host.o %sycl_options -Wno-sycl-strict
// RUN: %clangxx %t_fp64_host.o %t_fp64_device.o -o %t_fp64.out %sycl_options
// RUN: %CPU_RUN_PLACEHOLDER %t_fp64.out
// RUN: %ACC_RUN_PLACEHOLDER %t_fp64.out
