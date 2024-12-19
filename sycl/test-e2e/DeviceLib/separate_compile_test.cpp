// DEFINE: %{mathflags} = %if cl_options %{/clang:-fno-fast-math%} %else %{-fno-fast-math%}
// UNSUPPORTED: hip || cuda
// RUN: %clangxx -fsycl -fsycl-link %S/std_complex_math_test.cpp -o %t_device.o %{mathflags}
// RUN: %clangxx -fsycl-device-only -Xclang -fsycl-int-header=std_complex_math_test_ihdr.h %S/std_complex_math_test.cpp -Wno-sycl-strict %{mathflags}
// >> host compilation...
// RUN: %clangxx -Wno-error=unused-command-line-argument %sycl_include -Wno-error=ignored-attributes %cxx_std_optionc++17 %include_option std_complex_math_test_ihdr.h -c %S/std_complex_math_test.cpp -o %t_host.o %sycl_options -Wno-sycl-strict %{mathflags}
// RUN: %clangxx %t_host.o %t_device.o -Wno-unused-command-line-argument -o %t.out %sycl_options %{mathflags}
// RUN: %{run} %t.out

// RUN: %clangxx -fsycl -fsycl-link  %S/std_complex_math_fp64_test.cpp -o %t_fp64_device.o %{mathflags}
// RUN: %clangxx -fsycl-device-only -Xclang -fsycl-int-header=std_complex_math_fp64_test_ihdr.h %S/std_complex_math_fp64_test.cpp -Wno-sycl-strict %{mathflags}
// >> host compilation...
// RUN: %clangxx -Wno-error=unused-command-line-argument %sycl_include -Wno-error=ignored-attributes %cxx_std_optionc++17 %include_option std_complex_math_fp64_test_ihdr.h -c %S/std_complex_math_fp64_test.cpp -o %t_fp64_host.o %sycl_options -Wno-sycl-strict %{mathflags}
// RUN: %clangxx %t_fp64_host.o %t_fp64_device.o -Wno-unused-command-line-argument -o %t_fp64.out %sycl_options %{mathflags}
// RUN: %{run} %t.out
