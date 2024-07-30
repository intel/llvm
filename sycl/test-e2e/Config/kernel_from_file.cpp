// UNSUPPORTED: cuda || hip
// CUDA and HIP don't support SPIR-V.

// FIXME Disabled fallback assert as it'll require either online linking or
// explicit offline linking step here
// FIXME separate compilation requires -fno-sycl-dead-args-optimization
// As we are doing a separate device compilation here, we need to explicitly
// add the device lib instrumentation (itt_compiler_wrapper)
// RUN: %clangxx -DSYCL_DISABLE_FALLBACK_ASSERT %cxx_std_optionc++17 -fsycl-device-only -fno-sycl-dead-args-optimization -Xclang -fsycl-int-header=%t.h -c %s -o %t.bc -Xclang -verify-ignore-unexpected=note,warning -Wno-sycl-strict
// >> ---- unbundle compiler wrapper and sanitizer device objects
// RUN: clang-offload-bundler -type=o -targets=sycl-spir64-unknown-unknown -input=%sycl_static_libs_dir/libsycl-itt-compiler-wrappers%obj_ext -output=%t_compiler_wrappers.bc -unbundle
// RUN: %if linux %{ clang-offload-bundler -type=o -targets=sycl-spir64-unknown-unknown -input=%sycl_static_libs_dir/libsycl-sanitizer%obj_ext -output=%t_sanitizer.bc -unbundle %}
// >> ---- link device code
// RUN: %if linux %{ llvm-link -o=%t_app.bc %t.bc %t_compiler_wrappers.bc %t_sanitizer.bc %} %else %{ llvm-link -o=%t_app.bc %t.bc %t_compiler_wrappers.bc %}
// >> ---- translate to SPIR-V
// RUN: llvm-spirv -o %t.spv %t_app.bc
// RUN: %clangxx -DSYCL_DISABLE_FALLBACK_ASSERT %cxx_std_optionc++17 %include_option %t.h %s -o %t.out %sycl_options -fno-sycl-dead-args-optimization -Xclang -verify-ignore-unexpected=note,warning
// RUN: env SYCL_USE_KERNEL_SPV=%t.spv %{run} %t.out | FileCheck %s
// CHECK: Passed

#include <iostream>
#include <sycl/detail/core.hpp>

using namespace sycl;

int main(int argc, char **argv) {
  int data = 5;

  try {
    queue myQueue;
    buffer<int, 1> buf(&data, range<1>(1));

    event e = myQueue.submit([&](handler &cgh) {
      auto ptr = buf.get_access<access::mode::read_write>(cgh);

      cgh.single_task<class my_kernel>([=]() { ptr[0]++; });
    });
    e.wait_and_throw();

  } catch (sycl::exception const &e) {
    std::cerr << "SYCL exception caught:\n";
    std::cerr << e.what() << "\n";
    return 2;
  } catch (...) {
    std::cerr << "unknown exception caught\n";
    return 1;
  }

  if (data == 6) {
    std::cout << "Passed\n";
    return 0;
  } else {
    std::cout << "Failed: " << data << "!= 6(gold)\n";
    return 1;
  }
}
