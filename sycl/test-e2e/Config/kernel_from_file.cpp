// REQUIRES: target-spir

// FIXME separate compilation requires -fno-sycl-dead-args-optimization
// As we are doing a separate device compilation here, we need to explicitly
// add the device lib instrumentation (itt_compiler_wrapper)
// RUN: %clangxx -Wno-error=ignored-attributes -DUSED_KERNEL -fno-sycl-dead-args-optimization %cxx_std_optionc++17 -fsycl-device-only -Xclang -fsycl-int-header=%t.h %s -o %t.bc -Xclang -verify-ignore-unexpected=note,warning -Wno-sycl-strict
// >> ---- unbundle compiler wrapper and asan device objects
// RUN: clang-offload-bundler -type=o -targets=sycl-spir64-unknown-unknown -input=%sycl_static_libs_dir/libsycl-itt-compiler-wrappers%obj_ext -output=%t_compiler_wrappers.bc -unbundle
// RUN: %if linux %{ clang-offload-bundler -type=o -targets=sycl-spir64-unknown-unknown -input=%sycl_static_libs_dir/libsycl-asan%obj_ext -output=%t_asan.bc -unbundle %}
// >> ---- link device code
// RUN: %if linux %{ llvm-link -o=%t_app.bc %t.bc %t_compiler_wrappers.bc %t_asan.bc %} %else %{ llvm-link -o=%t_app.bc %t.bc %t_compiler_wrappers.bc %}
// >> ---- translate to SPIR-V
// RUN: llvm-spirv -o %t.spv %t_app.bc
// Need to perform full compilation here since the SYCL runtime uses image
// properties from the multi-architecture binary.
// RUN: %{build} -fno-sycl-dead-args-optimization -o %t.out
// RUN: env SYCL_USE_KERNEL_SPV=%t.spv %{run} %t.out

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

      cgh.single_task<class my_kernel>([=]() {
#ifdef USED_KERNEL
        ptr[0]++;
#else
        ptr[0]--;
#endif
      });
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

  if (data != 6) {
    std::cerr << "Failed: " << data << "!= 6(gold)\n";
    return 1;
  }

  return 0;
}
