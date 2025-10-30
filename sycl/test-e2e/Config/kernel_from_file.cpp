// REQUIRES: target-spir

// FIXME Disabled fallback assert as it'll require either online linking or
// explicit offline linking step here
// FIXME separate compilation requires -fno-sycl-dead-args-optimization
// As we are doing a separate device compilation here, we need to explicitly
// add the device lib instrumentation (itt_compiler_wrapper)
// RUN: %clangxx -Wno-error=ignored-attributes -DSYCL_DISABLE_FALLBACK_ASSERT %cxx_std_optionc++17 -fsycl-device-only -fno-sycl-dead-args-optimization -Xclang -fsycl-int-header=%t.h %s -o %t.bc -Xclang -verify-ignore-unexpected=note,warning -Wno-sycl-strict

// >> ---- unbundle compiler wrapper and asan device objects
// RUN:/localdisk3/yixingzh/offload_fp64_conv_fail/llvm/build/bin/llvm-link -o ./kernel_from_file.cpp.tmp_app.bc /localdisk3/yixingzh/offload_fp64_conv_fail/llvm/build/tools/sycl/test-e2e/Config/Output/kernel_from_file.cpp.tmp.bc /localdisk3/yixingzh/offload_fp64_conv_fail/llvm/build/lib/libsycl-itt-user-wrappers.bc /localdisk3/yixingzh/offload_fp64_conv_fail/llvm/build/lib/libsycl-asan.bc

// >> ---- translate to SPIR-V
// RUN: llvm-spirv -o %t.spv %t_app.bc
// RUN: %clangxx -Wno-error=ignored-attributes %sycl_include -DSYCL_DISABLE_FALLBACK_ASSERT %cxx_std_optionc++17 %include_option %t.h %s -o %t.out %sycl_options -Xclang -verify-ignore-unexpected=note,warning %if preview-mode %{-Wno-unused-command-line-argument%}
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

  if (data != 6) {
    std::cerr << "Failed: " << data << "!= 6(gold)\n";
    return 1;
  }

  return 0;
}
