// UNSUPPORTED: cuda || hip
// CUDA and HIP don't support SPIR-V.

// FIXME Disabled fallback assert as it'll require either online linking or
// explicit offline linking step here
// RUN: %clangxx -DSYCL_DISABLE_FALLBACK_ASSERT %cxx_std_optionc++17 -fsycl-device-only -fno-sycl-use-bitcode -Xclang -fsycl-int-header=%t.h -c %s -o %t.spv -Xclang -verify-ignore-unexpected=note,warning -Wno-sycl-strict
// RUN: %clangxx -DSYCL_DISABLE_FALLBACK_ASSERT %cxx_std_optionc++17 %include_option %t.h %s -o %t.out %sycl_options -Xclang -verify-ignore-unexpected=note,warning
// RUN: %BE_RUN_PLACEHOLDER env SYCL_USE_KERNEL_SPV=%t.spv %t.out | FileCheck %s
// CHECK: Passed

#include <CL/sycl.hpp>
#include <iostream>

using namespace cl::sycl;

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

  } catch (cl::sycl::exception const &e) {
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
