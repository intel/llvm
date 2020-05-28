// UNSUPPORTED: cuda
// CUDA does not support SPIR-V.

//-fsycl-targets=%sycl_triple
// RUN: %clangxx -fsycl-device-only -fno-sycl-use-bitcode -Xclang -fsycl-int-header=%t.h -c %s -o %t.spv -I %sycl_include -Xclang -verify-ignore-unexpected=note,warning -Wno-sycl-strict
// RUN: %clangxx -include %t.h -g %s -o %t.out -lsycl -I %sycl_include -Xclang -verify-ignore-unexpected=note,warning
// RUN: env SYCL_BE=%sycl_be SYCL_USE_KERNEL_SPV=%t.spv %t.out | FileCheck %s
// CHECK: Passed

// TODO: InvalidTargetTriple: Expects spir-unknown-unknown or spir64-unknown-unknown. Actual target triple is x86_64-unknown-linux-gnu

#include <CL/sycl.hpp>
#include <iostream>

using namespace cl::sycl;

int main(int argc, char **argv) {
  int data = 5;

  try {
    queue myQueue;
    buffer<int, 1> buf(&data, range<1>(1));

    event e = myQueue.submit([&](handler& cgh) {
      auto ptr = buf.get_access<access::mode::read_write>(cgh);

      cgh.single_task<class my_kernel>([=]() {
        ptr[0]++;
      });
    });
    e.wait_and_throw();

  } catch (cl::sycl::exception const& e) {
    std::cerr << "SYCL exception caught:\n";
    std::cerr << e.what() << "\n";
    return 2;
  }
  catch (...) {
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

