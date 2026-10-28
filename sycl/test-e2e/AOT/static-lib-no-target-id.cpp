// Check that, with the old offloading model, kernels from a static library
// compiled for spir64_gen without a target ID are kept when linking for a
// specific Intel GPU architecture.

// REQUIRES: ocloc, gpu, target-spir
// UNSUPPORTED: new-offload-model
// UNSUPPORTED-INTENDED: Checks clang-offload-deps of the old offloading model.

// RUN: %clangxx --no-offload-new-driver -fsycl -fsycl-targets=spir64_gen -Wno-deprecated -DLIB -c %s -o %t.lib.o
// RUN: rm -f %t.a
// RUN: llvm-ar crv %t.a %t.lib.o
// RUN: %clangxx --no-offload-new-driver -fsycl -fsycl-targets=%{gpu_aot_target} -Wno-sycl-target %s %t.a -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

#ifdef LIB
int run(sycl::queue &Q) {
  int Result = 0;
  {
    sycl::buffer<int, 1> Buf(&Result, 1);
    Q.submit([&](sycl::handler &CGH) {
      sycl::accessor Acc(Buf, CGH, sycl::write_only);
      CGH.single_task<class LibKernel>([=] { Acc[0] = 42; });
    });
  }
  return Result;
}
#else
#include <iostream>

int run(sycl::queue &Q);

int main() {
  sycl::queue Q;
  int Result = run(Q);
  if (Result != 42) {
    std::cout << "fail: " << Result << ", expected 42\n";
    return 1;
  }
  return 0;
}
#endif
