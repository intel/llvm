// Check that device code compiled for spir64_gen, which has no target ID, is
// used for each of several GPU architectures when linking objects and a static
// archive, and runs on any of these devices.

// REQUIRES: ocloc, target-spir
// REQUIRES: arch-intel_gpu_pvc || gpu-intel-dg2 || arch-intel_gpu_bmg_g21
// UNSUPPORTED: new-offload-model
// UNSUPPORTED-INTENDED: The new offloading model does not support
// -fsycl-targets=spir64_gen.

// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen -Wno-deprecated -DSOURCE1 -c %s -o %t1.o
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen -Wno-deprecated -DSOURCE2 -c %s -o %t2.o
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_pvc,intel_gpu_dg2,intel_gpu_bmg_g21 -Wno-sycl-target %t1.o %t2.o -o %t.out
// RUN: %{run} %t.out

// The kernel is in an object so that it pulls the device function from the
// archive.
// RUN: rm -f %t.a
// RUN: llvm-ar cr %t.a %t2.o
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_pvc,intel_gpu_dg2,intel_gpu_bmg_g21 -Wno-sycl-target %t1.o %t.a -o %t.a.out
// RUN: %{run} %t.a.out

#include <iostream>
#include <sycl/detail/core.hpp>

SYCL_EXTERNAL int get_value();

#ifdef SOURCE1
int main() {
  sycl::queue q;

  int data = 0;
  {
    sycl::buffer<int, 1> buf(&data, sycl::range<1>(1));

    q.submit([&](sycl::handler &h) {
       auto acc = buf.get_access<sycl::access_mode::write>(h);
       h.single_task([=]() { acc[0] = get_value(); });
     }).wait();
  }

  if (data != 42) {
    std::cerr << "Error: expected 42, got " << data << std::endl;
    return 1;
  }

  return 0;
}
#endif // SOURCE1

#ifdef SOURCE2
SYCL_EXTERNAL int get_value() { return 42; }
#endif // SOURCE2
