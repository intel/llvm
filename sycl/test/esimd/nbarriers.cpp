// RUN: %clangxx -fsycl -c -fsycl-device-only -Xclang -emit-llvm %s -o %t

#include <CL/sycl.hpp>
#include <sycl/ext/intel/experimental/esimd.hpp>

using namespace sycl::ext::intel::experimental::esimd;

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  kernelFunc();
}

void caller(int x) {
  kernel<class kernel_esimd>([=]() SYCL_ESIMD_KERNEL {
    nbarrier_init<7>();
    nbarrier_wait(2);
    nbarrier_signal(0, 0, 4, 4);
  });
}
