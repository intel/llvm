// RUN: %clangxx -fsycl -c -fsycl-device-only -Xclang -emit-llvm %s -o %t

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  kernelFunc();
}

void caller(int x) {
  kernel<class kernel_esimd>([=]() SYCL_ESIMD_KERNEL {
    named_barrier_init<7>();
    named_barrier_wait(2);
    named_barrier_signal(0, 0, 4, 4);
  });
}
