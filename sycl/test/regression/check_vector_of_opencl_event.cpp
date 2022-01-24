// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify -DSYCL2020_CONFORMANT_APIS %s
// expected-no-diagnostics
//
//===----------------------------------------------------------------------===//
// This test checks that sycl::get_native<sycl::backend::opencl>(event) return
// std::vector<cl_event> when backend = opencl, according to:
// SYCL™ 2020 Specification (revision 3)
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

int main() {
#ifdef SYCL_BACKEND_OPENCL
  sycl::queue Queue;
  if (Queue.get_backend() == sycl::backend::opencl) {
    sycl::event event = Queue.submit([&](sycl::handler &cgh) {
      cgh.single_task<class event_kernel>([]() {});
    });
    std::vector<cl_event> interopEventVec =
        sycl::get_native<sycl::backend::opencl>(event);
  }
#endif
}
