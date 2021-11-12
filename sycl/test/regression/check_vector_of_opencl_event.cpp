// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics

#include <sycl/sycl.hpp>

int main()
{
#ifdef SYCL_BACKEND_OPENCL
  sycl::queue Queue;
  if (Queue.get_backend() == sycl::backend::opencl)
  {
    sycl::event event = Queue.submit([&](sycl::handler &cgh)
                                     { cgh.single_task<class event_kernel>([]() {}); });
    std::vector<cl_event> interopEventVec = sycl::get_native<sycl::backend::opencl>(event);
  }
#endif
}