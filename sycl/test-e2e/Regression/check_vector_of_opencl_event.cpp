// REQUIRES: opencl
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//
//===----------------------------------------------------------------------===//
// This test verifies that sycl::get_native<backend::opencl> and
// sycl::make_event<backend::opencl> work according to the SYCL™ 2020
// Specification (revision 4)
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

int main() {
  sycl::queue Queue;
  sycl::event event = Queue.submit([&](sycl::handler &cgh) {
    cgh.single_task<class event_kernel>([]() {});
  });
  // Check that get_native function returns a vector
  std::vector<cl_event> ClEventVec = get_native<sycl::backend::opencl>(event);
  // Check that make_event is working properly with vector<cl_event> as a
  // param
  sycl::event SyclEvent =
      sycl::make_event<sycl::backend::opencl>(ClEventVec, Queue.get_context());
  std::vector<cl_event> ClEventVecFromMake =
      sycl::get_native<sycl::backend::opencl>(SyclEvent);
  if (ClEventVec[0] != ClEventVecFromMake[0])
    throw std::runtime_error("Cl events are not the same");
  return 0;
}
