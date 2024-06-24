// RUN: %{build} -o %t.out
// REQUIRES: cuda || hip || level_zero
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

#include <cassert>
#include <iostream>

using namespace sycl;

int main() {
  queue q;
  device dev = q.get_device();

#if !defined(SYCL_EXT_ONEAPI_MAX_WORK_GROUP_QUERY)
#error SYCL_EXT_ONEAPI_MAX_WORK_GROUP_QUERY is not defined!
#endif

  sycl::id<1> groupD = dev.get_info<
      sycl::ext::oneapi::experimental::info::device::max_work_groups<1>>();
  std::cout << "Max work group size in 1D \n";
  std::cout << "Dimension 1:" << groupD[0] << std::endl;

  sycl::id<2> group2D = dev.get_info<
      sycl::ext::oneapi::experimental::info::device::max_work_groups<2>>();
  std::cout << "Max work group size in 2D \n";
  std::cout << "Dimension 1:" << group2D[0] << "\n"
            << "Dimension 2:" << group2D[1] << std::endl;

  sycl::id<3> group3D = dev.get_info<
      sycl::ext::oneapi::experimental::info::device::max_work_groups<3>>();
  std::cout << "Max work group size in 3D \n";
  std::cout << "Dimension 1:" << group3D[0] << "\n"
            << "Dimension 2:" << group3D[1] << "\n"
            << "Dimension 3:" << group3D[2] << std::endl;

  size_t group_max = dev.get_info<
      sycl::ext::oneapi::experimental::info::device::max_global_work_groups>();
  std::cout << "Max global work group size:" << group_max << "\n";

  assert((group3D[0] <= group_max && group3D[1] <= group_max &&
          group3D[2] <= group_max) &&
         "Max work-group size of each dimension must be smaller than "
         "global work-group size");

  std::cout << "Passed!" << std::endl;
  return 0;
}
