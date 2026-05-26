// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsyntax-only -I %sycl_source_dir/include -I %sycl_include %s
// expected-no-diagnostics

#include <sycl/ext/oneapi/free_function_kernel_properties.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

constexpr size_t WorkGroupX = 2;
constexpr size_t WorkGroupY = 3;
constexpr size_t WorkGroupHintX = 4;
constexpr size_t WorkGroupHintY = 5;
constexpr size_t WorkGroupHintZ = 6;
constexpr uint32_t SubGroupSize = 7;
constexpr size_t MaxWorkGroupX = 8;
constexpr size_t MaxWorkGroupY = 9;
constexpr size_t MaxLinearWorkGroupSize = 10;

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::single_task_kernel))
void SingleTaskKernel();

int main() {
  static_assert(
      syclexp::is_property_value<decltype(syclexp::nd_range_kernel<1>)>::value);
  static_assert(
      syclexp::is_property_value<decltype(syclexp::single_task_kernel)>::value);
  static_assert(
      syclexp::is_property_value<
          decltype(syclexp::work_group_size<WorkGroupX, WorkGroupY>)>::value);
  static_assert(
      syclexp::is_property_value<
          decltype(syclexp::work_group_size_hint<WorkGroupHintX, WorkGroupHintY,
                                                 WorkGroupHintZ>)>::value);
  static_assert(syclexp::is_property_value<
                decltype(syclexp::sub_group_size<SubGroupSize>)>::value);
  static_assert(syclexp::is_property_value<
                decltype(syclexp::max_work_group_size<MaxWorkGroupX,
                                                      MaxWorkGroupY>)>::value);
  static_assert(
      syclexp::is_property_value<decltype(syclexp::max_linear_work_group_size<
                                          MaxLinearWorkGroupSize>)>::value);

  static_assert(syclexp::work_group_size<WorkGroupX, WorkGroupY>[0] ==
                WorkGroupX);
  static_assert(syclexp::work_group_size<WorkGroupX, WorkGroupY>[1] ==
                WorkGroupY);
  static_assert(syclexp::work_group_size_hint<WorkGroupHintX, WorkGroupHintY,
                                              WorkGroupHintZ>[2] ==
                WorkGroupHintZ);
  static_assert(syclexp::sub_group_size<SubGroupSize>.value == SubGroupSize);
  static_assert(syclexp::max_work_group_size<MaxWorkGroupX, MaxWorkGroupY>[1] ==
                MaxWorkGroupY);
  static_assert(
      syclexp::max_linear_work_group_size<MaxLinearWorkGroupSize>.value ==
      MaxLinearWorkGroupSize);
  return 0;
}
