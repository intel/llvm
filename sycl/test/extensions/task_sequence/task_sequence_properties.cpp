// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s
// expected-no-diagnostics

#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

using namespace sycl::ext::oneapi::experimental;
using namespace sycl::ext::intel::experimental;

void dummyTask() {}

static task_sequence<dummyTask,
                     decltype(properties(balanced, response_capacity<0>,
                                         invocation_capacity<0>, pipelined<-1>,
                                         stall_free_clusters))>
    TaskSequence1;
static task_sequence<
    dummyTask, decltype(properties(response_capacity<2>, invocation_capacity<3>,
                                   pipelined<1>, stall_enable_clusters))>
    TaskSequence2;
static task_sequence<dummyTask, decltype(properties(pipelined<0>))>
    TaskSequence3;

// Checks is_property_key_of and is_property_value_of for T.
template <typename T> void checkIsPropertyOf() {
  static_assert(is_property_key_of<balanced_key, T>::value);
  static_assert(is_property_key_of<response_capacity_key, T>::value);
  static_assert(is_property_key_of<invocation_capacity_key, T>::value);
  static_assert(is_property_key_of<pipelined_key, T>::value);
  static_assert(is_property_key_of<fpga_cluster_key, T>::value);

  static_assert(is_property_value_of<decltype(balanced), T>::value);
  static_assert(is_property_value_of<decltype(response_capacity<1>), T>::value);
  static_assert(
      is_property_value_of<decltype(invocation_capacity<1>), T>::value);
  static_assert(is_property_value_of<decltype(pipelined<1>), T>::value);
  static_assert(is_property_value_of<
                decltype(fpga_cluster<fpga_cluster_options_enum::stall_enable>),
                T>::value);
}

int main() {
  static_assert(is_property_key<balanced_key>::value);
  static_assert(is_property_key<response_capacity_key>::value);
  static_assert(is_property_key<invocation_capacity_key>::value);
  static_assert(is_property_key<pipelined_key>::value);
  static_assert(is_property_key<fpga_cluster_key>::value);

  static_assert(is_property_value<decltype(balanced)>::value);
  static_assert(is_property_value<decltype(response_capacity<1>)>::value);
  static_assert(is_property_value<decltype(invocation_capacity<1>)>::value);
  static_assert(is_property_value<decltype(pipelined<-1>)>::value);
  static_assert(is_property_value<decltype(pipelined<0>)>::value);
  static_assert(is_property_value<decltype(pipelined<1>)>::value);
  static_assert(is_property_value<decltype(stall_enable_clusters)>::value);
  static_assert(is_property_value<decltype(stall_free_clusters)>::value);

  checkIsPropertyOf<decltype(TaskSequence1)>();
  static_assert(TaskSequence1.has_property<balanced_key>());
  static_assert(TaskSequence1.has_property<response_capacity_key>());
  static_assert(TaskSequence1.has_property<invocation_capacity_key>());
  static_assert(TaskSequence1.has_property<pipelined_key>());
  static_assert(TaskSequence1.has_property<fpga_cluster_key>());
  static_assert(TaskSequence1.get_property<balanced_key>() == balanced);
  static_assert(TaskSequence1.get_property<response_capacity_key>() ==
                response_capacity<0>);
  static_assert(TaskSequence1.get_property<invocation_capacity_key>() ==
                invocation_capacity<0>);
  static_assert(TaskSequence1.get_property<pipelined_key>() == pipelined<-1>);
  static_assert(TaskSequence1.get_property<fpga_cluster_key>() ==
                stall_free_clusters);

  checkIsPropertyOf<decltype(TaskSequence2)>();
  static_assert(!TaskSequence2.has_property<balanced_key>());
  static_assert(TaskSequence2.has_property<response_capacity_key>());
  static_assert(TaskSequence2.has_property<invocation_capacity_key>());
  static_assert(TaskSequence2.has_property<pipelined_key>());
  static_assert(TaskSequence2.has_property<fpga_cluster_key>());
  static_assert(TaskSequence2.get_property<response_capacity_key>() ==
                response_capacity<2>);
  static_assert(TaskSequence2.get_property<invocation_capacity_key>() ==
                invocation_capacity<3>);
  static_assert(TaskSequence2.get_property<pipelined_key>() == pipelined<1>);
  static_assert(TaskSequence2.get_property<fpga_cluster_key>() ==
                stall_enable_clusters);

  checkIsPropertyOf<decltype(TaskSequence3)>();
  static_assert(!TaskSequence3.has_property<balanced_key>());
  static_assert(!TaskSequence3.has_property<response_capacity_key>());
  static_assert(!TaskSequence3.has_property<invocation_capacity_key>());
  static_assert(TaskSequence3.has_property<pipelined_key>());
  static_assert(!TaskSequence3.has_property<fpga_cluster_key>());
  static_assert(TaskSequence3.get_property<pipelined_key>() == pipelined<0>);

  return 0;
}
