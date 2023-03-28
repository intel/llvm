// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s
// expected-no-diagnostics

#include <sycl/sycl.hpp>

using namespace sycl::ext::oneapi::experimental;
using namespace sycl::ext::intel::experimental;

static task_sequence<int, decltype(properties(balanced))> TaskSeq1;
static task_sequence<int, decltype(properties(response_capacity<5>))> TaskSeq2;
static task_sequence<int, decltype(properties(invocation_capacity<2>))>
    TaskSeq3;
static task_sequence<int, decltype(properties(pipelined<-1>,
                                              use_stall_enable_clusters<1>))>
    TaskSeq4;
static task_sequence<int,
                     decltype(properties(balanced, invocation_capacity<2>,
                                         response_capacity<4>, pipelined<-1>,
                                         use_stall_enable_clusters<-1>))>
    TaskSeq5;

// Checks is_property_key_of and is_property_value_of for T.
template <typename T> void checkIsPropertyOf() {
  static_assert(is_property_key_of<balanced_key, T>::value);
  static_assert(is_property_key_of<response_capacity_key, T>::value);
  static_assert(is_property_key_of<invocation_capacity_key, T>::value);
  static_assert(is_property_key_of<pipelined_key, T>::value);
  static_assert(is_property_key_of<use_stall_enable_clusters_key, T>::value);

  static_assert(is_property_value_of<decltype(balanced), T>::value);
  static_assert(is_property_value_of<decltype(response_capacity<1>), T>::value);
  static_assert(
      is_property_value_of<decltype(invocation_capacity<1>), T>::value);
  static_assert(is_property_value_of<decltype(pipelined<1>), T>::value);
  static_assert(
      is_property_value_of<decltype(use_stall_enable_clusters<1>), T>::value);
}

int main() {
  static_assert(is_property_key<balanced_key>::value);
  static_assert(is_property_key<response_capacity_key>::value);
  static_assert(is_property_key<invocation_capacity_key>::value);
  static_assert(is_property_key<pipelined_key>::value);
  static_assert(is_property_key<use_stall_enable_clusters_key>::value);

  static_assert(is_property_value<decltype(balanced)>::value);
  static_assert(is_property_value<decltype(response_capacity<1>)>::value);
  static_assert(is_property_value<decltype(invocation_capacity<1>)>::value);
  static_assert(is_property_value<decltype(pipelined<1>)>::value);
  static_assert(
      is_property_value<decltype(use_stall_enable_clusters<1>)>::value);

  checkIsPropertyOf<decltype(TaskSeq1)>();
  static_assert(TaskSeq1.has_property<balanced_key>());
  static_assert(!TaskSeq1.has_property<response_capacity_key>());
  static_assert(!TaskSeq1.has_property<invocation_capacity_key>());
  static_assert(!TaskSeq1.has_property<pipelined_key>());
  static_assert(!TaskSeq1.has_property<use_stall_enable_clusters_key>());
  static_assert(TaskSeq1.get_property<balanced_key>() == balanced);

  checkIsPropertyOf<decltype(TaskSeq2)>();
  static_assert(TaskSeq2.has_property<response_capacity_key>());
  static_assert(!TaskSeq2.has_property<balanced_key>());
  static_assert(!TaskSeq2.has_property<invocation_capacity_key>());
  static_assert(!TaskSeq2.has_property<pipelined_key>());
  static_assert(!TaskSeq2.has_property<use_stall_enable_clusters_key>());
  static_assert(TaskSeq2.get_property<response_capacity_key>() ==
                response_capacity<5>);

  checkIsPropertyOf<decltype(TaskSeq3)>();
  static_assert(TaskSeq3.has_property<invocation_capacity_key>());
  static_assert(!TaskSeq3.has_property<balanced_key>());
  static_assert(!TaskSeq3.has_property<response_capacity_key>());
  static_assert(!TaskSeq3.has_property<pipelined_key>());
  static_assert(!TaskSeq3.has_property<use_stall_enable_clusters_key>());
  static_assert(TaskSeq3.get_property<invocation_capacity_key>() ==
                invocation_capacity<2>);

  checkIsPropertyOf<decltype(TaskSeq4)>();
  static_assert(TaskSeq4.has_property<pipelined_key>());
  static_assert(TaskSeq4.has_property<use_stall_enable_clusters_key>());
  static_assert(!TaskSeq4.has_property<response_capacity_key>());
  static_assert(!TaskSeq4.has_property<invocation_capacity_key>());
  static_assert(!TaskSeq4.has_property<balanced_key>());
  static_assert(TaskSeq4.get_property<pipelined_key>() == pipelined<-1>);
  static_assert(TaskSeq4.get_property<use_stall_enable_clusters_key>() ==
                use_stall_enable_clusters<1>);

  checkIsPropertyOf<decltype(TaskSeq5)>();
  static_assert(TaskSeq5.has_property<invocation_capacity_key>());
  static_assert(TaskSeq5.has_property<response_capacity_key>());
  static_assert(TaskSeq5.has_property<pipelined_key>());
  static_assert(TaskSeq5.has_property<use_stall_enable_clusters_key>());
  static_assert(TaskSeq5.has_property<balanced_key>());
  static_assert(TaskSeq5.get_property<invocation_capacity_key>() ==
                invocation_capacity<2>);
  static_assert(TaskSeq5.get_property<response_capacity_key>().value ==
                response_capacity<4>);
  static_assert(TaskSeq5.get_property<pipelined_key>().value == pipelined<-1>);
  static_assert(TaskSeq5.get_property<use_stall_enable_clusters_key>().value ==
                use_stall_enable_clusters<-1>);
  static_assert(TaskSeq5.get_property<balanced_key>().value == balanced);

  return 0;
}
