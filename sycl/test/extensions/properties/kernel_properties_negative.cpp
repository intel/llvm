// RUN: %clangxx -ferror-limit=0 %fsycl-host-only -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note,warning %s

// Negative tests for kernel properties.

#include <sycl/sycl.hpp>

namespace oneapi = sycl::ext::oneapi::experimental;

extern sycl::kernel TestKernel;

int main() {
  sycl::queue Q{};

  oneapi::properties props1{oneapi::sub_group_size<8>};
  oneapi::properties props2{
      oneapi::sub_group_size<8>,
      oneapi::work_group_progress<oneapi::forward_progress_guarantee::parallel,
                                  oneapi::execution_scope::root_group>};

  {
    // expected-error-re@sycl/ext/oneapi/experimental/enqueue_functions.hpp:* {{static assertion failed due to requirement {{.*}} This kernel enqueue function does not allow properties with compile-time kernel effects.}}
    oneapi::launch_config<sycl::range<1>, decltype(props1)> LC{{1}, props1};
    oneapi::parallel_for(Q, LC, TestKernel);
  }

  {
    // expected-error-re@sycl/ext/oneapi/experimental/enqueue_functions.hpp:* {{static assertion failed due to requirement {{.*}} This kernel enqueue function does not allow properties with compile-time kernel effects.}}
    oneapi::launch_config<sycl::range<1>, decltype(props2)> LC{{1}, props2};
    oneapi::parallel_for(Q, LC, TestKernel);
  }

  {
    // expected-error-re@sycl/ext/oneapi/experimental/enqueue_functions.hpp:* {{static assertion failed due to requirement {{.*}} This kernel enqueue function does not allow properties with compile-time kernel effects.}}
    oneapi::launch_config<sycl::range<2>, decltype(props1)> LC{{1, 1}, props1};
    oneapi::parallel_for(Q, LC, TestKernel);
  }

  {
    // expected-error-re@sycl/ext/oneapi/experimental/enqueue_functions.hpp:* {{static assertion failed due to requirement {{.*}} This kernel enqueue function does not allow properties with compile-time kernel effects.}}
    oneapi::launch_config<sycl::range<2>, decltype(props2)> LC{{1, 1}, props2};
    oneapi::parallel_for(Q, LC, TestKernel);
  }

  {
    // expected-error-re@sycl/ext/oneapi/experimental/enqueue_functions.hpp:* {{static assertion failed due to requirement {{.*}} This kernel enqueue function does not allow properties with compile-time kernel effects.}}
    oneapi::launch_config<sycl::range<3>, decltype(props1)> LC{{1, 1, 1},
                                                               props1};
    oneapi::parallel_for(Q, LC, TestKernel);
  }

  {
    // expected-error-re@sycl/ext/oneapi/experimental/enqueue_functions.hpp:* {{static assertion failed due to requirement {{.*}} This kernel enqueue function does not allow properties with compile-time kernel effects.}}
    oneapi::launch_config<sycl::range<3>, decltype(props2)> LC{{1, 1, 1},
                                                               props2};
    oneapi::parallel_for(Q, LC, TestKernel);
  }

  {
    // expected-error-re@sycl/ext/oneapi/experimental/enqueue_functions.hpp:* {{static assertion failed due to requirement {{.*}} This kernel enqueue function does not allow properties with compile-time kernel effects.}}
    oneapi::launch_config<sycl::nd_range<2>, decltype(props1)> LC{
        {{1, 1}, {1, 1}}, props1};
    oneapi::nd_launch(Q, LC, TestKernel);
  }

  {
    // expected-error-re@sycl/ext/oneapi/experimental/enqueue_functions.hpp:* {{static assertion failed due to requirement {{.*}} This kernel enqueue function does not allow properties with compile-time kernel effects.}}
    oneapi::launch_config<sycl::nd_range<2>, decltype(props2)> LC{
        {{1, 1}, {1, 1}}, props2};
    oneapi::nd_launch(Q, LC, TestKernel);
  }

  {
    // expected-error-re@sycl/ext/oneapi/experimental/enqueue_functions.hpp:* {{static assertion failed due to requirement {{.*}} This kernel enqueue function does not allow properties with compile-time kernel effects.}}
    oneapi::launch_config<sycl::nd_range<3>, decltype(props1)> LC{
        {{1, 1, 1}, {1, 1, 1}}, props1};
    oneapi::nd_launch(Q, LC, TestKernel);
  }

  {
    // expected-error-re@sycl/ext/oneapi/experimental/enqueue_functions.hpp:* {{static assertion failed due to requirement {{.*}} This kernel enqueue function does not allow properties with compile-time kernel effects.}}
    oneapi::launch_config<sycl::nd_range<3>, decltype(props2)> LC{
        {{1, 1, 1}, {1, 1, 1}}, props2};
    oneapi::nd_launch(Q, LC, TestKernel);
  }
}
