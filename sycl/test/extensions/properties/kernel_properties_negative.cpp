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

  // expected-error-re@sycl/ext/oneapi/experimental/enqueue_functions.hpp:* {{static assertion failed due to requirement {{.*}} launch_config does not allow properties with compile-time kernel effects.}}
  oneapi::launch_config<sycl::range<1>, decltype(props1)> LC1{{1}, props1};

  // expected-error-re@sycl/ext/oneapi/experimental/enqueue_functions.hpp:* {{static assertion failed due to requirement {{.*}} launch_config does not allow properties with compile-time kernel effects.}}
  oneapi::launch_config<sycl::range<1>, decltype(props2)> LC22{{1}, props2};
}
