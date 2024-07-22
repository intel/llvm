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

  Q.submit([&](sycl::handler &CGH) {
    // expected-error-re@sycl/handler.hpp:* {{static assertion failed due to requirement {{.*}} This kernel enqueue function does not allow properties with compile-time kernel effects.}}
    CGH.single_task(props1, TestKernel);
  });

  Q.submit([&](sycl::handler &CGH) {
    // expected-error-re@sycl/handler.hpp:* {{static assertion failed due to requirement {{.*}} This kernel enqueue function does not allow properties with compile-time kernel effects.}}
    CGH.single_task(props2, TestKernel);
  });

  Q.submit([&](sycl::handler &CGH) {
    // expected-error-re@sycl/handler.hpp:* {{static assertion failed due to requirement {{.*}} This kernel enqueue function does not allow properties with compile-time kernel effects.}}
    CGH.parallel_for(sycl::range<1>{1}, props1, TestKernel);
  });

  Q.submit([&](sycl::handler &CGH) {
    // expected-error-re@sycl/handler.hpp:* {{static assertion failed due to requirement {{.*}} This kernel enqueue function does not allow properties with compile-time kernel effects.}}
    CGH.parallel_for(sycl::range<1>{1}, props2, TestKernel);
  });

  Q.submit([&](sycl::handler &CGH) {
    // expected-error-re@sycl/handler.hpp:* {{static assertion failed due to requirement {{.*}} This kernel enqueue function does not allow properties with compile-time kernel effects.}}
    CGH.parallel_for(sycl::range<2>{1, 1}, props1, TestKernel);
  });

  Q.submit([&](sycl::handler &CGH) {
    // expected-error-re@sycl/handler.hpp:* {{static assertion failed due to requirement {{.*}} This kernel enqueue function does not allow properties with compile-time kernel effects.}}
    CGH.parallel_for(sycl::range<2>{1, 1}, props2, TestKernel);
  });

  Q.submit([&](sycl::handler &CGH) {
    // expected-error-re@sycl/handler.hpp:* {{static assertion failed due to requirement {{.*}} This kernel enqueue function does not allow properties with compile-time kernel effects.}}
    CGH.parallel_for(sycl::range<3>{1, 1, 1}, props1, TestKernel);
  });

  Q.submit([&](sycl::handler &CGH) {
    // expected-error-re@sycl/handler.hpp:* {{static assertion failed due to requirement {{.*}} This kernel enqueue function does not allow properties with compile-time kernel effects.}}
    CGH.parallel_for(sycl::range<3>{1, 1, 1}, props2, TestKernel);
  });

  Q.submit([&](sycl::handler &CGH) {
    // expected-error-re@sycl/handler.hpp:* {{static assertion failed due to requirement {{.*}} This kernel enqueue function does not allow properties with compile-time kernel effects.}}
    CGH.parallel_for(sycl::nd_range<1>{sycl::range<1>{1}, sycl::range<1>{1}},
                     props1, TestKernel);
  });

  Q.submit([&](sycl::handler &CGH) {
    // expected-error-re@sycl/handler.hpp:* {{static assertion failed due to requirement {{.*}} This kernel enqueue function does not allow properties with compile-time kernel effects.}}
    CGH.parallel_for(sycl::nd_range<1>{sycl::range<1>{1}, sycl::range<1>{1}},
                     props2, TestKernel);
  });

  Q.submit([&](sycl::handler &CGH) {
    // expected-error-re@sycl/handler.hpp:* {{static assertion failed due to requirement {{.*}} This kernel enqueue function does not allow properties with compile-time kernel effects.}}
    CGH.parallel_for(
        sycl::nd_range<2>{sycl::range<2>{1, 1}, sycl::range<2>{1, 1}}, props1,
        TestKernel);
  });

  Q.submit([&](sycl::handler &CGH) {
    // expected-error-re@sycl/handler.hpp:* {{static assertion failed due to requirement {{.*}} This kernel enqueue function does not allow properties with compile-time kernel effects.}}
    CGH.parallel_for(
        sycl::nd_range<2>{sycl::range<2>{1, 1}, sycl::range<2>{1, 1}}, props2,
        TestKernel);
  });

  Q.submit([&](sycl::handler &CGH) {
    // expected-error-re@sycl/handler.hpp:* {{static assertion failed due to requirement {{.*}} This kernel enqueue function does not allow properties with compile-time kernel effects.}}
    CGH.parallel_for(
        sycl::nd_range<3>{sycl::range<3>{1, 1, 1}, sycl::range<3>{1, 1, 1}},
        props1, TestKernel);
  });

  Q.submit([&](sycl::handler &CGH) {
    // expected-error-re@sycl/handler.hpp:* {{static assertion failed due to requirement {{.*}} This kernel enqueue function does not allow properties with compile-time kernel effects.}}
    CGH.parallel_for(
        sycl::nd_range<3>{sycl::range<3>{1, 1, 1}, sycl::range<3>{1, 1, 1}},
        props2, TestKernel);
  });
}
