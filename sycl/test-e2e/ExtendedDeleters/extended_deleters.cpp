// REQUIRES: hip, cuda

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out | FileCheck %s

// CHECK: This extended deleter should be called at ctx destruction.

#include <sycl/detail/core.hpp>

int main() {
  sycl::context c;
  sycl::detail::pi::contextSetExtendedDeleter(
      c,
      [](void *) {
        printf("This extended deleter should be called at ctx destruction.");
      },
      nullptr);
}
