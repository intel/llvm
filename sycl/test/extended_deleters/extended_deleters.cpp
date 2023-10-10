// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out | FileCheck %s

// CHECK: This extended deleter should be called at ctx destruction.

#include <sycl/sycl.hpp>

int main() {
  sycl::context c;
  sycl::detail::pi::contextSetExtendedDeleter(
      c,
      [](void *) {
        printf("This extended deleter should be called at ctx destruction.");
      },
      nullptr);
}
