// RUN: %clangxx -fsycl %s -o %t.out

// Tests that the assignment operator on atomic_ref for pointers do not cause
// overload resolution ambiguities.

#include <sycl/sycl.hpp>

#include <type_traits>

using AtomicRT = sycl::atomic_ref<int *, sycl::memory_order::relaxed,
                                  sycl::memory_scope::system,
                                  sycl::access::address_space::global_space>;

int main() {
  int OriginalVal = 2023;
  int NewVal = 1;
  int *OriginalValPtr = &OriginalVal;

  AtomicRT AtomicRef(OriginalValPtr);
  int *Desired = (AtomicRef = &NewVal);

  return 0;
}
