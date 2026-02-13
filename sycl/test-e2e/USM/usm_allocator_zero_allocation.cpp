// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>
#include <sycl/usm/usm_allocator.hpp>

using namespace sycl;

template <usm::alloc alloc_kind> void test(queue &q) {
  sycl::usm_allocator<int, alloc_kind> ua(q);
  int *p = ua.allocate(0);

  assert(!p && "Our implementation of usm_allocator is expected to return a "
               "null pointer when allocation size is zero.");

  ua.deallocate(p, 0);
}

int main() {
  queue q;
  auto dev = q.get_device();

  if (dev.has(aspect::usm_host_allocations)) {
    test<usm::alloc::host>(q);
  }
  if (dev.has(aspect::usm_shared_allocations)) {
    test<usm::alloc::shared>(q);
  }

  return 0;
}
