// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/sycl.hpp>

using namespace sycl;
#define ALLOC_SIZE 0

int main() {
  queue q;
  auto dev = q.get_device();
  auto ctxt = q.get_context();

  if (dev.get_info<info::device::usm_host_allocations>()) {
    sycl::usm_allocator<int, sycl::usm::alloc::host> ua{ctxt, dev};
    int *p = ua.allocate(ALLOC_SIZE);

    assert(!p && "usm_allocator should return a null pointer when allocation "
                 "size is zero.");

    ua.deallocate(p, ALLOC_SIZE);
  }

  return 0;
}
