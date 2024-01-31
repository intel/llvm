// REQUIRES: usm_shared_allocations
// DISABLED: aspect-atomic64
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/sycl.hpp>
using namespace sycl;

using AtomicRefT =
    atomic_ref<unsigned long long, memory_order::relaxed, memory_scope::device>;

int main() {
  queue q;
  auto *p = malloc_shared<unsigned long long>(1, q);
  try {
    q.submit([&](sycl::handler &cgh) {
       cgh.parallel_for_work_group(range{1}, range{1}, [=](group<1>) {
         AtomicRefT feature(*p);
         feature += 42;
       });
     }).wait();
  } catch (sycl::exception &e) {
    if (e.code() != sycl::errc::kernel_not_supported)
      throw;
    std::cout << "Caught right exception: " << e.what() << "\n";
    return 0;
  }
}
