// DISABLED: aspect-atomic64
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

#include <sycl/atomic_ref.hpp>

using namespace sycl;

using AtomicRefT =
    atomic_ref<unsigned long long, memory_order::relaxed, memory_scope::device>;

int main() {
  queue q;
  sycl::buffer<unsigned long long> p_buf{sycl::range{1}};
  try {
    q.submit([&](sycl::handler &cgh) {
       sycl::accessor p{p_buf, cgh};
       cgh.parallel_for_work_group(range{1}, range{1}, [=](group<1>) {
         AtomicRefT feature(p[0]);
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
