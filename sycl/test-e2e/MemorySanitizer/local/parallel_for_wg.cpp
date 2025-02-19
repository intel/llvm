#include <sycl/usm.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;

constexpr std::size_t global_size = 4;
constexpr std::size_t local_size = 1;

__attribute__((noinline)) int check(int data) { return data; }

int main() {
  sycl::queue Q;
  auto data = sycl::malloc_device<int>(global_size, Q);

  Q.submit([&](handler &cgh) {
    cgh.parallel_for_work_group(range<1>(global_size), range<1>(local_size), [=](group<1> myGroup) {
      auto j = myGroup.get_group_id(0);
      myGroup.parallel_for_work_item(
          [&](h_item<1> it) { A[(j * 2) + it.get_local_id(0)]++; });
    });
  });

  Q.wait();
  sycl::free(data, Q);
  return 0;
}
