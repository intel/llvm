// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

size_t copy_count = 0;
size_t move_count = 0;

template <int N> class kernel {
public:
  kernel() {};
  kernel(const kernel &other) { copy_count++; };
  kernel(kernel &&other) { ++move_count; }

  void operator()(sycl::id<1> id) const {}
  void operator()(sycl::nd_item<1> id) const {}
  void operator()() const {}
};
template <int N> struct sycl::is_device_copyable<kernel<N>> : std::true_type {};

int main(int argc, char **argv) {
  sycl::queue q;

  kernel<0> krn0;
  q.parallel_for(sycl::range<1>{1}, krn0);
  assert(copy_count == 1);
  assert(move_count == 0);
  copy_count = 0;

  kernel<1> krn1;
  q.parallel_for(sycl::nd_range<1>{1, 1}, krn1);
  assert(copy_count == 1);
  assert(move_count == 0);
  copy_count = 0;

  kernel<2> krn2;
  q.single_task(krn2);
  assert(copy_count == 1);
  assert(move_count == 0);
  copy_count = 0;

  return 0;
}
