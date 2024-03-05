// This test ensures that this_id returns the correct value
// even when a kernel is wrapped in a range rounding kernel.
// RUN: %{build} -o %t.out
// RUN: env SYCL_PARALLEL_FOR_RANGE_ROUNDING_PARAMS=16:32:0 \
// RUN:     SYCL_PARALLEL_FOR_RANGE_ROUNDING_TRACE=1 \
// RUN: %{run} %t.out | FileCheck %s
#include <sycl/sycl.hpp>

constexpr int N = 3;

using namespace sycl;

template <int D> std::ostream &operator<<(std::ostream &os, id<D> it) {
  os << "[";
  for (int i = 0; i < D; ++i) {
    os << it[i];
    if (i + 1 != D)
      os << ", ";
  }
  return os << "]";
}

int n_fail = 0;

template <int D> void test(queue &q) {
  range<D> range = {};
  for (int i = 0; i < D; ++i)
    range[i] = N;
  using T = struct {
    id<D> this_id;
    id<D> ref_id;
  };
  std::vector<T> vec(range.size());
  {
    sycl::buffer<T> p_buf{vec};
    q.submit([&](sycl::handler &h) {
       sycl::accessor p{p_buf, h};
       h.parallel_for(range, [=](auto it) {
         p[it.get_linear_id()] = {sycl::ext::oneapi::experimental::this_id<D>(),
                                  it.get_id()};
       });
     }).wait_and_throw();
  } // p_buf goes out of scope here and writed back to vec
  for (const auto &[this_item, ref_item] : vec) {
    if (this_item != ref_item) {
      std::cout << "fail: " << this_item << " != " << ref_item << "\n";
      ++n_fail;
    }
  }
}

int main() {
  queue q;
  // CHECK: parallel_for range adjusted at dim 0 from 3 to 32
  test<1>(q);
  // CHECK: parallel_for range adjusted at dim 0 from 3 to 32
  test<2>(q);
  // CHECK: parallel_for range adjusted at dim 0 from 3 to 32
  test<3>(q);
  if (n_fail == 0)
    std::cout << "pass\n";
  return n_fail != 0;
}
