// UNSUPPORTED: true
// RUN: %{build} %device_sanitizer_flags -g -o %t.out
// RUN: %{run} %t.out 2>&1 | FileCheck %s
#include <sycl/sycl.hpp>

using namespace syclex = sycl::ext::oneapi::experimental;
constexpr std::size_t N = 8ULL;
constexpr std::size_t group_size = 8;

// optional: static const
syclex::work_group_local<int[]> dynamic_program_scope_array;

int main() {
  sycl::queue Q;

  Q.submit([&](sycl::handler &h) {
    Q.parallel_for(
        sycl::nd_range<1>{N, group_size},
        syclex::properties{syclex::work_group_local_size(M * sizeof(int))},
        [=](sycl::nd_item<1> it) {
          (*dynamic_program_scope_array)[it.get_local_id(0)] = 0;
        });
  });

  Q.wait();
  return 0;
}