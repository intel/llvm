#include <cstdio>
#include <cstdlib>
#include <numeric>

#include <sycl/sycl.hpp>

namespace syclex = sycl::ext::oneapi::experimental;

int sum(sycl::queue q, int *input, size_t N) {

  int result = 42;
  {
    sycl::buffer<int> buf{&result, 1};

    q.submit([&](sycl::handler &h) {
      auto reduction =
          sycl::reduction(buf, h, sycl::plus<>(),
                          syclex::properties(syclex::initialize_to_identity));
      h.parallel_for(N, reduction,
                     [=](size_t i, auto &reducer) { reducer += input[i]; });
    });
  }
  return result;
}

int main(int argc, char *argv[]) {

  constexpr size_t N = 32;
  int *array = new int[N];
  std::iota(array, array + N, 1);

  sycl::queue q;
  int x = sum(q, array, N);
  assert(x == 528);

  delete[] array;
}
