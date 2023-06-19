// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  device d(default_selector_v);
  context ctx{d};
  queue q{ctx, d};

  int WGSize = 256;
  // Reduction implementation would spawn several other kernels to reduce
  // partial sums. At some point the number of partial sums won't be divisible
  // by the WG size and the code needs to adjust it for that. Ensure that is
  // done.
  int N = 22500 * 256;

  auto *data = malloc_device<int>(N, q);
  auto *r1 = malloc_device<int>(1, q);
  buffer<int, 1> r2buf(1);

  q.fill(data, 1, N).wait();
  q.fill(r1, 0, 1).wait();

  q.submit([&](handler &cgh) {
     cgh.parallel_for(
         nd_range(range(N), range(WGSize)), reduction(r1, std::plus<int>()),
         reduction(r2buf, cgh, std::plus<int>(),
                   {property::reduction::initialize_to_identity{}}),
         [=](auto id, auto &r1, auto &r2) {
           r1 += 1;
           r2 += 2;
         });
   }).wait();

  int res1, res2;
  q.copy(r1, &res1, 1).wait();
  auto r2acc = host_accessor{r2buf};
  res2 = r2acc[0];
  assert(res1 == N && res2 == 2 * N);

  free(r1, q);
  free(data, q);

  return 0;
}
