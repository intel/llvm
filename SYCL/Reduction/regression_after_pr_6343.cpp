// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  device d(default_selector{});
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
     auto r2 = r2buf.get_access<access::mode::discard_write>(cgh);
     cgh.parallel_for(nd_range(range(N), range(WGSize)),
                      sycl::reduction(r1, std::plus<int>()),
                      ext::oneapi::reduction(r2, std::plus<int>()),
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
