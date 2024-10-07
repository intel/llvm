// REQUIRES: native_cpu_ock

// Tests that no linker errors occur when group collective functions are used
// in conjuction with SYCL_EXTERNAL.

// RUN: %clangxx -fsycl -fsycl-targets=native_cpu -DFILE1 -c -o %t1.o %s
// RUN: %clangxx -fsycl -fsycl-targets=native_cpu -DFILE2 -c -o %t2.o %s
// RUN: llvm-ar crv %t1.a %t1.o
// RUN: %clangxx -fsycl -fsycl-targets=native_cpu %t2.o %t1.a -o %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR=native_cpu:cpu %t.out

/*
    test performs a lattice reduction.
    sycl::vec<float> is sensitive to .get_size() vs .size() in SYCL headers
    (ie, byte size versus vector size)
*/

#include <sycl/detail/core.hpp>
#include <sycl/group_algorithm.hpp>
#include <sycl/usm.hpp>

using namespace sycl;

#define NX 32
#define NZ 2
#define NV 8
using vecn = sycl::vec<float, NV>; // 8 floats
#ifdef FILE1

SYCL_EXTERNAL void groupSum(vecn *r, const vecn &in, const int k,
                            sycl::group<2> &grp, const int i) {

  vecn tin = (i == k ? in : vecn(0));
  auto out = reduce_over_group(grp, tin, sycl::plus<>());
  if (i == k && grp.get_local_id()[1] == 0)
    r[k] = out;
}
#endif

#ifdef FILE2
SYCL_EXTERNAL void groupSum(vecn *r, const vecn &in, const int k,
                            sycl::group<2> &grp, const int i);
void test(queue q, float *r, float *x,
          int n) { // r is 16 floats, x is 256 floats. n is 256

  sycl::range<2> globalSize(NZ, NX); // 2,32
  sycl::range<2> localSize(1, NX);   // 1,8       so 16 iterations
  sycl::nd_range<2> range{globalSize, localSize};

  q.submit([&](sycl::handler &h) {
    h.parallel_for<>(range, [=](sycl::nd_item<2> ndi) {
      int i = ndi.get_global_id(1);
      int k = ndi.get_global_id(0);

      auto vx = reinterpret_cast<vecn *>(x);
      auto vr = reinterpret_cast<vecn *>(r);

      auto myg = ndi.get_group();

      for (int iz = 0; iz < NZ; iz++) { // loop over Z (2)
        groupSum(vr, vx[k * NX + i], k, myg, iz);
      }
    });
  });
  q.wait();
}

int main() {

  queue q{default_selector_v};
  auto dev = q.get_device();
  std::cout << "Device: " << dev.get_info<info::device::name>() << std::endl;

  auto ctx = q.get_context();
  int n = NX * NZ * NV; // 16 * 8 * 2 => 256
  auto *x = (float *)sycl::malloc_shared(n * sizeof(float), dev,
                                         ctx); // 256 * sizeof(float)
  auto *r = (float *)sycl::malloc_shared(
      NZ * NV * sizeof(float), dev, ctx); // 2 * 8 => 16   ( * sizeof(float) )

  for (int i = 0; i < n; i++) {
    x[i] = i;
  }

  q.wait();

  test(q, r, x, n);

  int fails = 0;
  for (int k = 0; k < NZ; k++) {
    float s[NV] = {0};
    for (int i = 0; i < NX; i++) {
      for (int j = 0; j < NV; j++) {
        s[j] += x[(k * NX + i) * NV + j];
      }
    }
    for (int j = 0; j < NV; j++) {
      auto d = s[j] - r[k * NV + j];
      if (std::abs(d) > 1e-10) {
        printf("partial fail ");
        printf("%i\t%i\t%g\t%g\n", k, j, s[j], r[k * NV + j]);
        fails++;
      } else {
        printf("partial pass ");
        printf("%i\t%i\t%g\t%g\n", k, j, s[j], r[k * NV + j]);
      }
    }
  }

  if (fails == 0) {
    printf("test passed!\n");
  } else {
    printf("test failed!\n");
  }
  free(x, ctx);
  free(r, ctx);
  return fails;
}
#endif
