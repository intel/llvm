// REQUIRES: native_cpu_be
// RUN: %clangxx -fsycl -fsycl-targets=native_cpu -fno-inline -O0  %s -o %t
// RUN: env ONEAPI_DEVICE_SELECTOR="native_cpu:cpu" %t

#include <sycl/sycl.hpp>

#include <array>
#include <iostream>

constexpr sycl::access::mode sycl_read = sycl::access::mode::read;
constexpr sycl::access::mode sycl_write = sycl::access::mode::write;

class Test;

static constexpr int DEVICE_RET = 1;
static constexpr int HOST_RET = 2;

#ifdef __SYCL_DEVICE_ONLY__
#define RET_VAL DEVICE_RET
#else
#define RET_VAL HOST_RET
#endif

#define FTy __attribute__((noinline)) int

FTy get_val() { return RET_VAL; }

static FTy get_val2() { return RET_VAL; }

struct str {
  FTy m1() { return RET_VAL; }
  static FTy m2() { return RET_VAL; }
  FTy m3();
  static FTy m4();
};

FTy str::m3() { return RET_VAL; }

FTy str::m4() { return RET_VAL; }

int test_all(int expect_val) {
  if (get_val() != expect_val)
    return -1;
  if (get_val2() != expect_val)
    return -1;
  if (str().m1() != expect_val)
    return -1;
  if (str::m2() != expect_val)
    return -1;
  if (str().m3() != expect_val)
    return -1;
  if (str::m4() != expect_val)
    return -1;
  return expect_val;
}

int main() {
  const size_t N = 4;
  std::array<int, N> C{{-6, -6, -6, -6}};
  sycl::queue deviceQueue;
  sycl::range<1> numOfItems{N};
  {
    sycl::buffer<int, 1> bufferC(C.data(), numOfItems);

    if (test_all(HOST_RET) != HOST_RET)
      return 1;

    deviceQueue
        .submit([&](sycl::handler &cgh) {
          auto accessorC = bufferC.get_access<sycl_write>(cgh);

          auto kern = [=](sycl::id<1> wiID) {
            accessorC[wiID] = test_all(DEVICE_RET);
          };
          cgh.parallel_for<class SimpleVadd>(numOfItems, kern);
        })
        .wait();
  }

  bool pass = true;
  for (unsigned int i = 0; i < N; i++) {
    if (C[i] != DEVICE_RET) {
      std::cout << "The results are incorrect (element " << i << " is " << C[i]
                << "!\n";
      pass = false;
    }
  }
  if (pass) {
    std::cout << "The results are correct!\n";
    return 0;
  }
  return 2;
}
