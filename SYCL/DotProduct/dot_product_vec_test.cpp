// This test checks dp4a support with vec<> arguments
// For now we only check fallback support because DG1 hardware is not widespread

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <iostream>
#include <memory>
#include <stdio.h>
#include <sycl/ext/oneapi/dot_product.hpp>
#include <sycl/sycl.hpp>

// Change if tests are added/removed
static int testCount = 4;
static int passCount;

using namespace sycl;
using namespace sycl::detail::gtl;
using namespace sycl::ext::oneapi;

constexpr int RangeLength = 100;

// Verify 1D array
template <typename T>
static bool verify_1D(const char *name, int X, T *A, T *A_ref) {
  int ErrCnt = 0;

  for (int i = 0; i < X; i++) {
    if (A_ref[i] != A[i]) {
      if (++ErrCnt < 10) {
        std::cout << name << " mismatch at " << i << ". Expected " << A_ref[i]
                  << " result is " << A[i] << "\n";
      }
    }
  }

  if (ErrCnt == 0) {
    return true;
  }
  std::cout << "  Failed. Failure rate: " << ErrCnt << "/" << X << "("
            << ErrCnt / (float)X * 100.f << "%)\n";
  return false;
}

static bool testss(queue &Q) {
  vec<int8_t, 4> A[RangeLength];
  vec<int8_t, 4> B[RangeLength];
  int32_t C[RangeLength];
  int32_t D[RangeLength];
  int32_t D_ref[RangeLength];

  std::memset(D, 0, RangeLength * sizeof(int));
  std::memset(D_ref, 0, RangeLength * sizeof(int));

  for (int i = 0; i < RangeLength; i++) {
    A[i].s0() = A[i].s1() = A[i].s2() = A[i].s3() = i;
    B[i].s0() = B[i].s1() = B[i].s2() = B[i].s3() = 0xFF;
    C[i] = i;
  }
  for (int i = 0; i < RangeLength; i++) {
    D_ref[i] = 4 * (i * -1) + C[i];
  }

  buffer<vec<int8_t, 4>, 1> Abuf(A, range<1>(RangeLength));
  buffer<vec<int8_t, 4>, 1> Bbuf(B, range<1>(RangeLength));
  buffer<int32_t, 1> Cbuf(C, range<1>(RangeLength));
  buffer<int32_t, 1> Dbuf(D, range<1>(RangeLength));

  Q.submit([&](handler &cgh) {
    auto Ap = Abuf.get_access<access::mode::read>(cgh);
    auto Bp = Bbuf.get_access<access::mode::read>(cgh);
    auto Cp = Cbuf.get_access<access::mode::read>(cgh);
    auto Dp = Dbuf.get_access<access::mode::write>(cgh);

    cgh.parallel_for<class tss>(range<1>(RangeLength), [=](id<1> I) {
      Dp[I] = dot_acc(Ap[I], Bp[I], Cp[I]);
    });
  });
  const auto HAcc = Dbuf.get_access<sycl::access::mode::read>();

  return verify_1D("testss D", RangeLength, D, D_ref);
}

static bool testuu(queue &Q) {
  vec<uint8_t, 4> A[RangeLength];
  vec<uint8_t, 4> B[RangeLength];
  int32_t C[RangeLength];
  int32_t D[RangeLength];
  int32_t D_ref[RangeLength];

  std::memset(D, 0, RangeLength * sizeof(int));
  std::memset(D_ref, 0, RangeLength * sizeof(int));

  for (int i = 0; i < RangeLength; i++) {
    A[i].s0() = A[i].s1() = A[i].s2() = A[i].s3() = i;
    B[i].s0() = B[i].s1() = B[i].s2() = B[i].s3() = 0xFF;
    C[i] = i;
  }
  for (int i = 0; i < RangeLength; i++) {
    D_ref[i] = 4 * (i * 255) + C[i];
  }

  buffer<vec<uint8_t, 4>, 1> Abuf(A, range<1>(RangeLength));
  buffer<vec<uint8_t, 4>, 1> Bbuf(B, range<1>(RangeLength));
  buffer<int32_t, 1> Cbuf(C, range<1>(RangeLength));
  buffer<int32_t, 1> Dbuf(D, range<1>(RangeLength));

  Q.submit([&](handler &cgh) {
    auto Ap = Abuf.get_access<access::mode::read>(cgh);
    auto Bp = Bbuf.get_access<access::mode::read>(cgh);
    auto Cp = Cbuf.get_access<access::mode::read>(cgh);
    auto Dp = Dbuf.get_access<access::mode::write>(cgh);

    cgh.parallel_for<class tuu>(range<1>(RangeLength), [=](id<1> I) {
      Dp[I] = dot_acc(Ap[I], Bp[I], Cp[I]);
    });
  });
  const auto HAcc = Dbuf.get_access<sycl::access::mode::read>();

  return verify_1D("testuu D", RangeLength, D, D_ref);
}

static bool testsu(queue &Q) {
  vec<int8_t, 4> A[RangeLength];
  vec<uint8_t, 4> B[RangeLength];
  int32_t C[RangeLength];
  int32_t D[RangeLength];
  int32_t D_ref[RangeLength];

  std::memset(D, 0, RangeLength * sizeof(int));
  std::memset(D_ref, 0, RangeLength * sizeof(int));

  for (int i = 0; i < RangeLength; i++) {
    A[i].s0() = A[i].s1() = A[i].s2() = A[i].s3() = 0xFF;
    B[i].s0() = B[i].s1() = B[i].s2() = B[i].s3() = i;
    C[i] = i;
  }
  for (int i = 0; i < RangeLength; i++) {
    D_ref[i] = 4 * (i * -1) + C[i];
  }

  buffer<vec<int8_t, 4>, 1> Abuf(A, range<1>(RangeLength));
  buffer<vec<uint8_t, 4>, 1> Bbuf(B, range<1>(RangeLength));
  buffer<int32_t, 1> Cbuf(C, range<1>(RangeLength));
  buffer<int32_t, 1> Dbuf(D, range<1>(RangeLength));

  Q.submit([&](handler &cgh) {
    auto Ap = Abuf.get_access<access::mode::read>(cgh);
    auto Bp = Bbuf.get_access<access::mode::read>(cgh);
    auto Cp = Cbuf.get_access<access::mode::read>(cgh);
    auto Dp = Dbuf.get_access<access::mode::write>(cgh);

    cgh.parallel_for<class tsu>(range<1>(RangeLength), [=](id<1> I) {
      Dp[I] = dot_acc(Ap[I], Bp[I], Cp[I]);
    });
  });
  const auto HAcc = Dbuf.get_access<sycl::access::mode::read>();

  return verify_1D("testsu D", RangeLength, D, D_ref);
}

static bool testus(queue &Q) {
  vec<uint8_t, 4> A[RangeLength];
  vec<int8_t, 4> B[RangeLength];
  int32_t C[RangeLength];
  int32_t D[RangeLength];
  int32_t D_ref[RangeLength];

  std::memset(D, 0, RangeLength * sizeof(int));
  std::memset(D_ref, 0, RangeLength * sizeof(int));

  for (int i = 0; i < RangeLength; i++) {
    A[i].s0() = A[i].s1() = A[i].s2() = A[i].s3() = i;
    B[i].s0() = B[i].s1() = B[i].s2() = B[i].s3() = 0xFF;
    C[i] = i;
  }
  for (int i = 0; i < RangeLength; i++) {
    D_ref[i] = 4 * (i * -1) + C[i];
  }

  buffer<vec<uint8_t, 4>, 1> Abuf(A, range<1>(RangeLength));
  buffer<vec<int8_t, 4>, 1> Bbuf(B, range<1>(RangeLength));
  buffer<int32_t, 1> Cbuf(C, range<1>(RangeLength));
  buffer<int32_t, 1> Dbuf(D, range<1>(RangeLength));

  Q.submit([&](handler &cgh) {
    auto Ap = Abuf.get_access<access::mode::read>(cgh);
    auto Bp = Bbuf.get_access<access::mode::read>(cgh);
    auto Cp = Cbuf.get_access<access::mode::read>(cgh);
    auto Dp = Dbuf.get_access<access::mode::write>(cgh);

    cgh.parallel_for<class tus>(range<1>(RangeLength), [=](id<1> I) {
      Dp[I] = dot_acc(Ap[I], Bp[I], Cp[I]);
    });
  });
  const auto HAcc = Dbuf.get_access<sycl::access::mode::read>();

  return verify_1D("testus D", RangeLength, D, D_ref);
}

bool run_tests() {
  queue Q([](exception_list L) {
    for (auto ep : L) {
      try {
        std::rethrow_exception(ep);
      } catch (std::exception &E) {
        std::cout << "*** std exception caught:\n";
        std::cout << E.what();
      } catch (sycl::exception const &E1) {
        std::cout << "*** SYCL exception caught:\n";
        std::cout << E1.what();
      }
    }
  });

  passCount = 0;
  if (testss(Q)) {
    ++passCount;
  }
  if (testuu(Q)) {
    ++passCount;
  }
  if (testsu(Q)) {
    ++passCount;
  }
  if (testus(Q)) {
    ++passCount;
  }

  auto D = Q.get_device();
  const char *devType = D.is_host() ? "Host" : D.is_cpu() ? "CPU" : "GPU";
  std::cout << passCount << " of " << testCount << " tests passed on "
            << devType << "\n";

  return (testCount == passCount);
}

int main(int argc, char *argv[]) {
  bool passed = true;
  default_selector selector{};
  auto D = selector.select_device();
  const char *devType = D.is_host() ? "Host" : D.is_cpu() ? "CPU" : "GPU";
  std::cout << "Running on device " << devType << " ("
            << D.get_info<sycl::info::device::name>() << ")\n";
  try {
    passed &= run_tests();
  } catch (exception e) {
    std::cout << e.what();
  }

  if (!passed) {
    std::cout << "FAILED\n";
    return 1;
  }
  std::cout << "PASSED\n";
  return 0;
}
