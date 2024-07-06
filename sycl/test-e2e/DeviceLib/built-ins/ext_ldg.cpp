// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %t.out

// Checks that the sycl::ext::oneapi::experimental::cuda::ldg builtins are
// returning the correct values.

#include <sycl/ext/oneapi/experimental/cuda/builtins.hpp>
#include <sycl/sycl.hpp>

using namespace sycl::ext::oneapi::experimental::cuda;
using namespace sycl::ext::oneapi::experimental;
using namespace sycl;

template <typename T1> class KernelName;

template <typename T> bool checkEqual(vec<T, 2> A, vec<T, 2> B) {
  return A.x() == B.x() && A.y() == B.y();
}

template <typename T> bool checkEqual(vec<T, 4> A, vec<T, 4> B) {
  return A.x() == B.x() && A.y() == B.y() && A.z() == B.z() && A.w() == B.w();
}

template <typename T> void test(sycl::queue &q) {

  T a_loc;
  T b_loc;

  a_loc = 2;
  b_loc = 3;

  T *A = malloc_device<T>(1, q);
  T *B = malloc_device<T>(1, q);
  T *C = malloc_device<T>(1, q);

  q.memcpy(A, &a_loc, sizeof(T));
  q.memcpy(B, &b_loc, sizeof(T));
  q.wait();

  q.submit([=](sycl::handler &h) {
    h.parallel_for<KernelName<T>>(range<1>(1), [=](auto i) {
      auto cacheA = ldg(&A[0]);
      auto cacheB = ldg(&B[0]);
      C[0] = cacheA + cacheB;
    });
  });

  T dev_result;
  q.wait();

  q.memcpy(&dev_result, C, sizeof(T)).wait();
  if constexpr (std::is_same_v<T, char> || std::is_same_v<T, short> ||
                std::is_same_v<T, int> || std::is_same_v<T, long> ||
                std::is_same_v<T, long long> ||
                std::is_same_v<T, unsigned char> ||
                std::is_same_v<T, unsigned short> ||
                std::is_same_v<T, unsigned int> ||
                std::is_same_v<T, unsigned long long> ||
                std::is_same_v<T, double> || std::is_same_v<T, float>) {

    assert(dev_result == a_loc + b_loc);
  } else {
    assert(checkEqual(dev_result, a_loc + b_loc));
  }

  free(A, q);
  free(B, q);
  free(C, q);
}

int main() {
  queue q;

  test<char>(q);
  test<short>(q);
  test<int>(q);
  test<long>(q);
  test<long long>(q);

  test<unsigned char>(q);
  test<unsigned short>(q);
  test<unsigned int>(q);

  test<unsigned long long>(q);

  test<char2>(q);
  test<int2>(q);
  test<longlong2>(q);

  test<uchar2>(q);
  test<uint2>(q);
  test<ulonglong2>(q);

  test<char4>(q);
  test<short4>(q);
  test<int4>(q);

  test<uchar4>(q);
  test<ushort4>(q);
  test<uint4>(q);

  test<float>(q);
  test<float2>(q);
  test<float4>(q);

  if (q.get_device().has(sycl::aspect::fp64)) {
    test<double>(q);
    test<double2>(q);
  }
  return 0;
}
