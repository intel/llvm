// RUN: %{build} -Wno-absolute-value -Wno-deprecated-declarations -o %t.out
// RUN: %{run} %t.out

#include <array>
#include <sycl/builtins.hpp>
#include <sycl/detail/core.hpp>

using namespace sycl;

constexpr size_t BufferSize = 16;
constexpr size_t NElems = 32;
constexpr size_t WorkGroupSize = 8;

#if 0
template <typename T>
SYCL_EXTERNAL void test_builtins(
    T *out, T x, T y, sycl::vec<T, 4> *out2, sycl::vec<float, 4> f2,
    sycl::nd_item<2> item,
    sycl::multi_ptr<T, sycl::access::address_space::global_space> ptr_global,
    sycl::multi_ptr<T, sycl::access::address_space::local_space> ptr_local) {
  size_t num_elem = 4;
  const auto group = item.get_group();
  group.async_work_group_copy(ptr_local, ptr_global, num_elem);
}
#endif

template <typename T>
int check(const T *A, const T *B, T *C, const vec<float, 2> FVec) {
#define UNARY_CHECK(IDX, OP)                                                   \
  assert(C[IDX] == OP(A[IDX]) && "error: " #OP "failed")

  UNARY_CHECK(0, clz);
  UNARY_CHECK(1, ctz);
  UNARY_CHECK(2, std::abs);

#define BINARY_CHECK(IDX, OP)                                                  \
  assert(C[IDX] == OP(A[IDX], B[IDX]) && "error: " #OP "failed")

  BINARY_CHECK(3, std::min);
  BINARY_CHECK(4, std::max);

  auto hadd = [](auto x, auto y) {
    return (static_cast<int>(x) + static_cast<int>(y)) >> 1;
  };
  BINARY_CHECK(5, hadd);
  auto rhadd = [](auto x, auto y) {
    return (static_cast<int>(x) + static_cast<int>(y) + 1) >> 1;
  };
  BINARY_CHECK(6, rhadd);

  assert(C[7] == (T)(FVec[0]));
  assert(C[8] == (T)(FVec[1]));

  return 0;
#undef UNARY_CHECK
#undef BINARY_CHECK
}

template <typename T> int do_test(const T *A, const T *B, T *C) {
  queue Q;
  // Avoid out-of-range float->(u)char errors by keeping these values within
  // range.
  vec<float, 2> FVec = {1.0f, 127.0f};
  {
    buffer<T> ABuf(A, BufferSize);
    buffer<T> BBuf(B, BufferSize);
    buffer<T> CBuf(C, BufferSize);
    Q.submit([&](handler &CGH) {
      auto A = ABuf.template get_access<access::mode::read>(CGH);
      auto B = BBuf.template get_access<access::mode::read>(CGH);
      auto C = CBuf.template get_access<access::mode::write>(CGH);
      CGH.single_task<>([=]() {
        C[0] = clz(A[0]);
        C[1] = ctz(A[1]);
        C[2] = abs(A[2]);
        C[3] = min(A[3], B[3]);
        C[4] = max(A[4], B[4]);
        C[5] = hadd(A[5], B[5]);
        C[6] = rhadd(A[6], B[6]);

        vec<T, 2> conv = FVec.template convert<T>();
        C[7] = conv[0];
        C[8] = conv[1];
      });
    });
  }

  // Regression test async work-group copy builtins
  {
    buffer<T> ABuf(A, BufferSize);
    Q.submit([&](handler &CGH) {
      auto A = ABuf.template get_access<access::mode::read_write>(CGH);
      local_accessor<T, 1> Local(range<1>{WorkGroupSize}, CGH);

      nd_range<1> NDR{range<1>(NElems), range<1>(WorkGroupSize)};
      CGH.parallel_for<>(NDR, [=](nd_item<1> NDId) {
        auto GrId = NDId.get_group_linear_id();
        size_t NElemsToCopy = WorkGroupSize;
        size_t Offset = GrId * WorkGroupSize;
        auto E = NDId.async_work_group_copy(
            Local.template get_multi_ptr<access::decorated::legacy>(),
            A.template get_multi_ptr<access::decorated::legacy>() + Offset,
            NElemsToCopy);
        E.wait();
      });
    });
  }

  if (!std::is_same_v<T, char>) {
    return check<T>(A, B, C, FVec);
  }

  // Cast 'char' to signed or unsigned char to check the device's char
  // signedness matches the host's.
  if constexpr (std::numeric_limits<char>::is_signed) {
    return check(reinterpret_cast<const signed char *>(A),
                 reinterpret_cast<const signed char *>(B),
                 reinterpret_cast<signed char *>(C), FVec);
  }
  return check(reinterpret_cast<const unsigned char *>(A),
               reinterpret_cast<const unsigned char *>(B),
               reinterpret_cast<unsigned char *>(C), FVec);
}

int main() {
  std::array<unsigned char, BufferSize> A, B, C;

  std::fill(A.begin(), A.end(), 1);
  std::fill(B.begin(), B.end(), 128);
  std::fill(C.begin(), C.end(), std::numeric_limits<unsigned char>::max());

  int ret = do_test(A.data(), B.data(), C.data());

  ret |= do_test(reinterpret_cast<const signed char *>(A.data()),
                 reinterpret_cast<const signed char *>(B.data()),
                 reinterpret_cast<signed char *>(C.data()));

  ret |= do_test(reinterpret_cast<const char *>(A.data()),
                 reinterpret_cast<const char *>(B.data()),
                 reinterpret_cast<char *>(C.data()));
  return ret;
}
