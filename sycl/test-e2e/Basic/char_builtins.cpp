// RUN: %{build} -Wno-absolute-value -Wno-deprecated-declarations -o %t.out
// RUN: %{run} %t.out

#include <array>
#include <sycl/builtins.hpp>
#include <sycl/detail/core.hpp>

using namespace sycl;

constexpr size_t BufferSize = 16;
constexpr size_t NElems = 32;
constexpr size_t WorkGroupSize = 8;

template <typename T>
int check(const T *A, const T *B, T *C, const vec<float, 2> FVec) {
  assert(C[0] == clz(A[0]) && "error: clz failed");
  assert(C[1] == ctz(A[1]) && "error: ctz failed");
  assert(C[2] == abs(A[2]) && "error: abs failed");

  assert(C[3] == std::min(A[3], B[3]) && "error: min failed");
  assert(C[4] == std::max(A[4], B[4]) && "error: max failed");

  auto Hadd = [](auto x, auto y) {
    return (static_cast<int>(x) + static_cast<int>(y)) >> 1;
  };
  assert(C[5] == Hadd(A[5], B[5]) && "error: hadd failed");
  auto Rhadd = [](auto x, auto y) {
    return (static_cast<int>(x) + static_cast<int>(y) + 1) >> 1;
  };
  assert(C[6] == Rhadd(A[6], B[6]) && "error: rhadd failed");

  assert(C[7] == (T)(FVec[0]));
  assert(C[8] == (T)(FVec[1]));

  return 0;
}

template <typename T> int doCharTest(const T *A, const T *B, T *C) {
  queue Q;
  // Avoid out-of-range float->(u)char errors by keeping these values within
  // range.
  vec<float, 2> FVec = {1.0f, 127.0f};
  {
    buffer<T> ABuf(A, BufferSize);
    buffer<T> BBuf(B, BufferSize);
    buffer<T> CBuf(C, BufferSize);
    Q.submit([&](handler &CGH) {
      auto AAcc = ABuf.template get_access<access::mode::read>(CGH);
      auto BAcc = BBuf.template get_access<access::mode::read>(CGH);
      auto CAcc = CBuf.template get_access<access::mode::write>(CGH);
      CGH.single_task<>([=]() {
        CAcc[0] = clz(AAcc[0]);
        CAcc[1] = ctz(AAcc[1]);
        CAcc[2] = abs(AAcc[2]);
        CAcc[3] = min(AAcc[3], BAcc[3]);
        CAcc[4] = max(AAcc[4], BAcc[4]);
        CAcc[5] = hadd(AAcc[5], BAcc[5]);
        CAcc[6] = rhadd(AAcc[6], BAcc[6]);

        vec<T, 2> Conv = FVec.template convert<T>();
        CAcc[7] = Conv[0];
        CAcc[8] = Conv[1];
      });
    });
  }

  // Regression test async work-group copy builtins.
  {
    buffer<T> ABuf(A, BufferSize);
    Q.submit([&](handler &CGH) {
      auto AAcc = ABuf.template get_access<access::mode::read_write>(CGH);
      local_accessor<T, 1> Local(range<1>{WorkGroupSize}, CGH);

      nd_range<1> NDR{range<1>(NElems), range<1>(WorkGroupSize)};
      CGH.parallel_for<>(NDR, [=](nd_item<1> NDId) {
        auto GrId = NDId.get_group_linear_id();
        size_t NElemsToCopy = WorkGroupSize;
        size_t Offset = GrId * WorkGroupSize;
        auto E = NDId.async_work_group_copy(
            Local.template get_multi_ptr<access::decorated::legacy>(),
            AAcc.template get_multi_ptr<access::decorated::legacy>() + Offset,
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

  int Ret = doCharTest(A.data(), B.data(), C.data());

  Ret |= doCharTest(reinterpret_cast<const signed char *>(A.data()),
                    reinterpret_cast<const signed char *>(B.data()),
                    reinterpret_cast<signed char *>(C.data()));

  Ret |= doCharTest(reinterpret_cast<const char *>(A.data()),
                    reinterpret_cast<const char *>(B.data()),
                    reinterpret_cast<char *>(C.data()));
  return Ret;
}
