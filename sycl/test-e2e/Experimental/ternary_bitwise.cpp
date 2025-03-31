// REQUIRES: aspect-usm_shared_allocations

// XFAIL: opencl && cpu
// XFAIL-TRACKER: TODO

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Checks the results of the ternary bitwise function extension.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/ternary_bitwise.hpp>
#include <sycl/usm.hpp>

#include <random>

namespace syclex = sycl::ext::oneapi::experimental;

constexpr size_t NumOps = 256;

template <typename T, size_t... Is>
std::array<T, NumOps> apply(T *A, T *B, T *C, std::index_sequence<Is...>) {
  return std::array<T, NumOps>{
      syclex::ternary_bitwise<Is>(A[Is], B[Is], C[Is])...};
}

template <typename T> void fillRandom(T *Dest) {
  std::random_device RDev;
  std::mt19937 RNG(RDev());
  std::uniform_int_distribution<T> Dist;
  for (size_t I = 0; I < NumOps; ++I)
    Dest[I] = Dist(RNG);
}

template <typename T, int N> void fillRandom(sycl::vec<T, N> *Dest) {
  std::random_device RDev;
  std::mt19937 RNG(RDev());
  std::uniform_int_distribution<T> Dist;
  for (size_t I = 0; I < NumOps; ++I)
    for (size_t J = 0; J < N; ++J)
      Dest[I][J] = Dist(RNG);
}

template <typename T, size_t N> void fillRandom(sycl::marray<T, N> *Dest) {
  std::random_device RDev;
  std::mt19937 RNG(RDev());
  std::uniform_int_distribution<T> Dist;
  for (size_t I = 0; I < NumOps; ++I)
    for (size_t J = 0; J < N; ++J)
      Dest[I][J] = Dist(RNG);
}

bool allTrue(bool B) { return B; }

template <typename T, int N> bool allTrue(sycl::vec<T, N> B) {
  for (size_t I = 0; I < N; ++I)
    if (!static_cast<bool>(B[I]))
      return false;
  return true;
}

template <size_t N> bool allTrue(sycl::marray<bool, N> B) {
  return std::all_of(B.begin(), B.end(), [](bool b) { return b; });
}

template <typename T> std::string toString(T X) { return std::to_string(X); }

template <typename T, int N> std::string toString(sycl::vec<T, N> X) {
  std::string Result = "{" + toString(X[0]);
  for (size_t I = 1; I < N; ++I)
    Result += "," + toString(X[I]);
  return Result + "}";
}

template <typename T, size_t N> std::string toString(sycl::marray<T, N> X) {
  std::string Result = "{" + toString(X[0]);
  for (size_t I = 1; I < N; ++I)
    Result += "," + toString(X[I]);
  return Result + "}";
}

template <typename T> int Check(sycl::queue &Q, std::string_view TName) {
  constexpr auto IdxSeq = std::make_index_sequence<NumOps>{};

  int Failed = 0;

  T *A = sycl::malloc_shared<T>(NumOps, Q);
  T *B = sycl::malloc_shared<T>(NumOps, Q);
  T *C = sycl::malloc_shared<T>(NumOps, Q);
  auto *Out = sycl::malloc_shared<std::array<T, NumOps>>(1, Q);

  fillRandom(A);
  fillRandom(B);
  fillRandom(C);

  Q.single_task([=]() { *Out = apply(A, B, C, IdxSeq); }).wait_and_throw();

  std::array<T, NumOps> DevResults = *Out;
  std::array<T, NumOps> HostResults = apply(A, B, C, IdxSeq);

  for (size_t I = 0; I < NumOps; ++I) {
    if (allTrue(DevResults[I] != HostResults[I])) {
      std::cout << "Failed check for type " << TName << " at index " << I
                << ": " << toString(DevResults[I])
                << " != " << toString(HostResults[I]) << std::endl;
      ++Failed;
    }
  }

  sycl::free(A, Q);
  sycl::free(B, Q);
  sycl::free(C, Q);
  sycl::free(Out, Q);

  return Failed;
}

int main() {
  sycl::queue Q;

  int Failed = 0;
#define CHECK(...) Failed += Check<__VA_ARGS__>(Q, #__VA_ARGS__);
  CHECK(char)
  CHECK(signed char)
  CHECK(unsigned char)
  CHECK(short)
  CHECK(unsigned short)
  CHECK(int)
  CHECK(unsigned int)
  CHECK(long)
  CHECK(unsigned long)
  CHECK(sycl::vec<int8_t, 2>)
  CHECK(sycl::vec<int8_t, 8>)
  CHECK(sycl::vec<uint8_t, 2>)
  CHECK(sycl::vec<uint8_t, 8>)
  CHECK(sycl::vec<int16_t, 2>)
  CHECK(sycl::vec<int16_t, 8>)
  CHECK(sycl::vec<uint16_t, 2>)
  CHECK(sycl::vec<uint16_t, 8>)
  CHECK(sycl::vec<int32_t, 2>)
  CHECK(sycl::vec<int32_t, 8>)
  CHECK(sycl::vec<uint32_t, 2>)
  CHECK(sycl::vec<uint32_t, 8>)
  CHECK(sycl::vec<int64_t, 2>)
  CHECK(sycl::vec<int64_t, 8>)
  CHECK(sycl::vec<uint64_t, 2>)
  CHECK(sycl::vec<uint64_t, 8>)
  CHECK(sycl::marray<char, 3>)
  CHECK(sycl::marray<signed char, 3>)
  CHECK(sycl::marray<unsigned char, 3>)
  CHECK(sycl::marray<short, 3>)
  CHECK(sycl::marray<unsigned short, 3>)
  CHECK(sycl::marray<int, 3>)
  CHECK(sycl::marray<unsigned int, 3>)
  CHECK(sycl::marray<long, 3>)
  CHECK(sycl::marray<unsigned long, 3>)
  return Failed;
}
