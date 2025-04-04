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
constexpr auto IdxSeq = std::make_index_sequence<NumOps>{};

static std::random_device RDev;
static std::mt19937 RNG(RDev());

template <typename T, size_t... Is>
std::array<T, NumOps> apply(std::array<T, NumOps> &A, std::array<T, NumOps> &B,
                            std::array<T, NumOps> &C,
                            std::index_sequence<Is...>) {
  return std::array<T, NumOps>{
      syclex::ternary_bitwise<Is>(A[Is], B[Is], C[Is])...};
}

template <typename T> void fillRandom(T *Dest) {
  std::uniform_int_distribution<T> Dist;
  for (size_t I = 0; I < NumOps; ++I)
    Dest[I] = Dist(RNG);
}

template <typename T, int N> void fillRandom(sycl::vec<T, N> *Dest) {
  std::uniform_int_distribution<T> Dist;
  for (size_t I = 0; I < NumOps; ++I)
    for (size_t J = 0; J < N; ++J)
      Dest[I][J] = Dist(RNG);
}

template <typename T, size_t N> void fillRandom(sycl::marray<T, N> *Dest) {
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

template <typename T> struct MemObj {
  std::array<T, NumOps> A;
  std::array<T, NumOps> B;
  std::array<T, NumOps> C;
  std::array<T, NumOps> Out;
};

template <typename T> MemObj<T> *createMem(sycl::queue &Q) {
  MemObj<T> *Obj = sycl::malloc_shared<MemObj<T>>(NumOps, Q);
  fillRandom(Obj->A.data());
  fillRandom(Obj->B.data());
  fillRandom(Obj->C.data());
  return Obj;
}

template <typename T> int checkResult(MemObj<T> &Mem, std::string_view TName) {
  std::array<T, NumOps> &DevResults = Mem.Out;
  std::array<T, NumOps> HostResults = apply(Mem.A, Mem.B, Mem.C, IdxSeq);

  int Failed = 0;
  for (size_t I = 0; I < NumOps; ++I) {
    if (allTrue(DevResults[I] != HostResults[I])) {
      std::cout << "Failed check for type " << TName << " at index " << I
                << ": " << toString(DevResults[I])
                << " != " << toString(HostResults[I]) << std::endl;
      ++Failed;
    }
  }
  return Failed;
}

int main() {
  sycl::queue Q;

  auto *CharObj = createMem<char>(Q);
  auto *SCharObj = createMem<signed char>(Q);
  auto *UCharObj = createMem<unsigned char>(Q);
  auto *ShortObj = createMem<short>(Q);
  auto *UShortObj = createMem<unsigned short>(Q);
  auto *IntObj = createMem<int>(Q);
  auto *UIntObj = createMem<unsigned int>(Q);
  auto *LongObj = createMem<long>(Q);
  auto *ULongObj = createMem<unsigned long>(Q);
  auto *SChar2Obj = createMem<sycl::vec<int8_t, 2>>(Q);
  auto *UShort8Obj = createMem<sycl::vec<uint16_t, 8>>(Q);
  auto *Int2Obj = createMem<sycl::vec<int32_t, 2>>(Q);
  auto *ULong8Obj = createMem<sycl::vec<uint64_t, 8>>(Q);
  auto *CharMarrayObj = createMem<sycl::marray<char, 3>>(Q);
  auto *UShortMarrayObj = createMem<sycl::marray<unsigned short, 3>>(Q);
  auto *IntMarrayObj = createMem<sycl::marray<int, 3>>(Q);
  auto *ULongMarrayObj = createMem<sycl::marray<unsigned long, 3>>(Q);

  Q.parallel_for(17, [=](sycl::id<1> Idx) {
     // We let the ID determine which memory object the work-item processes.
     size_t WorkCounter = 0;
#define APPLY(MEM_OBJ)                                                         \
  if ((WorkCounter++) == Idx[0])                                               \
    MEM_OBJ->Out = apply(MEM_OBJ->A, MEM_OBJ->B, MEM_OBJ->C, IdxSeq);
     APPLY(CharObj)
     APPLY(SCharObj)
     APPLY(UCharObj)
     APPLY(ShortObj)
     APPLY(UShortObj)
     APPLY(IntObj)
     APPLY(UIntObj)
     APPLY(LongObj)
     APPLY(ULongObj)
     APPLY(SChar2Obj)
     APPLY(UShort8Obj)
     APPLY(Int2Obj)
     APPLY(ULong8Obj)
     APPLY(CharMarrayObj)
     APPLY(UShortMarrayObj)
     APPLY(IntMarrayObj)
     APPLY(ULongMarrayObj)
   }).wait_and_throw();

  int Failed = 0;

  Failed += checkResult(*CharObj, "char");
  Failed += checkResult(*SCharObj, "signed char");
  Failed += checkResult(*UCharObj, "unsigned char");
  Failed += checkResult(*ShortObj, "short");
  Failed += checkResult(*UShortObj, "unsigned short");
  Failed += checkResult(*IntObj, "int");
  Failed += checkResult(*UIntObj, "unsigned int");
  Failed += checkResult(*LongObj, "long");
  Failed += checkResult(*ULongObj, "unsigned long");
  Failed += checkResult(*SChar2Obj, "sycl::vec<int8_t, 2>");
  Failed += checkResult(*UShort8Obj, "sycl::vec<uint16_t, 8>");
  Failed += checkResult(*Int2Obj, "sycl::vec<int32_t, 2>");
  Failed += checkResult(*ULong8Obj, "sycl::vec<uint64_t, 8>");
  Failed += checkResult(*CharMarrayObj, "sycl::marray<char, 3>");
  Failed += checkResult(*UShortMarrayObj, "sycl::marray<unsigned short, 3>");
  Failed += checkResult(*IntMarrayObj, "sycl::marray<int, 3>");
  Failed += checkResult(*ULongMarrayObj, "sycl::marray<unsigned long, 3>");

  sycl::free(CharObj, Q);
  sycl::free(SCharObj, Q);
  sycl::free(UCharObj, Q);
  sycl::free(ShortObj, Q);
  sycl::free(UShortObj, Q);
  sycl::free(IntObj, Q);
  sycl::free(UIntObj, Q);
  sycl::free(LongObj, Q);
  sycl::free(ULongObj, Q);
  sycl::free(SChar2Obj, Q);
  sycl::free(UShort8Obj, Q);
  sycl::free(Int2Obj, Q);
  sycl::free(ULong8Obj, Q);
  sycl::free(CharMarrayObj, Q);
  sycl::free(UShortMarrayObj, Q);
  sycl::free(IntMarrayObj, Q);
  sycl::free(ULongMarrayObj, Q);

  return Failed;
}
