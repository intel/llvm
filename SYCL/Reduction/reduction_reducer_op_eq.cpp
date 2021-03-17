// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsycl-unnamed-lambda %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// TODO: this is a temporary solution until the odd performance effect
// on opencl:cpu is analyzed/fixed. Running 2x more test cases with USM
// reductions may cause 10x longer execution time right now.
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsycl-unnamed-lambda -DTEST_SYCL2020_REDUCTIONS %s -o %t2020.out
// RUN: %CPU_RUN_PLACEHOLDER %t2020.out
// RUN: %GPU_RUN_PLACEHOLDER %t2020.out
// RUN: %ACC_RUN_PLACEHOLDER %t2020.out

// This test checks that operators ++, +=, *=, |=, &=, ^= are supported
// whent the corresponding std::plus<>, std::multiplies, etc are defined.

#include <CL/sycl.hpp>
#include <cmath>
#include <iostream>

using namespace sycl;

struct XY {
  constexpr XY() : X(0), Y(0) {}
  constexpr XY(int64_t X, int64_t Y) : X(X), Y(Y) {}
  int64_t X;
  int64_t Y;
  int64_t x() const { return X; };
  int64_t y() const { return Y; };
};

enum OperationEqual {
  PlusPlus,
  PlusPlusInt,
  PlusEq,
  MultipliesEq,
  BitwiseOREq,
  BitwiseXOREq,
  BitwiseANDEq
};

namespace std {
template <> struct plus<XY> {
  using result_type = XY;
  using first_argument_type = XY;
  using second_argument_type = XY;
  constexpr XY operator()(const XY &lhs, const XY &rhs) const {
    return XY(lhs.X + rhs.X, lhs.Y + rhs.Y);
  }
};

template <> struct multiplies<XY> {
  using result_type = XY;
  using first_argument_type = XY;
  using second_argument_type = XY;
  constexpr XY operator()(const XY &lhs, const XY &rhs) const {
    return XY(lhs.X * rhs.X, lhs.Y * rhs.Y);
  }
};

template <> struct bit_or<XY> {
  using result_type = XY;
  using first_argument_type = XY;
  using second_argument_type = XY;
  constexpr XY operator()(const XY &lhs, const XY &rhs) const {
    return XY(lhs.X | rhs.X, lhs.Y | rhs.Y);
  }
};

template <> struct bit_xor<XY> {
  using result_type = XY;
  using first_argument_type = XY;
  using second_argument_type = XY;
  constexpr XY operator()(const XY &lhs, const XY &rhs) const {
    return XY(lhs.X ^ rhs.X, lhs.Y ^ rhs.Y);
  }
};

template <> struct bit_and<XY> {
  using result_type = XY;
  using first_argument_type = XY;
  using second_argument_type = XY;
  constexpr XY operator()(const XY &lhs, const XY &rhs) const {
    return XY(lhs.X & rhs.X, lhs.Y & rhs.Y);
  }
};
} // namespace std

template <bool IsSYCL2020Mode, typename T, typename BinaryOperation>
auto createReduction(T *USMPtr, T Identity, BinaryOperation BOp) {
  if constexpr (IsSYCL2020Mode)
    return sycl::reduction(USMPtr, Identity, BOp);
  else
    return ONEAPI::reduction(USMPtr, Identity, BOp);
}

template <typename T, bool IsSYCL2020Mode, typename BinaryOperation,
          OperationEqual OpEq, bool IsFP>
int test(T Identity) {
  constexpr size_t N = 16;
  constexpr size_t L = 4;

  queue Q;
  T *Data = malloc_host<T>(N, Q);
  T *Res = malloc_host<T>(1, Q);
  T Expected = Identity;
  BinaryOperation BOp;
  if constexpr (OpEq == PlusPlus || OpEq == PlusPlusInt) {
    Expected = T{N, N};
  } else {
    for (int I = 0; I < N; I++) {
      Data[I] = T{I, I + 1};
      Expected = BOp(Expected, T{I, I + 1});
    }
  }

  *Res = Identity;
  auto Red = createReduction<IsSYCL2020Mode>(Res, Identity, BOp);
  nd_range<1> NDR{N, L};
  if constexpr (OpEq == PlusPlus) {
    auto Lambda = [=](nd_item<1> ID, auto &Sum) { ++Sum; };
    Q.submit([&](handler &H) { H.parallel_for(NDR, Red, Lambda); }).wait();
  } else if constexpr (OpEq == PlusPlusInt) {
    auto Lambda = [=](nd_item<1> ID, auto &Sum) { Sum++; };
    Q.submit([&](handler &H) { H.parallel_for(NDR, Red, Lambda); }).wait();
  } else if constexpr (OpEq == PlusEq) {
    auto Lambda = [=](nd_item<1> ID, auto &Sum) {
      Sum += Data[ID.get_global_id(0)];
    };
    Q.submit([&](handler &H) { H.parallel_for(NDR, Red, Lambda); }).wait();
  } else if constexpr (OpEq == MultipliesEq) {
    auto Lambda = [=](nd_item<1> ID, auto &Sum) {
      Sum *= Data[ID.get_global_id(0)];
    };
    Q.submit([&](handler &H) { H.parallel_for(NDR, Red, Lambda); }).wait();
  } else if constexpr (OpEq == BitwiseOREq) {
    auto Lambda = [=](nd_item<1> ID, auto &Sum) {
      Sum |= Data[ID.get_global_id(0)];
    };
    Q.submit([&](handler &H) { H.parallel_for(NDR, Red, Lambda); }).wait();
  } else if constexpr (OpEq == BitwiseXOREq) {
    auto Lambda = [=](nd_item<1> ID, auto &Sum) {
      Sum ^= Data[ID.get_global_id(0)];
    };
    Q.submit([&](handler &H) { H.parallel_for(NDR, Red, Lambda); }).wait();
  } else if constexpr (OpEq == BitwiseANDEq) {
    auto Lambda = [=](nd_item<1> ID, auto &Sum) {
      Sum &= Data[ID.get_global_id(0)];
    };
    Q.submit([&](handler &H) { H.parallel_for(NDR, Red, Lambda); }).wait();
  }

  int Error = 0;
  if constexpr (IsFP) {
    T Diff = (Expected / *Res) - T{1};
    Error = (std::abs(Diff.x()) > 0.5 || std::abs(Diff.y()) > 0.5) ? 1 : 0;
  } else {
    Error = (Expected.x() != Res->x() || Expected.y() != Res->y()) ? 1 : 0;
  }
  if (Error)
    std::cerr << "Error: expected = (" << Expected.x() << ", " << Expected.y()
              << "); computed = (" << Res->x() << ", " << Res->y() << ")\n";

  free(Res, Q);
  free(Data, Q);
  return Error;
}

template <typename T, typename BinaryOperation, OperationEqual OpEq,
          bool IsFP = false>
int testBoth(T Identity) {
  int Error = 0;
#ifdef TEST_SYCL2020_REDUCTIONS
  Error += test<T, true, BinaryOperation, OpEq, IsFP>(Identity);
#else
  Error += test<T, false, BinaryOperation, OpEq, IsFP>(Identity);
#endif
  return Error;
}

template <typename T> int testFPPack() {
  int Error = 0;
  Error += testBoth<T, std::plus<T>, PlusEq, true>(T{});
  Error += testBoth<T, std::multiplies<T>, MultipliesEq, true>(T{1, 1});
  return Error;
}

template <typename T, bool TestPlusPlus> int testINTPack() {
  int Error = 0;
  if constexpr (TestPlusPlus) {
    Error += testBoth<T, std::plus<T>, PlusPlus>(T{});
    Error += testBoth<T, std::plus<>, PlusPlusInt>(T{});
  }
  Error += testBoth<T, std::plus<T>, PlusEq>(T{});
  Error += testBoth<T, std::multiplies<T>, MultipliesEq>(T{1, 1});
  Error += testBoth<T, std::bit_or<T>, BitwiseOREq>(T{});
  Error += testBoth<T, std::bit_xor<T>, BitwiseXOREq>(T{});
  Error += testBoth<T, std::bit_and<T>, BitwiseANDEq>(T{~0, ~0});
  return Error;
}

int main() {
  int Error = 0;
  Error += testFPPack<float2>();
  Error += testINTPack<int2, true>();
  Error += testINTPack<XY, false>();

  std::cout << (Error ? "Failed\n" : "Passed.\n");
  return Error;
}
