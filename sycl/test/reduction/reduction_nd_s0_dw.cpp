// RUN: %clangxx -fsycl %s -o %t.out
// RUNx: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//==----------------reduction_ctor.cpp - SYCL reduction basic test ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test performs basic checks of parallel_for(nd_range, reduction, func)
// with reductions initialized with 0-dimensional discard_write accessor.

#include <CL/sycl.hpp>
#include <cassert>

using namespace cl::sycl;

template <typename T, class BinaryOperation>
void initInputData(buffer<T, 1> &InBuf, T &ExpectedOut, T Identity,
                   BinaryOperation BOp, size_t N) {
  ExpectedOut = Identity;
  auto In = InBuf.template get_access<access::mode::write>();
  for (int I = 0; I < N; ++I) {
    if (std::is_same<BinaryOperation, std::multiplies<T>>::value)
      In[I] = 1 + (((I % 37) == 0) ? 1 : 0);
    else
      In[I] = ((I + 1) % 5) + 1.1;
    ExpectedOut = BOp(ExpectedOut, In[I]);
  }
};

template <typename T, int Dim, class BinaryOperation>
class Known;
template <typename T, int Dim, class BinaryOperation>
class Unknown;

template <typename T>
struct Vec {
  Vec() : X(0), Y(0) {}
  Vec(T X, T Y) : X(X), Y(Y) {}
  Vec(T V) : X(V), Y(V) {}
  bool operator==(const Vec &P) const {
    return P.X == X && P.Y == Y;
  }
  bool operator!=(const Vec &P) const {
    return !(*this == P);
  }
  T X;
  T Y;
};
template <typename T>
bool operator==(const Vec<T> &A, const Vec<T> &B) {
  return A.X == B.X && A.Y == B.Y;
}
template <typename T>
std::ostream &operator<<(std::ostream &OS, const Vec<T> &P) {
  return OS << "(" << P.X << ", " << P.Y << ")";
}

template <class T>
struct VecPlus {
  using P = Vec<T>;
  P operator()(const P &A, const P &B) const {
    return P(A.X + B.X, A.Y + B.Y);
  }
};

template <typename T, int Dim, class BinaryOperation>
void test(T Identity, size_t WGSize, size_t NWItems) {
  buffer<T, 1> InBuf(NWItems);
  buffer<T, 1> OutBuf(1);

  // Initialize.
  BinaryOperation BOp;
  T CorrectOut;
  initInputData(InBuf, CorrectOut, Identity, BOp, NWItems);

  // Compute.
  queue Q;
  Q.submit([&](handler &CGH) {
    auto In = InBuf.template get_access<access::mode::read>(CGH);
    accessor<T, Dim, access::mode::discard_write, access::target::global_buffer>
        Out(OutBuf, CGH);
    auto Redu = intel::reduction(Out, Identity, BOp);

    range<1> GlobalRange(NWItems);
    range<1> LocalRange(WGSize);
    nd_range<1> NDRange(GlobalRange, LocalRange);
    CGH.parallel_for<Known<T, Dim, BinaryOperation>>(
        NDRange, Redu, [=](nd_item<1> NDIt, auto &Sum) {
          Sum.combine(In[NDIt.get_global_linear_id()]);
        });
  });

  // Check correctness.
  auto Out = OutBuf.template get_access<access::mode::read>();
  T ComputedOut = *(Out.get_pointer());
  if (ComputedOut != CorrectOut) {
    std::cout << "NWItems = " << NWItems << ", WGSize = " << WGSize << "\n";
    std::cout << "Computed value: " << ComputedOut
              << ", Expected value: " << CorrectOut << "\n";
    assert(0 && "Wrong value.");
  }
}

int main() {
  // Check some less standards WG sizes and corner cases first.
  test<int, 0, intel::plus<int>>(0, 2, 2);
  test<int, 0, intel::plus<int>>(0, 7, 7);
  test<int, 0, intel::plus<int>>(0, 9, 18);
  test<int, 0, intel::plus<int>>(0, 49, 49 * 5);

  // Try some power-of-two work-group sizes.
  test<int, 0, intel::plus<int>>(0, 2, 64);
  test<int, 0, intel::plus<int>>(0, 4, 64);
  test<int, 0, intel::plus<int>>(0, 8, 128);
  test<int, 0, intel::plus<int>>(0, 16, 256);
  test<int, 0, intel::plus<int>>(0, 32, 256);
  test<int, 0, intel::plus<int>>(0, 64, 256);
  test<int, 0, intel::plus<int>>(0, 128, 256);
  test<int, 0, intel::plus<int>>(0, 256, 256);

  // Check with various operations.
  test<int, 0, std::multiplies<int>>(1, 8, 256);
  test<int, 0, intel::bit_or<int>>(0, 8, 256);
  test<int, 0, intel::bit_xor<int>>(0, 8, 256);
  test<int, 0, intel::bit_and<int>>(~0, 8, 256);
  test<int, 0, intel::minimum<int>>(std::numeric_limits<int>::max(), 8, 256);
  test<int, 0, intel::maximum<int>>(std::numeric_limits<int>::min(), 8, 256);

  // Check with various types.
  test<float, 0, std::multiplies<float>>(1, 8, 256);
  test<float, 0, intel::minimum<float>>(std::numeric_limits<float>::max(), 8, 256);
  test<float, 0, intel::maximum<float>>(std::numeric_limits<float>::min(), 8, 256);

  test<double, 0, std::multiplies<double>>(1, 8, 256);
  test<double, 0, intel::minimum<double>>(std::numeric_limits<double>::max(), 8, 256);
  test<double, 0, intel::maximum<double>>(std::numeric_limits<double>::min(), 8, 256);

  // Check with CUSTOM type.
  test<Vec<long long>, 0, VecPlus<long long>>(Vec<long long>(0), 8, 256);

  std::cout << "Test passed\n";
  return 0;
}
