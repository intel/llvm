//==-------------------- ballot.cpp  - DPC++ ESIMD ballot test -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test checks esimd::ballot function.

#include "../esimd_test_utils.hpp"

#include <array>
#include <random>
#include <sycl/builtins_esimd.hpp>

using namespace sycl;
using namespace sycl::ext::intel;
using namespace sycl::ext::intel::esimd;

template <class T, int N> bool test(queue &Q) {
  std::cout << "  Running " << typeid(T).name() << " test, N=" << N << "...\n";

  std::array<T, N> Pred;
  for (int I = 0; I < N; ++I)
    Pred[I] = (I & 1) ? I : 0;

  std::random_device RD;
  std::mt19937 Gen(RD());
  std::shuffle(Pred.begin(), Pred.end(), Gen);

  uint Res = 0;
  try {
    buffer<T, 1> PB(Pred.data(), range<1>(N));
    buffer<decltype(Res), 1> RB(&Res, range<1>(1));

    auto E = Q.submit([&](handler &CGH) {
      auto In = PB.template get_access<access::mode::read>(CGH);
      auto Out = RB.template get_access<access::mode::write>(CGH);

      CGH.parallel_for(sycl::range<1>{1}, [=](id<1>) SYCL_ESIMD_KERNEL {
        simd<T, N> Mask;
        Mask.copy_from(In, 0);

        uint Res = esimd::ballot(Mask);
        scalar_store(Out, 0, Res);
      });
    });
    E.wait();
  } catch (sycl::exception const &E) {
    std::cout << "ERROR. SYCL exception caught: " << E.what() << std::endl;
    return false;
  }

  uint Ref = 0;
  for (int I = 0; I < N; ++I)
    if (Pred[I] != 0)
      Ref |= 0x1 << I;

  std::cout << (Res == Ref ? "    Passed\n" : "    FAILED\n");
  return Res == Ref;
}

template <class T> bool test(queue &Q) {
  bool Pass = true;

  Pass &= test<T, 4>(Q);
  Pass &= test<T, 8>(Q);
  Pass &= test<T, 12>(Q);
  Pass &= test<T, 16>(Q);
  Pass &= test<T, 20>(Q);
  Pass &= test<T, 24>(Q);
  Pass &= test<T, 28>(Q);
  Pass &= test<T, 32>(Q);

  return Pass;
}

int main(void) {
  queue Q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());
  auto Dev = Q.get_device();
  std::cout << "Running on " << Dev.get_info<sycl::info::device::name>()
            << "\n";

  bool Pass = true;
  Pass &= test<ushort>(Q);
  Pass &= test<uint>(Q);

  std::cout << (Pass ? "Test Passed\n" : "Test FAILED\n");
  return Pass ? 0 : 1;
}
