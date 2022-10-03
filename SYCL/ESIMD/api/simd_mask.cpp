//==---------------- simd_mask.cpp  - DPC++ ESIMD simd_mask API test -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// UNSUPPORTED: cuda || hip
// RUN: %clangxx -fsycl-unnamed-lambda -fsycl -I%S/.. %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Smoke test for simd_mask API functionality.

#include "esimd_test_utils.hpp"

#include <iostream>
#include <limits>
#include <memory>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>
#include <utility>

using namespace sycl::ext::intel::esimd;
using namespace sycl;

template <int N> using value_type = typename simd_mask<N>::element_type;

template <int N> static inline constexpr value_type<N> Error = 0;
template <int N> static inline constexpr value_type<N> Pass = 1;

// Slow mask storage function independent of simd_mask::copy_to (memory) and
// simd_mask::value_type.
template <int N>
static SYCL_ESIMD_FUNCTION void store(value_type<N> *Ptr, simd_mask<N> M) {
  value_type<N> Arr[N];
  M.copy_to(Arr);

  for (auto I = 0; I < N; ++I) {
    Ptr[I] = Arr[I] ? 1 : 0;
  }
}

// Slow mask storage function independent of simd_mask::copy_from (memory) and
// simd_mask::value_type.
template <int N>
static SYCL_ESIMD_FUNCTION simd_mask<N> load(value_type<N> *Ptr) {
  value_type<N> Arr[N];
  for (auto I = 0; I < N; ++I) {
    Arr[I] = Ptr[I] ? 1 : 0;
  }
  simd_mask<N> M(std::move(Arr));
  return M;
}

// Apply F to each element of M and write result to Res.
template <typename PerElemF, int N>
static SYCL_ESIMD_FUNCTION void
check_mask(const simd_mask<N> &M, typename simd_mask<N>::element_type *Res,
           PerElemF F) {
  for (auto I = 0; I < N; ++I) {
    value_type<N> Val = F(M[I]) ? Pass<N> : Error<N>;
    Res[I] = Val;
  }
}

// Slow check if M1 and M2 are equal and write result to Res.
template <int N>
static SYCL_ESIMD_FUNCTION void
check_masks_equal(const simd_mask<N> &M1, const simd_mask<N> &M2,
                  typename simd_mask<N>::element_type *Res) {
  for (auto I = 0; I < N; ++I) {
    value_type<N> Val = ((M1[I] == 0) == (M2[I] == 0)) ? Pass<N> : Error<N>;
    Res[I] = Val;
  }
}

// Represents a generic test case. Each test case has two optional inputs -
// In and InvIn, and one mandatory output - Res. Each input and output element
// matches the simd_mask value type, there is one data element in each per
// NDRange element. InvIn is a logical inversion of In for easier validation of
// operations.
template <int N> struct sub_test {
  using value_type = typename simd_mask<N>::element_type;

  // Used to automatically free USM memory allocated for input/output.
  struct usm_deleter {
    queue Q;

    void operator()(value_type *Ptr) {
      if (Ptr) {
        sycl::free(Ptr, Q);
      }
    }
  };

  queue Q;
  using ptr_type = std::unique_ptr<value_type, usm_deleter>;
  ptr_type In;
  ptr_type InvIn;
  ptr_type Res;
  size_t Size = N * 7;

  sub_test(queue Q, bool Need2Inputs = false) : Q(Q) {
    In = ptr_type{nullptr, usm_deleter{Q}};
    InvIn = ptr_type{nullptr, usm_deleter{Q}};
    Res = ptr_type{nullptr, usm_deleter{Q}};
    init(Need2Inputs);
  }

  void init(bool Need2Inputs) {
    device Dev = Q.get_device();
    context Ctx = Q.get_context();
    const auto Sz = Size * sizeof(value_type);
    In.reset(static_cast<value_type *>(malloc_shared(Sz, Dev, Ctx)));
    if (Need2Inputs)
      InvIn.reset(static_cast<value_type *>(malloc_shared(Sz, Dev, Ctx)));
    Res.reset(static_cast<value_type *>(malloc_shared(Sz, Dev, Ctx)));
    if (!In || (Need2Inputs && !InvIn) || !Res) {
      throw sycl::exception(std::error_code{}, "malloc_shared failed");
    }
    for (unsigned I = 0; I < Size; I += N) {
      unsigned J = 0;

      for (; J < N / 2; ++J) {
        auto Ind = I + J;
        In.get()[Ind] = 1;
        if (Need2Inputs)
          InvIn.get()[Ind] = 0;
        Res.get()[Ind] = Error<N>;
      }
      for (; J < N; ++J) {
        auto Ind = I + J;
        In.get()[Ind] = 0;
        if (Need2Inputs)
          InvIn.get()[Ind] = 1;
        Res.get()[Ind] = Error<N>;
      }
    }
  }

  // The main test function which submits the test kernel F.
  template <typename FuncType> bool run(const char *Name, FuncType F) {
    std::cout << "  Running " << Name << " API test, N=" << N << "...\n";

    // Submit the kernel.
    try {
      sycl::range<1> R{Size / N};
      auto E = Q.submit([&](handler &CGH) { CGH.parallel_for(R, F); });
      E.wait();
    } catch (sycl::exception &Exc) {
      std::cout << "    *** ERROR. SYCL exception caught: << " << Exc.what()
                << "\n";
      return false;
    }
    // Verify results - basically see if there are no non-zeros in the 'Res'
    // array.
    int ErrCnt = 0;

    for (auto I = 0; I < Size; ++I) {
      if (Res.get()[I] == Error<N>) {
        if (++ErrCnt < 10) {
          std::cout << "    failed at index " << I << "\n";
        }
      }
    }
    if (ErrCnt > 0) {
      std::cout << "    pass rate: "
                << ((float)(Size - ErrCnt) / (float)Size) * 100.0f << "% ("
                << (Size - ErrCnt) << "/" << Size << ")\n";
    }
    std::cout << (ErrCnt > 0 ? "    FAILED\n" : "    Passed\n");
    return ErrCnt == 0;
  }
};

// Defines actual test cases.
template <int N> struct simd_mask_api_test {
  using value_type = typename simd_mask<N>::element_type;

  bool run(queue Q) {
    bool Passed = true;

    // Tests for constructors and operators ! [].
    {
      sub_test<N> Test(Q);
      value_type *In = Test.In.get();
      value_type *Res = Test.Res.get();
      Passed &= Test.run(
          "broadcast constructor, operator[]", [=](id<1> Id) SYCL_ESIMD_KERNEL {
            auto Off = Id * N;
            simd_mask<N> M0 = load<N>(In + Off); // 1..1,0...0
            simd_mask<N> M1(M0[0]);
            check_mask(M1, Res + Off, [](value_type V) { return V != 0; });
          });
    }
    {
      sub_test<N> Test(Q);
      value_type *Res = Test.Res.get();
      Passed &=
          Test.run("value initialization", [=](id<1> Id) SYCL_ESIMD_KERNEL {
            auto Off = Id * N;
            simd_mask<N> M0{};
        // TODO FIXME Shorter version not work due to a BE bug
#define WORKAROUND_BE_BUG
#ifdef WORKAROUND_BE_BUG
            for (auto I = 0; I < N; ++I) {
              if (M0[I] == 0) {
                Res[Off + I] = Pass<N>;
              }
              // else write Error<N>, but its already there
            }
#else
        check_mask(M0, Res + Off, [](value_type V) { return (V == 0); });
#endif // WORKAROUND_BE_BUG
#undef WORKAROUND_BE_BUG
          });
    }
    {
      sub_test<N> Test(Q, true /*need InInv*/);
      value_type *In = Test.In.get();
      value_type *InInv = Test.InvIn.get();
      value_type *Res = Test.Res.get();
      Passed &= Test.run("operator!", [=](id<1> Id) SYCL_ESIMD_KERNEL {
        auto Off = Id * N;
        simd_mask<N> M0 = load<N>(In + Off); // 1..1,0...0
        simd_mask<N> M1 = !M0;
        simd_mask<N> M2 = load<N>(InInv + Off); // 0..0,1...1
        check_masks_equal(M1, M2, Res + Off);
      });
    }

    // Tests for binary and assignment operators.

#define RUN_TEST(Op, Gold)                                                     \
  {                                                                            \
    sub_test<N> Test(Q, true /*need InInv*/);                                  \
    value_type *In = Test.In.get();                                            \
    value_type *InInv = Test.InvIn.get();                                      \
    value_type *Res = Test.Res.get();                                          \
    Passed &= Test.run("operator " #Op, [=](id<1> Id) SYCL_ESIMD_KERNEL {      \
      auto Off = Id * N;                                                       \
      simd_mask<N> M0 = load<N>(In + Off);    /* 1..1,0...0 */                 \
      simd_mask<N> M1 = load<N>(InInv + Off); /* 0..0,1...1 */                 \
      simd_mask<N> M2 = M0 Op M1;                                              \
      simd_mask<N> MGold((value_type)Gold);                                    \
      check_masks_equal(M2, MGold, Res + Off);                                 \
    });                                                                        \
  }

    RUN_TEST(&&, 0);
    RUN_TEST(||, 1);
    RUN_TEST(&, 0);
    RUN_TEST(|, 1);
    RUN_TEST(^, 1);
    RUN_TEST(==, 0);
    RUN_TEST(!=, 1);
    RUN_TEST(&=, 0);
    RUN_TEST(|=, 1);
    RUN_TEST(^=, 1);
#undef RUN_TEST

    if constexpr (N == 8 || N == 32) {
      // Tests for APIs that access memory.
      {
        sub_test<N> Test(Q);
        value_type *In = Test.In.get();
        value_type *Res = Test.Res.get();
        Passed &= Test.run("load constructor", [=](id<1> Id) SYCL_ESIMD_KERNEL {
          auto Off = Id * N;
          simd_mask<N> M0 = load<N>(In + Off);
          simd_mask<N> M1(In + Off);
          check_masks_equal(M0, M1, Res + Off);
        });
      }
      {
        sub_test<N> Test(Q);
        value_type *In = Test.In.get();
        value_type *Res = Test.Res.get();
        Passed &= Test.run("copy_from", [=](id<1> Id) SYCL_ESIMD_KERNEL {
          auto Off = Id * N;
          simd_mask<N> M0 = load<N>(In + Off);
          simd_mask<N> M1;
          M1.copy_from(In + Off);
          check_masks_equal(M0, M1, Res + Off);
        });
      }
      {
        sub_test<N> Test(Q, true /*need InInv*/);
        value_type *In = Test.In.get();
        value_type *InInv = Test.InvIn.get();
        value_type *Res = Test.Res.get();
        Passed &= Test.run("copy_to", [=](id<1> Id) SYCL_ESIMD_KERNEL {
          auto Off = Id * N;
          simd_mask<N> M0 = load<N>(In + Off);
          M0.copy_to(InInv + Off);
          simd_mask<N> M1 = load<N>(InInv + Off);
          check_masks_equal(M0, M1, Res + Off);
        });
      }
      // Tests for APIs select operation.
      {
        sub_test<N> Test(Q, true /*need InInv*/);
        value_type *In = Test.In.get();
        value_type *InInv = Test.InvIn.get();
        value_type *Res = Test.Res.get();
        Passed &= Test.run("read/write through simd_mask::select() ",
                           [=](id<1> Id) SYCL_ESIMD_KERNEL {
                             auto Off = Id * N;
                             simd_mask<N> M0 = load<N>(In + Off); // 1..1,0...0
                             simd_mask<N> M1(0);
                             // swap halves of M0 into M1
                             M1.template select<N / 2, 1>(0) =
                                 M0.template select<N / 2, 1>(N / 2);
                             M1.template select<N / 2, 1>(N / 2) =
                                 M0.template select<N / 2, 1>(0);
                             // Read the inversed mask, which should be equal to
                             // M1
                             simd_mask<N> M2 = load<N>(InInv + Off);
                             M2.template select<N / 2, 1>(0) &=
                                 M1.template select<N / 2, 1>(0); // no-op
                             check_masks_equal(M1, M2, Res + Off);
                           });
      }
    }
    return Passed;
  }
};

int main(int argc, char **argv) {
  queue Q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());
  auto Dev = Q.get_device();
  std::cout << "Running on " << Dev.get_info<sycl::info::device::name>()
            << "\n";
  bool Passed = true;
  // Run tests for different mask size, including the one exceeding the h/w flag
  // register width and being not multiple of such.
  Passed &= simd_mask_api_test<8>().run(Q);
  Passed &= simd_mask_api_test<32>().run(Q);
  Passed &= simd_mask_api_test<67>().run(Q);
  std::cout << (Passed ? "Test Passed\n" : "Test FAILED\n");
  return Passed ? 0 : 1;
}
