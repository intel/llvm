//==- hier_par_wgscope_impl.hpp --- hier. parallelism test for WG scope ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Shared code for the hier_par_wgscope* tests

#include <iostream>
#include <memory>
#include <sycl/sycl.hpp>

using namespace sycl;

// Tests complex patterns of data and code usage at work group scope:
// - wg scope code and data inside a loop
// - PFWIs inside complex control flow
static bool testWgScope(queue &Q) {
  std::cout << "+++ Testing work group scope code and Data handling...\n";
  const int MAX_HW_SIMD = 2;
  const int N_WG = 3;
  const int PHYS_WG_SIZE = MAX_HW_SIMD + 3;
  const int FLEX_RANGE_SIZE = PHYS_WG_SIZE + 2;
  const int N_ITER = 3;
  const int N_INNER_ITER = 2;
  const int N_OUTER_ITER = 3;
  const int RangeLength = N_WG * PHYS_WG_SIZE;
  const int VAL1 = 10;
  const int VAL2 = 1000;
  const int GROUP_ID_SPLIT = 2;

  std::unique_ptr<int[]> Data(new int[RangeLength]);
  int *Ptr = Data.get();
  std::memset(Ptr, 0, RangeLength * sizeof(Ptr[0]));

  try {
    // 1 element per physical WI:
    buffer<int, 1> Buf(Ptr, range<1>(RangeLength));
    int N = N_INNER_ITER;

    Q.submit([&](handler &cgh) {
      auto DevPtr = Buf.get_access<access::mode::read_write>(cgh);

      cgh.parallel_for_work_group<class hpar_hw>(
          range<1>(N_WG), range<1>(PHYS_WG_SIZE), [=](group<1> G) {
            // offset of group'S chunk in the 'DevPtr' array:
            int GroupOff = PHYS_WG_SIZE * G.get_id(0);

            for (int CntOuter = 0; CntOuter < N_OUTER_ITER; CntOuter++) {
              // local group-shared array; declared inside a loop
              int WgShared[PHYS_WG_SIZE];

              // Step 0 - initialize it to VAL1
              for (int Cnt = 0; Cnt < PHYS_WG_SIZE; Cnt++) {
                WgShared[Cnt] = VAL1;
              }

              // Step 1 - increment array elements in each WI
              G.parallel_for_work_item([&](h_item<1> I) {
                int Ind = I.get_global_id(0);
                DevPtr[Ind]++;
              });

              // only for groups with IDs 0, 1:
              if (G.get_id(0) < GROUP_ID_SPLIT) {
                // invoke PFWI in the loop
                // Step 2a (1) - additionally increment 1 element per group as a
                //   "side effect" of the increment part
                for (int Cnt = 0; Cnt < N; Cnt++, DevPtr[GroupOff]++) {
                  // Step 2a
                  G.parallel_for_work_item([&](h_item<1> I) {
                    int Ind = I.get_global_id(0);
                    //   2a (2) add VAL1 in each physical WI
                    DevPtr[Ind] += VAL1;
                    //   2a (3) increment elements of the group-local shared
                    //   array
                    WgShared[I.get_local_id(0)]++;
                  });
                  // Step 3a - one more per-group increment
                  DevPtr[GroupOff]++;
                }
              }
              // only for groups with IDs 2 and up:
              else {
                // Step 2b - add VAL2 in each logical WI in a flexible range
                G.parallel_for_work_item(range<1>{FLEX_RANGE_SIZE},
                                         [&](h_item<1> I) {
                                           int Ind = I.get_global_id(0);
                                           DevPtr[Ind] += VAL2;
                                         });
              }
              // Step 4 - store the local shared array into its chunk within the
              // global array
              for (int Cnt = 0; Cnt < PHYS_WG_SIZE; Cnt++) {
                DevPtr[GroupOff + Cnt] += WgShared[Cnt];
              }
            }
          });
    });
    Q.wait();
  } catch (sycl::exception const &E) {
    std::cout << "SYCL exception caught: " << E.what() << '\n';
    return 2;
  }
  // verify
  int ErrCnt = 0;

  for (int WG = 0; WG < N_WG; WG++) {
    for (int WI = 0; WI < PHYS_WG_SIZE; WI++) {
      int GlobalId = WG * PHYS_WG_SIZE + WI;
      // Step 0
      int Gold = VAL1;
      // Step 1
      Gold++;

      if (WG < GROUP_ID_SPLIT) {
        int GoldInner = 0;
        if (WI == 0)
          // Step 2a - 1
          GoldInner++;
        // Step 2a - 2
        GoldInner += VAL1;
        // Step 2a - 3
        GoldInner++;
        if (WI == 0)
          // Step 3a
          GoldInner++;
        Gold += GoldInner * N_INNER_ITER;
      } else {
        // Step 2b
        for (int I = 0; I < FLEX_RANGE_SIZE / PHYS_WG_SIZE; I++)
          Gold += VAL2;
        if (WI < FLEX_RANGE_SIZE % PHYS_WG_SIZE)
          Gold += VAL2;
      }
      Gold *= N_OUTER_ITER;
      int Val = Ptr[GlobalId];

      if (Gold != Val) {
        if (++ErrCnt < 16) {
          std::cout << "*** ERROR at WG=" << WG << " WI=" << WI << ": " << Val
                    << " != " << Gold << "(Gold)\n";
        }
      }
    }
  }

  if (ErrCnt == 0) {
    std::cout << "  Passed\n";
    return true;
  }
  std::cout << "  Failed. Failure rate: " << ErrCnt << "/" << RangeLength << "("
            << ErrCnt / (float)RangeLength * 100.f << "%)\n";
  return false;
}

template <typename GoldFnTy>
bool verify(int testcase, int RangeLength, int *Ptr, GoldFnTy get_gold) {
  int ErrCnt = 0;

  for (int I = 0; I < RangeLength; I++) {
    int Gold = get_gold(I);

    if (Ptr[I] != Gold) {
      if (++ErrCnt < 20) {
        std::cout << testcase << " - ERROR at " << I << ": " << Ptr[I]
                  << " != " << Gold << "(expected)\n";
      }
    }
  }
  if (ErrCnt > 0)
    std::cout << "-- Failure rate: " << ErrCnt << "/" << RangeLength << "("
              << ErrCnt / (float)RangeLength * 100.f << "%)\n";
  return ErrCnt == 0;
}

struct MyStruct {
  size_t x;
  size_t y;
};

// Tests complex private_memory usage:
// - Data type is a structure
// - two different variables are used, both are live across two PFWI scopes
bool testPrivateMemory(queue &Q) {
  std::cout << "+++ Testing private_memory class implementation...\n";
  constexpr int N_ITER = 3;
  constexpr int N_WG = 2;
  constexpr int WG_X_SIZE = 7;
  constexpr int WG_Y_SIZE = 3;
  constexpr int WG_LINEAR_SIZE = WG_X_SIZE * WG_Y_SIZE;
  constexpr int RangeLength = N_WG * WG_LINEAR_SIZE;
  constexpr int C1 = 5;
  constexpr int C2 = 1;

  std::unique_ptr<int[]> Data(new int[RangeLength]);
  int *Ptr = Data.get();

  std::memset(Ptr, 0, RangeLength * sizeof(Ptr[0]));
  buffer<int, 1> Buf(Ptr, range<1>(RangeLength));
  Q.submit([&](handler &cgh) {
    auto DevPtr = Buf.get_access<access::mode::read_write>(cgh);

    cgh.parallel_for_work_group<class hpar_priv_mem>(
        range<2>(N_WG, 1), range<2>(WG_X_SIZE, WG_Y_SIZE), [=](group<2> G) {
          private_memory<MyStruct, 2> Priv1(G);
          private_memory<MyStruct, 2> Priv2(G);

          for (int Cnt = 0; Cnt < N_ITER; Cnt++) {
            G.parallel_for_work_item(
                range<2>(WG_X_SIZE, WG_Y_SIZE), [&](h_item<2> I) {
                  auto GlobId = I.get_global().get_linear_id();
                  DevPtr[GlobId]++;
                  MyStruct &S1 = Priv1(I);
                  S1.x = GlobId;
                  S1.y = C1;
                  MyStruct &S2 = Priv2(I);
                  S2.x = C2;
                  S2.y = GlobId;
                });
            G.parallel_for_work_item(range<2>(WG_X_SIZE, WG_Y_SIZE),
                                     [&](h_item<2> I) {
                                       MyStruct &S1 = Priv1(I);
                                       MyStruct &S2 = Priv2(I);
                                       DevPtr[I.get_global().get_linear_id()] +=
                                           S1.x + S1.y + S2.x + S2.y;
                                     });
          }
        });
  });
  auto Ptr1 = Buf.get_access<access::mode::read>().get_pointer();
  bool Res = verify(0, RangeLength, Ptr1, [&](int I) -> int {
    return N_ITER * (1 + C1 + C2 + 2 * I);
  });
  std::cout << (Res ? "  Passed\n" : "  FAILED\n");
  return Res;
}

int run() {
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
  std::cout << "Using device: "
            << Q.get_device().get_info<sycl::info::device::name>() << "\n";

  bool Passed = true;
  Passed &= testWgScope(Q);
  Passed &= testPrivateMemory(Q);

  if (!Passed) {
    std::cout << "FAILED\n";
    return 1;
  }
  std::cout << "Passed\n";
  return 0;
}
