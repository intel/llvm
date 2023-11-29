//
// RUN: %{build} -fsycl-esimd-force-stateless-mem -o %t.out
// RUN: %{run} %t.out

// This test verifies usage of accessor methods operator[] and get_pointer().

#include "esimd_test_utils.hpp"

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include <iostream>

using namespace sycl;
using namespace sycl::ext::intel::esimd;

constexpr int VL = 16;

template <typename T, bool TestSubscript>
bool test(queue Q, uint32_t LocalRange, uint32_t GlobalRange) {
  std::cout << "Running case: T=" << esimd_test::type_name<T>()
            << ", TestSubscript=" << TestSubscript << std::endl;

  T *Tmp = new T[GlobalRange * VL];
  T *Out = malloc_shared<T>(GlobalRange * VL, Q);
  for (int I = 0; I < GlobalRange * VL; I++)
    Out[I] = -1;

  try {
    nd_range<1> NDRange{range<1>{GlobalRange}, range<1>{LocalRange}};
    // TODO: Try accessor with non-zero offset when ESIMD is ready.
    buffer<T, 1> TmpBuf(Tmp, GlobalRange * VL);
    Q.submit([&](handler &CGH) {
       accessor TmpAcc{TmpBuf, CGH};
       CGH.parallel_for(NDRange, [=](nd_item<1> Item) SYCL_ESIMD_KERNEL {
         uint32_t GID = Item.get_global_id(0);
         uint32_t LID = Item.get_local_id(0);
         if constexpr (TestSubscript) {
           for (int I = 0; I < VL; I++)
             TmpAcc[GID * VL + I] = GID * 100 + I;
         } else {
           T *Ptr = TmpAcc.get_pointer();
           simd<int, VL> IntValues(GID * 100, 1);
           simd<T, VL> Values = IntValues;
           block_store(Ptr + GID * VL, Values);
         }

         Item.barrier();

         if (LID == 0) {
           for (int LID = 0; LID < LocalRange; LID++) {
             if constexpr (TestSubscript) {
               for (int I = 0; I < VL; I++)
                 Out[(GID + LID) * VL + I] = TmpAcc[(GID + LID) * VL + I];
             } else {
               T *Ptr = TmpAcc.get_pointer();
               simd<T, VL> Values = block_load<T, VL>(Ptr + (GID + LID) * VL);
               Values.template copy_to(Out + (GID + LID) * VL);
             }
           } // end for (int LID = 0; LID < LocalRange; LID++)
         }   // end if (LID == 0)
       });
     }).wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    free(Out, Q);
    return false;
  }

  bool Pass = true;
  for (int I = 0; I < GlobalRange * VL; I++) {
    int GID = I / VL;
    int LID = GID % LocalRange;
    int VecElementIndex = I % VL;

    T Expected = GID * 100 + VecElementIndex;
    if (Out[I] != Expected) {
      std::cout << "Error: Out[" << I << "]:" << Out[I] << " != " << Expected
                << ":[expected]" << std::endl;
      Pass = false;
    }
    if (Tmp[I] != Expected) {
      std::cout << "Error: Tmp[" << I << "]:" << Tmp[I] << " != " << Expected
                << ":[expected]" << std::endl;
      Pass = false;
    }
  }

  free(Out, Q);
  delete[] Tmp;
  return Pass;
}

template <typename T>
bool tests(queue Q, uint32_t LocalRange, uint32_t GlobalRange) {
  constexpr bool TestSubscript = true;

  bool Pass = true;
  Pass &= test<T, TestSubscript>(Q, LocalRange, GlobalRange);
  Pass &= test<T, !TestSubscript>(Q, LocalRange, GlobalRange);

  return Pass;
}

int main() {
  auto Q = queue{gpu_selector_v};
  auto Dev = Q.get_device();
  auto DeviceSLMSize = Dev.get_info<sycl::info::device::local_mem_size>();
  std::cout << "Running on " << Dev.get_info<sycl::info::device::name>()
            << ", Local memory size available : " << DeviceSLMSize << std::endl;

  uint32_t LocalRange = 16;
  uint32_t GlobalRange = LocalRange * 2; // 2 groups.

  bool Pass = true;
  Pass &= tests<int>(Q, LocalRange, GlobalRange);
  Pass &= tests<float>(Q, LocalRange, GlobalRange);
  if (Dev.has(aspect::fp16))
    Pass &= tests<sycl::half>(Q, LocalRange, GlobalRange);

  std::cout << "Test result: " << (Pass ? "Pass" : "Fail") << std::endl;
  return Pass ? 0 : 1;
}
