// REQUIRES-INTEL-DRIVER: lin: 27202, win: 101.4677
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// This test verifies usage of local_accessor methods operator[]
// and get_pointer().

#include "esimd_test_utils.hpp"

using namespace sycl;
using namespace sycl::ext::intel::esimd;

constexpr int VL = 16;

template <typename T, bool TestSubscript>
bool test(queue Q, uint32_t LocalRange, uint32_t GlobalRange) {
  std::cout << "Running case: T=" << esimd_test::type_name<T>()
            << ", TestSubscript=" << TestSubscript << std::endl;

  // The test is going to use (LocalRange * VL) elements of T type.
  auto Dev = Q.get_device();
  auto DeviceSLMSize = Dev.get_info<sycl::info::device::local_mem_size>();
  if (DeviceSLMSize < LocalRange * VL * sizeof(T)) {
    // Report an error - the test needs a fix.
    std::cerr << "Error: Test needs more SLM memory than device has!"
              << std::endl;
    return false;
  }

  T *Out = malloc_shared<T>(GlobalRange * VL, Q);
  for (int I = 0; I < GlobalRange * VL; I++)
    Out[I] = -1;

  try {
    nd_range<1> NDRange{range<1>{GlobalRange}, range<1>{LocalRange}};
    Q.submit([&](handler &CGH) {
       auto LocalAcc = local_accessor<T, 1>(LocalRange * VL, CGH);

       CGH.parallel_for(NDRange, [=](nd_item<1> Item) SYCL_ESIMD_KERNEL {
         uint32_t GID = Item.get_global_id(0);
         uint32_t LID = Item.get_local_id(0);
         uint32_t LocalAccOffset =
             static_cast<uint32_t>(reinterpret_cast<std::uintptr_t>(
                 LocalAcc.template get_multi_ptr<access::decorated::yes>()
                     .get()));
         if constexpr (TestSubscript) {
           for (int I = 0; I < VL; I++)
             LocalAcc[LID * VL + I] = GID * 100 + I;
         } else {
           simd<int, VL> IntValues(GID * 100, 1);
           simd<T, VL> ValuesToSLM = IntValues;
           slm_block_store(LocalAccOffset + LID * VL * sizeof(T), ValuesToSLM);
         }

         Item.barrier();

         if (LID == 0) {
           for (int LID = 0; LID < LocalRange; LID++) {
             if constexpr (TestSubscript) {
               for (int I = 0; I < VL; I++)
                 Out[(GID + LID) * VL + I] = LocalAcc[LID * VL + I];
             } else {
               simd<T, VL> ValuesFromSLM =
                   slm_block_load<T, VL>(LocalAccOffset + LID * VL * sizeof(T));
               ValuesFromSLM.copy_to(Out + (GID + LID) * VL);
             }
           } // end for (int LID = 0; LID < LocalRange; LID++)
         } // end if (LID == 0)
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
    T Computed = Out[I];
    if (Computed != Expected) {
      std::cout << "Error: Out[" << I << "]:" << Computed << " != " << Expected
                << ":[expected]" << std::endl;
      Pass = false;
    }
  }

  free(Out, Q);
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
