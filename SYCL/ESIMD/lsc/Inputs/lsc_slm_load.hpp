#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include <iostream>

using namespace sycl;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;

// TODO: The SPEC does not say what values are returned for lsc_slm_gather
// when the corresponding elements of the predicate/mask is zero.
// It is assumed to be undefined values there.
// Thus this test does not check those elements now. From the API point of view
// it may be better to have another argument for the values being copied to
// the result when the mask bit is 0.

template <int CaseNum, typename T, uint32_t Groups, uint32_t LocalRange,
          uint16_t VL, uint16_t NChannels, bool Transpose,
          lsc_data_size DS = lsc_data_size::default_size>
bool test(uint32_t PMask = ~0) {
  static_assert((NChannels == 1) || !Transpose,
                "Transpose must have exec size 1");
  if constexpr (DS == lsc_data_size::u8u32 || DS == lsc_data_size::u16u32) {
    static_assert(!Transpose, "Conversion types may not use vector");
    static_assert(NChannels == 1, "Only D32 and D64 support vector load");
  }

  static_assert(DS != lsc_data_size::u16u32h, "D16U32h not supported in HW");
  static_assert(sizeof(T) >= 4,
                "D8 and D16 are valid only in 2D block load/store");

  if constexpr (!Transpose && NChannels > 1) {
    static_assert(VL == 16 || VL == 32,
                  "IGC prohibits execution size less than SIMD size when "
                  "vector size is greater than 1");
  }

  T VMask = static_cast<T>(-1);
  if constexpr (DS == lsc_data_size::u8u32)
    VMask = static_cast<T>(0xff);
  else if constexpr (DS == lsc_data_size::u16u32)
    VMask = static_cast<T>(0xffff);
  else if constexpr (DS == lsc_data_size::u16u32h)
    VMask = static_cast<T>(0xffff0000);

  queue Q(gpu_selector_v);
  auto D = Q.get_device();
  std::cout << "Running case #" << CaseNum << " on "
            << D.get_info<sycl::info::device::name>() << std::endl;

  nd_range<1> Range{range<1>{Groups * LocalRange}, range<1>{LocalRange}};
  constexpr uint16_t OutSize = Groups * LocalRange * VL * NChannels;
  T *Out = malloc_shared<T>(OutSize, Q);
  memset(Out, 0, OutSize * sizeof(T));

  try {
    Q.submit([&](handler &cgh) {
       cgh.parallel_for(Range, [=](sycl::nd_item<1> NDId) SYCL_ESIMD_KERNEL {
         uint32_t GID = NDId.get_global_id(0);
         uint32_t LID = NDId.get_local_id(0);
         uint32_t GroupID = NDId.get_group_linear_id();

         // Allocate and init 128-byte multiple size SLM memory with
         // consequential values. i-th group gets values:
         // {0, 1, 2, ...} + GroupID * 1000000.
         constexpr uint32_t ResultSIMDByteSize = VL * NChannels * sizeof(T);
         constexpr uint32_t SLMSize =
             (ResultSIMDByteSize * LocalRange + 127) & ~127;
         slm_init(SLMSize);
         if (NDId.get_local_id(0) == 0) {
           simd<T, 4> Vals(GroupID * 1000000, 1);
           for (int I = 0; I < SLMSize; I += 4 * sizeof(T)) {
             slm_block_store<T, 4>(I, Vals);
             Vals += 4;
           }
         }
         barrier();

         if constexpr (Transpose) {
           auto Vals = lsc_slm_block_load<T, VL, DS>(LID * VL * sizeof(T));
           Vals.copy_to(Out + GID * VL);
         } else {
           simd<uint32_t, VL> Offsets(LID * VL * NChannels * sizeof(T),
                                      NChannels * sizeof(T));

           // Create  the predicate for the gather from 'PMask'.
           simd_mask<VL> Pred;
           for (int I = 0; I < VL; I++)
             Pred.template select<1, 1>(I) = (PMask >> I) & 1;

           simd<T, VL *NChannels> Vals =
               lsc_slm_gather<T, NChannels, DS>(Offsets, Pred);

           Vals.copy_to(Out + GID * VL * NChannels);
         }
       });
     }).wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    sycl::free(Out, Q);
    return false;
  }

  bool Passed = true;

  if constexpr (Transpose) {
    for (uint32_t I = 0; I < OutSize; I++) {
      uint32_t GroupId = I / (LocalRange * VL * NChannels);
      uint32_t LID = I % (LocalRange * VL * NChannels);
      T ExpectedVal = GroupId * 1000000 + LID;
      if (Out[I] != ExpectedVal) {
        Passed = false;
        std::cout << I << ": Value = " << Out[I]
                  << ", Expected value = " << ExpectedVal << std::endl;
      }
    }
  } else {
    for (uint32_t I = 0; I < OutSize; I += VL * NChannels) {
      uint32_t GroupId = I / (LocalRange * VL * NChannels);
      uint32_t LID = I % (LocalRange * VL * NChannels);
      T ExpectedValBase = GroupId * 1000000 + LID;
      for (int ChannelId = 0; ChannelId < NChannels; ChannelId++) {
        for (int J = 0; J < VL; J++) {
          uint32_t OutIndex = I + ChannelId * VL + J;

          if (((PMask >> J) & 1) == 0)
            continue;
          T ExpectedVal = (ExpectedValBase + ChannelId + J * NChannels) & VMask;
          if (Out[OutIndex] != ExpectedVal) {
            Passed = false;
            std::cout << OutIndex << ": Value = " << Out[OutIndex]
                      << ", Expected value = " << ExpectedVal << std::endl;
          }
        }
      }
    }
  }

  sycl::free(Out, Q);

  if (!Passed)
    std::cout << "Case #" << CaseNum << " FAILED" << std::endl;
  return Passed;
}
