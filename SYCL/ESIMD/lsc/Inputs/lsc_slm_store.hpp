#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include <iostream>

using namespace sycl;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;

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

  queue Q(gpu_selector{});
  auto D = Q.get_device();
  std::cout << "Running case #" << CaseNum << " on "
            << D.get_info<info::device::name>() << std::endl;

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

         // 1. Allocate and init 128-byte multiple size SLM memory with special
         // values.
         constexpr uint32_t ResultSIMDByteSize = VL * NChannels * sizeof(T);
         constexpr uint32_t SLMSize =
             (ResultSIMDByteSize * LocalRange + 127) & ~127;
         slm_init(SLMSize);
         if (NDId.get_local_id(0) == 0) {
           simd<T, 4> Vals = static_cast<T>(0xBAADF00DBAADF00D);
           for (int I = 0; I < SLMSize; I += 4 * sizeof(T))
             slm_block_store<T, 4>(I, Vals);
         }
         barrier();

         // 2. Use STORE intrinscis that are being verified in this test.
         if constexpr (Transpose) {
           simd<T, VL> Vals(GroupID * 1000000 + LID * 1000, 1);
           lsc_slm_block_store<T, VL, DS>(LID * VL * sizeof(T), Vals);
         } else {

           // Create  the predicate for the gather from 'PMask'.
           simd_mask<VL> Pred;
           for (int I = 0; I < VL; I++)
             Pred.template select<1, 1>(I) = (PMask >> I) & 1;

           simd<T, VL * NChannels> Vals(GroupID * 1000000 + LID * 1000, 1);
           simd<uint32_t, VL> Offsets(LID * VL * NChannels * sizeof(T),
                                      NChannels * sizeof(T));
           lsc_slm_scatter<T, NChannels, DS>(Offsets, Vals, Pred);
         }
         barrier();

         // 3. Simply load the content of SLM and store it to USM.
         if (NDId.get_local_id(0) == 0) {
           int End = LocalRange * VL * NChannels;
           for (int I = 0; I < End; I += 4) {
             auto Vals = slm_block_load<T, 4>(I * sizeof(T));

             // If 'VL' is small, simd<T, 4> cannot be safely used
             if (I + 4 > End) {
               for (int J = 0; J + I < End; J++)
                 Out[GroupID * LocalRange * VL * NChannels + I + J] =
                     (T)Vals[J];
             } else {
               Vals.copy_to(Out + GroupID * LocalRange * VL * NChannels + I);
             }
           }
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
      uint32_t GroupId = I / (LocalRange * VL);
      uint32_t LID = I / VL % LocalRange;
      T ExpectedVal = GroupId * 1000000 + LID * 1000 + I % VL;
      if (Out[I] != ExpectedVal) {
        Passed = false;
        std::cout << I << ": Value = " << Out[I]
                  << ", Expected value = " << ExpectedVal << std::endl;
      }
    }
  } else {
    for (uint32_t I = 0; I < OutSize; I += VL * NChannels) {
      uint32_t GroupId = I / (LocalRange * VL * NChannels);
      uint32_t LID = I / (VL * NChannels) % LocalRange;
      T ExpectedValBase = GroupId * 1000000 + LID * 1000 + I % (VL * NChannels);
      T ExpectedValInc = 0;
      uint32_t MaskIndex = 0;
      uint32_t MaskIndexTimer = 0;
      for (int ChannelId = 0; ChannelId < NChannels; ChannelId++) {
        for (int J = 0; J < VL; J++) {
          uint32_t OutIndex = I + ChannelId * VL + J;
          T ExpectedVal = ((PMask >> MaskIndex) & 1)
                              ? (ExpectedValBase + ExpectedValInc)
                              : static_cast<T>(0xBAADF00DBAADF00D);
          ExpectedVal &= VMask;
          MaskIndexTimer++;
          if (MaskIndexTimer >= NChannels) {
            MaskIndexTimer = 0;
            MaskIndex++;
          }

          ExpectedValInc += VL;
          if (ExpectedValInc >= VL * NChannels)
            ExpectedValInc = (ExpectedValInc % (VL * NChannels)) + 1;

          T OutVal = Out[OutIndex] & VMask;
          if (OutVal != ExpectedVal) {
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
