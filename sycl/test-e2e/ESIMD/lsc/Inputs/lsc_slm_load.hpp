#include "../../esimd_test_utils.hpp"

using namespace sycl;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;

template <typename T, uint32_t Groups, uint32_t LocalRange, uint16_t VL,
          uint16_t NChannels, bool Transpose, bool TestMergeOperand,
          lsc_data_size DS = lsc_data_size::default_size>
bool test(queue Q, uint32_t PMask = ~0) {
  using Tuint = esimd_test::uint_type_t<sizeof(T)>;

  static_assert((NChannels == 1) || !Transpose,
                "Transpose must have exec size 1");
  if constexpr (DS == lsc_data_size::u8u32 || DS == lsc_data_size::u16u32) {
    static_assert(!Transpose, "Conversion types may not use vector");
    static_assert(NChannels == 1, "Only D32 and D64 support vector load");
  }

  static_assert(DS != lsc_data_size::u16u32h, "D16U32h not supported in HW");

  if constexpr (!Transpose && NChannels > 1) {
    static_assert(VL == 16 || VL == 32,
                  "IGC prohibits execution size less than SIMD size when "
                  "vector size is greater than 1");
  }

  std::cout << "Running test: T=" << esimd_test::type_name<T>() << ", VL=" << VL
            << ", NChannels=" << NChannels
            << ", DS=" << esimd_test::toString(DS)
            << ", Transpose=" << Transpose
            << ", TestMergeOperand=" << TestMergeOperand
            << ", Groups=" << Groups << ", LocalRange=" << LocalRange
            << std::endl;

  Tuint VMask = static_cast<Tuint>(-1);
  if constexpr (DS == lsc_data_size::u8u32)
    VMask = 0xff;
  else if constexpr (DS == lsc_data_size::u16u32)
    VMask = 0xffff;
  else if constexpr (DS == lsc_data_size::u16u32h)
    VMask = 0xffff0000;

  nd_range<1> Range{range<1>{Groups * LocalRange}, range<1>{LocalRange}};
  uint32_t OutSize = Groups * LocalRange * VL * NChannels;
  T *Out = malloc_shared<T>(OutSize, Q);
  memset(Out, 0, OutSize * sizeof(T));
  T MergeValue = 2;

  try {
    Q.parallel_for(Range, [=](sycl::nd_item<1> NDId) SYCL_ESIMD_KERNEL {
       uint32_t GID = NDId.get_global_id(0);
       uint32_t LID = NDId.get_local_id(0);
       uint32_t GroupID = NDId.get_group_linear_id();

       // Allocate and init 128-byte multiple size SLM memory with
       // consequential values. i-th group gets values:
       // {0, 1, 2, ...} + GroupID * 1000000.
       constexpr uint32_t ResultSIMDByteSize = VL * NChannels * sizeof(T);
       constexpr uint32_t SLMSize =
           (ResultSIMDByteSize * LocalRange + 127) & ~127;
       slm_init<SLMSize>();
       if (NDId.get_local_id(0) == 0) {
         simd<Tuint, 4> Vals(GroupID * 1000000, 1);
         for (int I = 0; I < SLMSize; I += 4 * sizeof(T)) {
           slm_block_store<Tuint, 4>(I, Vals);
           Vals += 4;
         }
       }
       barrier();

       if constexpr (Transpose) {
         if constexpr (TestMergeOperand) {
           simd_mask<1> Pred =
               (GID & 0x1) == 0; // Do actual load of even elements.
           simd<T, VL> OldValues(GID, 1);
           auto Vals = lsc_slm_block_load<T, VL, DS>(LID * VL * sizeof(T), Pred,
                                                     OldValues);
           Vals.copy_to(Out + GID * VL);
         } else {
           auto Vals = lsc_slm_block_load<T, VL, DS>(LID * VL * sizeof(T));
           Vals.copy_to(Out + GID * VL);
         }
       } else {
         simd<uint32_t, VL> Offsets(LID * VL * NChannels * sizeof(T),
                                    NChannels * sizeof(T));

         // Create  the predicate for the gather from 'PMask'.
         simd_mask<VL> Pred;
         for (int I = 0; I < VL; I++)
           Pred.template select<1, 1>(I) = (PMask >> I) & 1;

         simd<T, VL * NChannels> Vals;
         if constexpr (TestMergeOperand) {
           simd<T, VL * NChannels> OldVals = MergeValue;
           Vals = lsc_slm_gather<T, NChannels, DS>(Offsets, Pred, OldVals);
         } else {
           Vals = lsc_slm_gather<T, NChannels, DS>(Offsets, Pred);
         }

         Vals.copy_to(Out + GID * VL * NChannels);
       }
     }).wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    sycl::free(Out, Q);
    return false;
  }

  int NErrors = 0;

  if constexpr (Transpose) {
    for (uint32_t I = 0; I < OutSize; I++) {
      uint32_t GroupId = I / (LocalRange * VL * NChannels);
      uint32_t LID = I % (LocalRange * VL * NChannels);
      uint32_t GID = I / VL;
      bool Pred = (GID & 0x1) == 0;
      Tuint ExpectedVal = GroupId * 1000000 + LID;
      if (TestMergeOperand && !Pred)
        ExpectedVal = sycl::bit_cast<Tuint>((T)(GID + (I % VL)));

      if (sycl::bit_cast<Tuint>(Out[I]) != ExpectedVal && NErrors++ < 32) {
        std::cout << "Error: " << I << ": Value = " << Out[I]
                  << ", Expected value = " << ExpectedVal << std::endl;
      }
    }
  } else {
    for (uint32_t I = 0; I < OutSize; I += VL * NChannels) {
      uint32_t GroupId = I / (LocalRange * VL * NChannels);
      uint32_t LID = I % (LocalRange * VL * NChannels);
      Tuint ExpectedValBase = GroupId * 1000000 + LID;
      for (int ChannelId = 0; ChannelId < NChannels; ChannelId++) {
        for (int J = 0; J < VL; J++) {
          uint32_t OutIndex = I + ChannelId * VL + J;

          bool IsMaskSet = (PMask >> J) & 1;
          if (!TestMergeOperand && !IsMaskSet)
            continue;
          Tuint ExpectedVal =
              IsMaskSet ? (ExpectedValBase + ChannelId + J * NChannels) & VMask
                        : sycl::bit_cast<Tuint>(MergeValue);
          Tuint ComputedVal = sycl::bit_cast<Tuint>(Out[OutIndex]) & VMask;
          if (ComputedVal != ExpectedVal && NErrors++ < 32) {
            std::cout << "Error: " << OutIndex << ": Value = " << ComputedVal
                      << ", Expected value = " << ExpectedVal
                      << ", Mask = " << IsMaskSet << std::endl;
          }
        }
      }
    }
  }

  sycl::free(Out, Q);

  if (NErrors)
    std::cout << " FAILED" << std::endl;
  return NErrors == 0;
}
