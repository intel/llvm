// RUN-IF: sg-8, %{build} -DSG_SIZE=8 -o %t8.out && %{run} %t8.out
// RUN-IF: sg-16, %{build} -DSG_SIZE=16 -o %t16.out && %{run} %t16.out
// RUN-IF: sg-32, %{build} -DSG_SIZE=32 -o %t32.out && %{run} %t32.out

// Test CPU AOT as well when possible.
// RUN-IF: any-device-is-cpu && opencl-aot, %{run-aux} %clangxx -fsycl -fsycl-targets=spir64_x86_64 -DSG_SIZE=8 -o %t8.x86.out %s
// RUN-IF: sg-8 && cpu && opencl-aot, %{run} %t8.x86.out
// RUN-IF: any-device-is-cpu && opencl-aot, %{run-aux} %clangxx -fsycl -fsycl-targets=spir64_x86_64 -DSG_SIZE=16 -o %t16.x86.out %s
// RUN-IF: sg-16 && cpu && opencl-aot, %{run} %t16.x86.out
// RUN-IF: any-device-is-cpu && opencl-aot, %{run-aux} %clangxx -fsycl -fsycl-targets=spir64_x86_64 -DSG_SIZE=32 -o %t32.x86.out %s
// RUN-IF: sg-32 && cpu && opencl-aot, %{run} %t32.x86.out

// REQUIRES: cpu || gpu
// REQUIRES: sg-8 || sg-16 || sg-32
// REQUIRES: aspect-ext_oneapi_fragment

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/fragment.hpp>
#include <sycl/group_algorithm.hpp>
#include <sycl/group_barrier.hpp>

#include <cstdio>

static int Failures = 0;
#define CHECK(Cond)                                                            \
  do {                                                                         \
    if (!(Cond)) {                                                             \
      std::printf("FAIL WI=%d: %s\n", static_cast<int>(WI), #Cond);            \
      ++Failures;                                                              \
    }                                                                          \
  } while (0)

namespace syclex = sycl::ext::oneapi::experimental;

class TestKernel;

constexpr uint32_t SGSize = SG_SIZE;
constexpr uint32_t ArbitraryItem = 5;

int main() {
  sycl::queue Q;

  sycl::buffer<size_t, 1> TmpBuf{sycl::range{SGSize}};
  sycl::buffer<bool, 1> BarrierBuf{sycl::range{SGSize}};
  sycl::buffer<bool, 1> BroadcastBuf{sycl::range{SGSize}};
  sycl::buffer<bool, 1> AnyBuf{sycl::range{SGSize}};
  sycl::buffer<bool, 1> AllBuf{sycl::range{SGSize}};
  sycl::buffer<bool, 1> NoneBuf{sycl::range{SGSize}};
  sycl::buffer<bool, 1> ReduceBuf{sycl::range{SGSize}};
  sycl::buffer<bool, 1> ExScanBuf{sycl::range{SGSize}};
  sycl::buffer<bool, 1> IncScanBuf{sycl::range{SGSize}};
  sycl::buffer<bool, 1> ExScanFloatBuf{sycl::range{SGSize}};
  sycl::buffer<bool, 1> IncScanFloatBuf{sycl::range{SGSize}};
  sycl::buffer<bool, 1> ExScanInitBuf{sycl::range{SGSize}};
  sycl::buffer<bool, 1> IncScanInitBuf{sycl::range{SGSize}};
  sycl::buffer<bool, 1> ShiftLeftBuf{sycl::range{SGSize}};
  sycl::buffer<bool, 1> ShiftRightBuf{sycl::range{SGSize}};
  sycl::buffer<bool, 1> SelectBuf{sycl::range{SGSize}};
  sycl::buffer<bool, 1> PermuteXorBuf{sycl::range{SGSize}};

  const auto NDR = sycl::nd_range<1>{SGSize, SGSize};
  Q.submit([&](sycl::handler &CGH) {
    sycl::accessor TmpAcc{TmpBuf, CGH, sycl::write_only};
    sycl::accessor BarrierAcc{BarrierBuf, CGH, sycl::write_only};
    sycl::accessor BroadcastAcc{BroadcastBuf, CGH, sycl::write_only};
    sycl::accessor AnyAcc{AnyBuf, CGH, sycl::write_only};
    sycl::accessor AllAcc{AllBuf, CGH, sycl::write_only};
    sycl::accessor NoneAcc{NoneBuf, CGH, sycl::write_only};
    sycl::accessor ReduceAcc{ReduceBuf, CGH, sycl::write_only};
    sycl::accessor ExScanAcc{ExScanBuf, CGH, sycl::write_only};
    sycl::accessor IncScanAcc{IncScanBuf, CGH, sycl::write_only};
    sycl::accessor ExScanFloatAcc{ExScanFloatBuf, CGH, sycl::write_only};
    sycl::accessor IncScanFloatAcc{IncScanFloatBuf, CGH, sycl::write_only};
    sycl::accessor ExScanInitAcc{ExScanInitBuf, CGH, sycl::write_only};
    sycl::accessor IncScanInitAcc{IncScanInitBuf, CGH, sycl::write_only};
    sycl::accessor ShiftLeftAcc{ShiftLeftBuf, CGH, sycl::write_only};
    sycl::accessor ShiftRightAcc{ShiftRightBuf, CGH, sycl::write_only};
    sycl::accessor SelectAcc{SelectBuf, CGH, sycl::write_only};
    sycl::accessor PermuteXorAcc{PermuteXorBuf, CGH, sycl::write_only};
    const auto KernelFunc =
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SGSize)]] {
          auto WI = item.get_global_id();
          auto SG = item.get_sub_group();

          uint32_t OriginalLID = SG.get_local_linear_id();

          // Given the dynamic nature of opportunistic groups, the simplest
          // case we can reason about is a single work-item. This isn't a very
          // robust test, but choosing an arbitrary work-item (i.e. rather
          // than the leader) should test an implementation's ability to handle
          // arbitrary group membership.
          if (OriginalLID == ArbitraryItem) {
            auto OpportunisticGroup =
                syclex::this_work_item::get_opportunistic_group();

            // This is trivial, but does test that group_barrier can be called.
            TmpAcc[WI] = 1;
            sycl::group_barrier(OpportunisticGroup);
            size_t Visible = TmpAcc[WI];
            BarrierAcc[WI] = (Visible == 1);

            // Simple check of group algorithms.
            uint32_t LID = OpportunisticGroup.get_local_linear_id();
            uint32_t OpportunisticGroupSize =
                OpportunisticGroup.get_local_linear_range();

            uint32_t BroadcastResult =
                sycl::group_broadcast(OpportunisticGroup, OriginalLID, 0);
            BroadcastAcc[WI] = (BroadcastResult == OriginalLID);

            bool AnyResult = sycl::any_of_group(OpportunisticGroup, (LID == 0));
            AnyAcc[WI] = (AnyResult == true);

            bool AllResult = sycl::all_of_group(OpportunisticGroup, (LID == 0));
            AllAcc[WI] = (AllResult == true);

            bool NoneResult =
                sycl::none_of_group(OpportunisticGroup, (LID != 0));
            NoneAcc[WI] = (NoneResult == true);

            uint32_t ReduceResult =
                sycl::reduce_over_group(OpportunisticGroup, 1, sycl::plus<>());
            ReduceAcc[WI] =
                (ReduceResult == OpportunisticGroup.get_local_linear_range());

            uint32_t ExScanResult = sycl::exclusive_scan_over_group(
                OpportunisticGroup, 1, sycl::plus<>());
            ExScanAcc[WI] = (ExScanResult == LID);

            uint32_t IncScanResult = sycl::inclusive_scan_over_group(
                OpportunisticGroup, 1, sycl::plus<>());
            IncScanAcc[WI] = (IncScanResult == LID + 1);

            float ExScanFloatResult = sycl::exclusive_scan_over_group(
                OpportunisticGroup, 1.0f, sycl::plus<>());
            ExScanFloatAcc[WI] = (ExScanFloatResult == static_cast<float>(LID));

            float IncScanFloatResult = sycl::inclusive_scan_over_group(
                OpportunisticGroup, 1.0f, sycl::plus<>());
            IncScanFloatAcc[WI] =
                (IncScanFloatResult == static_cast<float>(LID + 1));

            constexpr uint32_t Init = 42;
            uint32_t ExScanInitResult = sycl::exclusive_scan_over_group(
                OpportunisticGroup, 1, Init, sycl::plus<>());
            ExScanInitAcc[WI] = (ExScanInitResult == Init + LID);

            uint32_t IncScanInitResult = sycl::inclusive_scan_over_group(
                OpportunisticGroup, 1, sycl::plus<>(), Init);
            IncScanInitAcc[WI] = (IncScanInitResult == Init + LID + 1);

            uint32_t ShiftLeftResult =
                sycl::shift_group_left(OpportunisticGroup, LID, 2);
            ShiftLeftAcc[WI] = (LID + 2 >= OpportunisticGroupSize ||
                                ShiftLeftResult == LID + 2);

            uint32_t ShiftRightResult =
                sycl::shift_group_right(OpportunisticGroup, LID, 2);
            ShiftRightAcc[WI] = (LID < 2 || ShiftRightResult == LID - 2);

            uint32_t SelectResult = sycl::select_from_group(
                OpportunisticGroup, LID,
                (OpportunisticGroup.get_local_id() + 2) %
                    OpportunisticGroupSize);
            SelectAcc[WI] =
                (SelectResult == (LID + 2) % OpportunisticGroupSize);

            uint32_t PermuteXorResult =
                sycl::permute_group_by_xor(OpportunisticGroup, LID, 0);
            PermuteXorAcc[WI] = (PermuteXorResult == LID);
          } else {
            BarrierAcc[WI] = false;
            BroadcastAcc[WI] = false;
            AnyAcc[WI] = false;
            AllAcc[WI] = false;
            NoneAcc[WI] = false;
            ReduceAcc[WI] = false;
            ExScanAcc[WI] = false;
            IncScanAcc[WI] = false;
            ExScanFloatAcc[WI] = false;
            IncScanFloatAcc[WI] = false;
            ExScanInitAcc[WI] = false;
            IncScanInitAcc[WI] = false;
            ShiftLeftAcc[WI] = false;
            ShiftRightAcc[WI] = false;
            SelectAcc[WI] = false;
            PermuteXorAcc[WI] = false;
          }
        };
    CGH.parallel_for<TestKernel>(NDR, KernelFunc);
  });

  sycl::host_accessor BarrierAcc{BarrierBuf, sycl::read_only};
  sycl::host_accessor BroadcastAcc{BroadcastBuf, sycl::read_only};
  sycl::host_accessor AnyAcc{AnyBuf, sycl::read_only};
  sycl::host_accessor AllAcc{AllBuf, sycl::read_only};
  sycl::host_accessor NoneAcc{NoneBuf, sycl::read_only};
  sycl::host_accessor ReduceAcc{ReduceBuf, sycl::read_only};
  sycl::host_accessor ExScanAcc{ExScanBuf, sycl::read_only};
  sycl::host_accessor IncScanAcc{IncScanBuf, sycl::read_only};
  sycl::host_accessor ExScanFloatAcc{ExScanFloatBuf, sycl::read_only};
  sycl::host_accessor IncScanFloatAcc{IncScanFloatBuf, sycl::read_only};
  sycl::host_accessor ExScanInitAcc{ExScanInitBuf, sycl::read_only};
  sycl::host_accessor IncScanInitAcc{IncScanInitBuf, sycl::read_only};
  sycl::host_accessor ShiftLeftAcc{ShiftLeftBuf, sycl::read_only};
  sycl::host_accessor ShiftRightAcc{ShiftRightBuf, sycl::read_only};
  sycl::host_accessor SelectAcc{SelectBuf, sycl::read_only};
  sycl::host_accessor PermuteXorAcc{PermuteXorBuf, sycl::read_only};
  for (uint32_t WI = 0; WI < SGSize; ++WI) {
    bool ExpectedResult = (WI == ArbitraryItem);
    CHECK(BarrierAcc[WI] == ExpectedResult);
    CHECK(BroadcastAcc[WI] == ExpectedResult);
    CHECK(AnyAcc[WI] == ExpectedResult);
    CHECK(AllAcc[WI] == ExpectedResult);
    CHECK(NoneAcc[WI] == ExpectedResult);
    CHECK(ReduceAcc[WI] == ExpectedResult);
    CHECK(ExScanAcc[WI] == ExpectedResult);
    CHECK(IncScanAcc[WI] == ExpectedResult);
    CHECK(ExScanFloatAcc[WI] == ExpectedResult);
    CHECK(IncScanFloatAcc[WI] == ExpectedResult);
    CHECK(ExScanInitAcc[WI] == ExpectedResult);
    CHECK(IncScanInitAcc[WI] == ExpectedResult);
    CHECK(ShiftLeftAcc[WI] == ExpectedResult);
    CHECK(ShiftRightAcc[WI] == ExpectedResult);
    CHECK(SelectAcc[WI] == ExpectedResult);
    CHECK(PermuteXorAcc[WI] == ExpectedResult);
  }
  std::printf("Total failures: %d\n", Failures);
  return Failures != 0;
}
