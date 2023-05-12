// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
// REQUIRES: gpu
// UNSUPPORTED: cuda || hip

#include <sycl/sycl.hpp>
#include <vector>
namespace syclex = sycl::ext::oneapi::experimental;

class TestKernel;

constexpr uint32_t SGSize = 32;
constexpr uint32_t ArbitraryItem = 5;

int main() {
  sycl::queue Q;

  auto SGSizes = Q.get_device().get_info<sycl::info::device::sub_group_sizes>();
  if (std::find(SGSizes.begin(), SGSizes.end(), SGSize) == SGSizes.end()) {
    std::cout << "Test skipped due to missing support for sub-group size 32."
              << std::endl;
    return 0;
  }

  sycl::buffer<size_t, 1> TmpBuf{sycl::range{SGSize}};
  sycl::buffer<bool, 1> BarrierBuf{sycl::range{SGSize}};
  sycl::buffer<bool, 1> BroadcastBuf{sycl::range{SGSize}};
  sycl::buffer<bool, 1> AnyBuf{sycl::range{SGSize}};
  sycl::buffer<bool, 1> AllBuf{sycl::range{SGSize}};
  sycl::buffer<bool, 1> NoneBuf{sycl::range{SGSize}};
  sycl::buffer<bool, 1> ReduceBuf{sycl::range{SGSize}};
  sycl::buffer<bool, 1> ExScanBuf{sycl::range{SGSize}};
  sycl::buffer<bool, 1> IncScanBuf{sycl::range{SGSize}};

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
                syclex::this_kernel::get_opportunistic_group();

            // This is trivial, but does test that group_barrier can be called.
            TmpAcc[WI] = 1;
            sycl::group_barrier(OpportunisticGroup);
            size_t Visible = TmpAcc[WI];
            BarrierAcc[WI] = (Visible == 1);

            // Simple check of group algorithms.
            uint32_t LID = OpportunisticGroup.get_local_linear_id();

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
          } else {
            BarrierAcc[WI] = false;
            BroadcastAcc[WI] = false;
            AnyAcc[WI] = false;
            AllAcc[WI] = false;
            NoneAcc[WI] = false;
            ReduceAcc[WI] = false;
            ExScanAcc[WI] = false;
            IncScanAcc[WI] = false;
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
  for (uint32_t WI = 0; WI < 32; ++WI) {
    bool ExpectedResult = (WI == ArbitraryItem);
    assert(BarrierAcc[WI] == ExpectedResult);
    assert(BroadcastAcc[WI] == ExpectedResult);
    assert(AnyAcc[WI] == ExpectedResult);
    assert(AllAcc[WI] == ExpectedResult);
    assert(NoneAcc[WI] == ExpectedResult);
    assert(ReduceAcc[WI] == ExpectedResult);
    assert(ExScanAcc[WI] == ExpectedResult);
    assert(IncScanAcc[WI] == ExpectedResult);
  }
  return 0;
}
