// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
//
// UNSUPPORTED: cpu || cuda || hip

#include <sycl/sycl.hpp>
#include <vector>
namespace syclex = sycl::ext::oneapi::experimental;

class TestKernel;

int main() {
  sycl::queue Q;

  auto SGSizes = Q.get_device().get_info<sycl::info::device::sub_group_sizes>();
  if (std::find(SGSizes.begin(), SGSizes.end(), 32) == SGSizes.end()) {
    std::cout << "Test skipped due to missing support for sub-group size 32."
              << std::endl;
    return 0;
  }

  sycl::buffer<size_t, 1> TmpBuf{sycl::range{32}};
  sycl::buffer<bool, 1> BarrierBuf{sycl::range{32}};
  sycl::buffer<bool, 1> BroadcastBuf{sycl::range{32}};
  sycl::buffer<bool, 1> AnyBuf{sycl::range{32}};
  sycl::buffer<bool, 1> AllBuf{sycl::range{32}};
  sycl::buffer<bool, 1> NoneBuf{sycl::range{32}};
  sycl::buffer<bool, 1> ReduceBuf{sycl::range{32}};
  sycl::buffer<bool, 1> ExScanBuf{sycl::range{32}};
  sycl::buffer<bool, 1> IncScanBuf{sycl::range{32}};

  const auto NDR = sycl::nd_range<1>{32, 32};
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
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
          auto WI = item.get_global_id();
          auto SG = item.get_sub_group();

          // Split into odd and even work-items
          bool Predicate = WI % 2 == 0;
          auto BallotGroup = syclex::get_ballot_group(SG, Predicate);

          // Check all other members' writes are visible after a barrier
          TmpAcc[WI] = 1;
          sycl::group_barrier(BallotGroup);
          size_t Visible = 0;
          for (size_t Other = 0; Other < 32; ++Other) {
            if (WI % 2 == Other % 2) {
              Visible += TmpAcc[Other];
            }
          }
          BarrierAcc[WI] = Visible;

          // Simple check of group algorithms
          uint32_t OriginalLID = SG.get_local_linear_id();
          uint32_t LID = BallotGroup.get_local_linear_id();

          uint32_t BroadcastResult =
              sycl::group_broadcast(BallotGroup, OriginalLID, 0);
          if (Predicate) {
            BroadcastAcc[WI] = (BroadcastResult == 0);
          } else {
            BroadcastAcc[WI] = (BroadcastResult == 1);
          }

          bool AnyResult = sycl::any_of_group(BallotGroup, Predicate);
          if (Predicate) {
            AnyAcc[WI] = (AnyResult == true);
          } else {
            AnyAcc[WI] = (AnyResult == false);
          }

          bool AllResult = sycl::all_of_group(BallotGroup, Predicate);
          if (Predicate) {
            AllAcc[WI] = (AllResult == true);
          } else {
            AllAcc[WI] = (AllResult == false);
          }

          bool NoneResult = sycl::none_of_group(BallotGroup, Predicate);
          if (Predicate) {
            NoneAcc[WI] = (NoneResult == false);
          } else {
            NoneAcc[WI] = (NoneResult == true);
          }

          uint32_t ReduceResult =
              sycl::reduce_over_group(BallotGroup, 1, sycl::plus<>());
          ReduceAcc[WI] =
              (ReduceResult == BallotGroup.get_local_linear_range());

          uint32_t ExScanResult =
              sycl::exclusive_scan_over_group(BallotGroup, 1, sycl::plus<>());
          ExScanAcc[WI] = (ExScanResult == LID);

          uint32_t IncScanResult =
              sycl::inclusive_scan_over_group(BallotGroup, 1, sycl::plus<>());
          IncScanAcc[WI] = (IncScanResult == LID + 1);
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
  for (int WI = 0; WI < 32; ++WI) {
    assert(BarrierAcc[WI] == true);
    assert(BroadcastAcc[WI] == true);
    assert(AnyAcc[WI] == true);
    assert(AllAcc[WI] == true);
    assert(NoneAcc[WI] == true);
    assert(ReduceAcc[WI] == true);
    assert(ExScanAcc[WI] == true);
    assert(IncScanAcc[WI] == true);
  }
  return 0;
}
