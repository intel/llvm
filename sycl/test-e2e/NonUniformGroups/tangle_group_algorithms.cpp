// RUN: %{build} -fno-sycl-early-optimizations -o %t.out
// RUN: %{run} %t.out
//
// REQUIRES: gpu
// REQUIRES: sg-32
// REQUIRES: aspect-ext_oneapi_tangle_group
// UNSUPPORTED: cuda || windows
// Tangle groups exhibit unpredictable behavior on Windows.
// The test is disabled while we investigate the root cause.

#include <sycl/sycl.hpp>
#include <vector>
namespace syclex = sycl::ext::oneapi::experimental;

class TestKernel;

constexpr uint32_t SGSize = 32;

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
    sycl::accessor ShiftLeftAcc{ShiftLeftBuf, CGH, sycl::write_only};
    sycl::accessor ShiftRightAcc{ShiftRightBuf, CGH, sycl::write_only};
    sycl::accessor SelectAcc{SelectBuf, CGH, sycl::write_only};
    sycl::accessor PermuteXorAcc{PermuteXorBuf, CGH, sycl::write_only};
    const auto KernelFunc =
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SGSize)]] {
          auto WI = item.get_global_id();
          auto SG = item.get_sub_group();

          auto BranchBody = [=](size_t WI, auto Tangle, size_t TangleLeader,
                                size_t TangleSize, auto IsMember) {
            // Check all other members' writes are visible after a barrier.
            TmpAcc[WI] = 1;
            sycl::group_barrier(Tangle);
            size_t Visible = 0;
            for (size_t Other = 0; Other < SGSize; ++Other) {
              if (IsMember(Other)) {
                Visible += TmpAcc[Other];
              }
            }
            BarrierAcc[WI] = (Visible == TangleSize);

            // Simple check of group algorithms.
            uint32_t OriginalLID = SG.get_local_linear_id();
            uint32_t LID = Tangle.get_local_linear_id();

            uint32_t BroadcastResult =
                sycl::group_broadcast(Tangle, OriginalLID, 0);
            BroadcastAcc[WI] = (BroadcastResult == TangleLeader);

            bool AnyResult = sycl::any_of_group(Tangle, (LID == 0));
            AnyAcc[WI] = (AnyResult == true);

            bool AllResult = sycl::all_of_group(Tangle, (LID < TangleSize));
            AllAcc[WI] = (AllResult == true);

            bool NoneResult = sycl::none_of_group(Tangle, (LID >= TangleSize));
            NoneAcc[WI] = (NoneResult == true);

            uint32_t ReduceResult =
                sycl::reduce_over_group(Tangle, 1, sycl::plus<>());
            ReduceAcc[WI] = (ReduceResult == TangleSize);

            uint32_t ExScanResult =
                sycl::exclusive_scan_over_group(Tangle, 1, sycl::plus<>());
            ExScanAcc[WI] = (ExScanResult == LID);

            uint32_t IncScanResult =
                sycl::inclusive_scan_over_group(Tangle, 1, sycl::plus<>());
            IncScanAcc[WI] = (IncScanResult == LID + 1);

            uint32_t ShiftLeftResult = sycl::shift_group_left(Tangle, LID, 2);
            ShiftLeftAcc[WI] =
                (LID + 2 >= TangleSize || ShiftLeftResult == LID + 2);

            uint32_t ShiftRightResult = sycl::shift_group_right(Tangle, LID, 2);
            ShiftRightAcc[WI] = (LID + TangleSize - 2 <= TangleSize ||
                                 ShiftRightResult == LID - 2);

            uint32_t SelectResult = sycl::select_from_group(
                Tangle, LID, (Tangle.get_local_id() + 2) % TangleSize);
            SelectAcc[WI] = (SelectResult == (LID + 2) % TangleSize);

            uint32_t PermuteXorResult =
                sycl::permute_group_by_xor(Tangle, LID, 2);
            PermuteXorAcc[WI] = (PermuteXorResult == (LID ^ 2));
          };

          // Split into three groups of different sizes, using control flow
          // Body of each branch is deliberately duplicated
          if (WI < 4) {
            auto Tangle = syclex::get_tangle_group(SG);
            size_t TangleLeader = 0;
            size_t TangleSize = 4;
            auto IsMember = [](size_t Other) { return (Other < 4); };
            BranchBody(WI, Tangle, TangleLeader, TangleSize, IsMember);
          } else if (WI < 24) {
            auto Tangle = syclex::get_tangle_group(SG);
            size_t TangleLeader = 4;
            size_t TangleSize = 20;
            auto IsMember = [](size_t Other) {
              return (Other >= 4 and Other < 24);
            };
            BranchBody(WI, Tangle, TangleLeader, TangleSize, IsMember);
          } else /* if WI < 32) */ {
            auto Tangle = syclex::get_tangle_group(SG);
            size_t TangleLeader = 24;
            size_t TangleSize = 8;
            auto IsMember = [](size_t Other) {
              return (Other >= 24 and Other < 32);
            };
            BranchBody(WI, Tangle, TangleLeader, TangleSize, IsMember);
          };
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
  sycl::host_accessor ShiftLeftAcc{ShiftLeftBuf, sycl::read_only};
  sycl::host_accessor ShiftRightAcc{ShiftRightBuf, sycl::read_only};
  sycl::host_accessor SelectAcc{SelectBuf, sycl::read_only};
  sycl::host_accessor PermuteXorAcc{PermuteXorBuf, sycl::read_only};
  for (int WI = 0; WI < SGSize; ++WI) {
    assert(BarrierAcc[WI] == true);
    assert(BroadcastAcc[WI] == true);
    assert(AnyAcc[WI] == true);
    assert(AllAcc[WI] == true);
    assert(NoneAcc[WI] == true);
    assert(ReduceAcc[WI] == true);
    assert(ExScanAcc[WI] == true);
    assert(IncScanAcc[WI] == true);
    assert(ShiftLeftAcc[WI] == true);
    assert(ShiftRightAcc[WI] == true);
    assert(SelectAcc[WI] == true);
    assert(PermuteXorAcc[WI] == true);
  }
  return 0;
}
