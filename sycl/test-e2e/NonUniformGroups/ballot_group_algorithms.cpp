// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
// REQUIRES: gpu
// REQUIRES: sg-32
// REQUIRES: aspect-ext_oneapi_ballot_group

#include <sycl/sycl.hpp>
#include <vector>
namespace syclex = sycl::ext::oneapi::experimental;

class TestKernel;

int main() {
  sycl::queue Q;

  sycl::buffer<size_t, 1> TmpBuf{sycl::range{32}};
  sycl::buffer<bool, 1> BarrierBuf{sycl::range{32}};
  sycl::buffer<bool, 1> BroadcastBuf{sycl::range{32}};
  sycl::buffer<bool, 1> AnyBuf{sycl::range{32}};
  sycl::buffer<bool, 1> AllBuf{sycl::range{32}};
  sycl::buffer<bool, 1> NoneBuf{sycl::range{32}};
  sycl::buffer<bool, 1> ReduceBuf{sycl::range{32}};
  sycl::buffer<bool, 1> ExScanBuf{sycl::range{32}};
  sycl::buffer<bool, 1> IncScanBuf{sycl::range{32}};
  sycl::buffer<bool, 1> ShiftLeftBuf{sycl::range{32}};
  sycl::buffer<bool, 1> ShiftRightBuf{sycl::range{32}};
  sycl::buffer<bool, 1> SelectBuf{sycl::range{32}};
  sycl::buffer<bool, 1> PermuteXorBuf{sycl::range{32}};

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
    sycl::accessor ShiftLeftAcc{ShiftLeftBuf, CGH, sycl::write_only};
    sycl::accessor ShiftRightAcc{ShiftRightBuf, CGH, sycl::write_only};
    sycl::accessor SelectAcc{SelectBuf, CGH, sycl::write_only};
    sycl::accessor PermuteXorAcc{PermuteXorBuf, CGH, sycl::write_only};
    const auto KernelFunc =
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
          auto WI = item.get_global_id();
          auto SG = item.get_sub_group();

          // Split into odd and even work-items.
          bool Predicate = WI % 2 == 0;
          auto BallotGroup = syclex::get_ballot_group(SG, Predicate);
          uint32_t BallotGroupSize = BallotGroup.get_local_linear_range();

          // Check all other members' writes are visible after a barrier.
          TmpAcc[WI] = 1;
          sycl::group_barrier(BallotGroup);
          size_t Visible = 0;
          for (size_t Other = 0; Other < 32; ++Other) {
            if (WI % 2 == Other % 2) {
              Visible += TmpAcc[Other];
            }
          }
          BarrierAcc[WI] = (Visible == BallotGroup.get_local_linear_range());

          // Simple check of group algorithms.
          uint32_t OriginalLID = SG.get_local_linear_id();
          uint32_t LID = BallotGroup.get_local_linear_id();

          uint32_t BroadcastResult =
              sycl::group_broadcast(BallotGroup, OriginalLID, 0);
          if (Predicate) {
            BroadcastAcc[WI] = (BroadcastResult == 0);
          } else {
            BroadcastAcc[WI] = (BroadcastResult == 1);
          }

          bool AnyResult = sycl::any_of_group(BallotGroup, (LID == 0));
          AnyAcc[WI] = (AnyResult == true);

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
          ReduceAcc[WI] = (ReduceResult == BallotGroupSize);

          uint32_t ExScanResult =
              sycl::exclusive_scan_over_group(BallotGroup, 1, sycl::plus<>());
          ExScanAcc[WI] = (ExScanResult == LID);

          uint32_t IncScanResult =
              sycl::inclusive_scan_over_group(BallotGroup, 1, sycl::plus<>());
          IncScanAcc[WI] = (IncScanResult == LID + 1);

          uint32_t ShiftLeftResult =
              sycl::shift_group_left(BallotGroup, LID, 2);
          ShiftLeftAcc[WI] =
              (LID + 2 >= BallotGroupSize || ShiftLeftResult == LID + 2);

          uint32_t ShiftRightResult =
              sycl::shift_group_right(BallotGroup, LID, 2);
          ShiftRightAcc[WI] = (LID < 2 || ShiftRightResult == LID - 2);

          uint32_t SelectResult = sycl::select_from_group(
              BallotGroup, LID,
              (BallotGroup.get_local_id() + 2) % BallotGroupSize);
          SelectAcc[WI] = (SelectResult == (LID + 2) % BallotGroupSize);

          uint32_t PermuteXorResult =
              sycl::permute_group_by_xor(BallotGroup, LID, 2);
          PermuteXorAcc[WI] = (PermuteXorResult == (LID ^ 2));
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
  for (int WI = 0; WI < 32; ++WI) {
    assert(BarrierAcc[WI] == true);
    assert(BroadcastAcc[WI] == true);
    assert(AnyAcc[WI] == true);
    assert(AllAcc[WI] == true);
    assert(NoneAcc[WI] == true);
    assert(ReduceAcc[WI] == true);
    assert(ExScanAcc[WI] == true);
    assert(IncScanAcc[WI] == true);
    // TODO: Enable for CUDA devices when issue with shuffles have been
    // addressed.
    if (Q.get_backend() != sycl::backend::ext_oneapi_cuda) {
      assert(ShiftLeftAcc[WI] == true);
      assert(ShiftRightAcc[WI] == true);
      assert(SelectAcc[WI] == true);
      assert(PermuteXorAcc[WI] == true);
    }
  }
  return 0;
}
