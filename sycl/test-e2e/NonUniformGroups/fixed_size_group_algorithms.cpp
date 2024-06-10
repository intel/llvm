// RUN: %{build} -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %{run} %t.out
//
// REQUIRES: gpu
// REQUIRES: sg-32
// REQUIRES: aspect-ext_oneapi_fixed_size_group

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/fixed_size_group.hpp>
#include <sycl/group_algorithm.hpp>
#include <sycl/group_barrier.hpp>
#include <vector>
namespace syclex = sycl::ext::oneapi::experimental;

template <size_t PartitionSize> class TestKernel;

template <size_t PartitionSize> void test() {
  sycl::queue Q;

  constexpr uint32_t SGSize = 32;

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

          // Split into partitions of fixed size
          auto Partition = syclex::get_fixed_size_group<PartitionSize>(SG);

          // Check all other members' writes are visible after a barrier.
          TmpAcc[WI] = 1;
          sycl::group_barrier(Partition);
          size_t Visible = 0;
          for (size_t Other = 0; Other < SGSize; ++Other) {
            if ((WI / PartitionSize) == (Other / PartitionSize)) {
              Visible += TmpAcc[Other];
            }
          }
          BarrierAcc[WI] = (Visible == PartitionSize);

          // Simple check of group algorithms.
          uint32_t OriginalLID = SG.get_local_linear_id();
          uint32_t LID = Partition.get_local_linear_id();

          uint32_t PartitionLeader =
              (OriginalLID / PartitionSize) * PartitionSize;
          uint32_t BroadcastResult =
              sycl::group_broadcast(Partition, OriginalLID, 0);
          BroadcastAcc[WI] = (BroadcastResult == PartitionLeader);

          bool AnyResult = sycl::any_of_group(Partition, (LID == 0));
          AnyAcc[WI] = (AnyResult == true);

          bool Predicate = ((OriginalLID / PartitionSize) % 2 == 0);
          bool AllResult = sycl::all_of_group(Partition, Predicate);
          if (Predicate) {
            AllAcc[WI] = (AllResult == true);
          } else {
            AllAcc[WI] = (AllResult == false);
          }

          bool NoneResult = sycl::none_of_group(Partition, Predicate);
          if (Predicate) {
            NoneAcc[WI] = (NoneResult == false);
          } else {
            NoneAcc[WI] = (NoneResult == true);
          }

          uint32_t ReduceResult =
              sycl::reduce_over_group(Partition, 1, sycl::plus<>());
          ReduceAcc[WI] = (ReduceResult == PartitionSize);

          uint32_t ExScanResult =
              sycl::exclusive_scan_over_group(Partition, 1, sycl::plus<>());
          ExScanAcc[WI] = (ExScanResult == LID);

          uint32_t IncScanResult =
              sycl::inclusive_scan_over_group(Partition, 1, sycl::plus<>());
          IncScanAcc[WI] = (IncScanResult == LID + 1);

          uint32_t ShiftLeftResult = sycl::shift_group_left(Partition, LID, 2);
          ShiftLeftAcc[WI] =
              (LID + 2 >= PartitionSize || ShiftLeftResult == LID + 2);

          uint32_t ShiftRightResult =
              sycl::shift_group_right(Partition, LID, 2);
          ShiftRightAcc[WI] = (LID < 2 || ShiftRightResult == LID - 2);

          uint32_t SelectResult = sycl::select_from_group(
              Partition, OriginalLID,
              (Partition.get_local_id() + 2) % PartitionSize);
          SelectAcc[WI] =
              SelectResult == OriginalLID - LID + ((LID + 2) % PartitionSize);

          uint32_t Mask = PartitionSize <= 2 ? 0 : 2;
          uint32_t PermuteXorResult =
              sycl::permute_group_by_xor(Partition, LID, Mask);
          PermuteXorAcc[WI] = (PermuteXorResult == (LID ^ Mask));
        };
    CGH.parallel_for<TestKernel<PartitionSize>>(NDR, KernelFunc);
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
}

int main() {
  test<1>();
  test<2>();
  test<4>();
  test<8>();
  test<16>();
  test<32>();
  return 0;
}
