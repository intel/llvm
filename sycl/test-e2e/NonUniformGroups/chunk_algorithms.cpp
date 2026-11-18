// RUN-IF: sg-8, %{build} -fsycl-device-code-split=per_kernel -DSG_SIZE=8 -o %t8.out && %{run} %t8.out
// RUN-IF: sg-16, %{build} -fsycl-device-code-split=per_kernel -DSG_SIZE=16 -o %t16.out && %{run} %t16.out
// RUN-IF: sg-32, %{build} -fsycl-device-code-split=per_kernel -DSG_SIZE=32 -o %t32.out && %{run} %t32.out

// Test CPU AOT as well when possible.
// RUN-IF: any-device-is-cpu && opencl-aot, %{run-aux} %clangxx -fsycl -fsycl-targets=spir64_x86_64 -fsycl-device-code-split=per_kernel -DSG_SIZE=8 -o %t8.x86.out %s
// RUN-IF: sg-8 && cpu && opencl-aot, %{run} %t8.x86.out
// RUN-IF: any-device-is-cpu && opencl-aot, %{run-aux} %clangxx -fsycl -fsycl-targets=spir64_x86_64 -fsycl-device-code-split=per_kernel -DSG_SIZE=16 -o %t16.x86.out %s
// RUN-IF: sg-16 && cpu && opencl-aot, %{run} %t16.x86.out
// RUN-IF: any-device-is-cpu && opencl-aot, %{run-aux} %clangxx -fsycl -fsycl-targets=spir64_x86_64 -fsycl-device-code-split=per_kernel -DSG_SIZE=32 -o %t32.x86.out %s
// RUN-IF: sg-32 && cpu && opencl-aot, %{run} %t32.x86.out

// REQUIRES: cpu || gpu
// REQUIRES: sg-8 || sg-16 || sg-32
// REQUIRES: aspect-ext_oneapi_chunk
// UNSUPPORTED: target-amd
// UNSUPPORTED-INTENDED: fixed_size_group aspect not available on amd

// XFAIL: target-native_cpu
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/20142

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/chunk.hpp>
#include <sycl/group_algorithm.hpp>
#include <sycl/group_barrier.hpp>

#include <cstdio>

static int Failures = 0;
#define CHECK(Cond)                                                            \
  do {                                                                         \
    if (!(Cond)) {                                                             \
      std::printf("FAIL PartitionSize=%zu WI=%d: %s\n", PartitionSize,         \
                  static_cast<int>(WI), #Cond);                                \
      ++Failures;                                                              \
    }                                                                          \
  } while (0)

namespace syclex = sycl::ext::oneapi::experimental;

template <size_t PartitionSize> class TestKernel;

template <size_t PartitionSize> void test() {
  sycl::queue Q;

  constexpr uint32_t SGSize = SG_SIZE;

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

          // Split into partitions of fixed size
          auto ChunkGroup = syclex::chunked_partition<PartitionSize>(SG);

          // Check all other members' writes are visible after a barrier.
          TmpAcc[WI] = 1;
          sycl::group_barrier(ChunkGroup);
          size_t Visible = 0;
          for (size_t Other = 0; Other < SGSize; ++Other) {
            if ((WI / PartitionSize) == (Other / PartitionSize)) {
              Visible += TmpAcc[Other];
            }
          }
          BarrierAcc[WI] = (Visible == PartitionSize);

          // Simple check of group algorithms.
          uint32_t OriginalLID = SG.get_local_linear_id();
          uint32_t LID = ChunkGroup.get_local_linear_id();

          uint32_t PartitionLeader =
              (OriginalLID / PartitionSize) * PartitionSize;
          uint32_t BroadcastResult =
              sycl::group_broadcast(ChunkGroup, OriginalLID, 0);
          BroadcastAcc[WI] = (BroadcastResult == PartitionLeader);

          bool AnyResult = sycl::any_of_group(ChunkGroup, (LID == 0));
          AnyAcc[WI] = (AnyResult == true);

          bool Predicate = ((OriginalLID / PartitionSize) % 2 == 0);
          bool AllResult = sycl::all_of_group(ChunkGroup, Predicate);
          if (Predicate) {
            AllAcc[WI] = (AllResult == true);
          } else {
            AllAcc[WI] = (AllResult == false);
          }

          bool NoneResult = sycl::none_of_group(ChunkGroup, Predicate);
          if (Predicate) {
            NoneAcc[WI] = (NoneResult == false);
          } else {
            NoneAcc[WI] = (NoneResult == true);
          }

          uint32_t ReduceResult =
              sycl::reduce_over_group(ChunkGroup, 1, sycl::plus<>());
          ReduceAcc[WI] = (ReduceResult == PartitionSize);

          uint32_t ExScanResult =
              sycl::exclusive_scan_over_group(ChunkGroup, 1, sycl::plus<>());
          ExScanAcc[WI] = (ExScanResult == LID);

          uint32_t IncScanResult =
              sycl::inclusive_scan_over_group(ChunkGroup, 1, sycl::plus<>());
          IncScanAcc[WI] = (IncScanResult == LID + 1);

          float ExScanFloatResult =
              sycl::exclusive_scan_over_group(ChunkGroup, 1.0f, sycl::plus<>());
          ExScanFloatAcc[WI] = (ExScanFloatResult == static_cast<float>(LID));

          float IncScanFloatResult =
              sycl::inclusive_scan_over_group(ChunkGroup, 1.0f, sycl::plus<>());
          IncScanFloatAcc[WI] =
              (IncScanFloatResult == static_cast<float>(LID + 1));

          constexpr uint32_t Init = 42;
          uint32_t ExScanInitResult = sycl::exclusive_scan_over_group(
              ChunkGroup, 1, Init, sycl::plus<>());
          ExScanInitAcc[WI] = (ExScanInitResult == Init + LID);

          uint32_t IncScanInitResult = sycl::inclusive_scan_over_group(
              ChunkGroup, 1, sycl::plus<>(), Init);
          IncScanInitAcc[WI] = (IncScanInitResult == Init + LID + 1);

          uint32_t ShiftLeftResult = sycl::shift_group_left(ChunkGroup, LID, 2);
          ShiftLeftAcc[WI] =
              (LID + 2 >= PartitionSize || ShiftLeftResult == LID + 2);

          uint32_t ShiftRightResult =
              sycl::shift_group_right(ChunkGroup, LID, 2);
          ShiftRightAcc[WI] = (LID < 2 || ShiftRightResult == LID - 2);

          uint32_t SelectResult = sycl::select_from_group(
              ChunkGroup, OriginalLID,
              (ChunkGroup.get_local_id() + 2) % PartitionSize);
          SelectAcc[WI] =
              SelectResult == OriginalLID - LID + ((LID + 2) % PartitionSize);

          uint32_t Mask = PartitionSize <= 2 ? 0 : 2;
          uint32_t PermuteXorResult =
              sycl::permute_group_by_xor(ChunkGroup, LID, Mask);
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
  sycl::host_accessor ExScanFloatAcc{ExScanFloatBuf, sycl::read_only};
  sycl::host_accessor IncScanFloatAcc{IncScanFloatBuf, sycl::read_only};
  sycl::host_accessor ExScanInitAcc{ExScanInitBuf, sycl::read_only};
  sycl::host_accessor IncScanInitAcc{IncScanInitBuf, sycl::read_only};
  sycl::host_accessor ShiftLeftAcc{ShiftLeftBuf, sycl::read_only};
  sycl::host_accessor ShiftRightAcc{ShiftRightBuf, sycl::read_only};
  sycl::host_accessor SelectAcc{SelectBuf, sycl::read_only};
  sycl::host_accessor PermuteXorAcc{PermuteXorBuf, sycl::read_only};
  for (int WI = 0; WI < SGSize; ++WI) {
    CHECK(BarrierAcc[WI] == true);
    CHECK(BroadcastAcc[WI] == true);
    CHECK(AnyAcc[WI] == true);
    CHECK(AllAcc[WI] == true);
    CHECK(NoneAcc[WI] == true);
    CHECK(ReduceAcc[WI] == true);
    CHECK(ExScanAcc[WI] == true);
    CHECK(IncScanAcc[WI] == true);
    CHECK(ExScanFloatAcc[WI] == true);
    CHECK(IncScanFloatAcc[WI] == true);
    CHECK(ExScanInitAcc[WI] == true);
    CHECK(IncScanInitAcc[WI] == true);
    CHECK(ShiftLeftAcc[WI] == true);
    CHECK(ShiftRightAcc[WI] == true);
    CHECK(SelectAcc[WI] == true);
    CHECK(PermuteXorAcc[WI] == true);
  }
}

int main() {
  test<1>();
  test<2>();
  test<4>();
  test<8>();

  if constexpr (SG_SIZE >= 16) {
    test<16>();
  }

  if constexpr (SG_SIZE >= 32) {
    test<32>();
  }

  std::printf("Total failures: %d\n", Failures);
  return Failures != 0;
}
