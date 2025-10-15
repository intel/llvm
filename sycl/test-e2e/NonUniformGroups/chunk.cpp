// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
// Test CPU AOT as well when possible.
// RUN: %if any-device-is-cpu && opencl-aot %{ %{run-aux} %clangxx -fsycl -fsycl-targets=spir64_x86_64 -o %t.x86.out %s %}
// RUN: %if cpu && opencl-aot %{ %{run} %t.x86.out %}
//
// REQUIRES: cpu || gpu
// REQUIRES: sg-32
// REQUIRES: aspect-ext_oneapi_chunk

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/chunk.hpp>

namespace syclex = sycl::ext::oneapi::experimental;

template <size_t PartitionSize> class SubgroupTestKernel;
template <size_t PartitionSize, size_t SubPartitionSize> class ChunkTestKernel;

template <size_t PartitionSize> void test() {
  sycl::queue Q;

  // Test for both the full sub-group size and a case with less work than a full
  // sub-group.
  for (size_t WGS : std::array<size_t, 2>{32, 16}) {
    if (WGS < PartitionSize)
      continue;

    sycl::buffer<bool, 1> MatchBuf{sycl::range{WGS}};
    sycl::buffer<bool, 1> LeaderBuf{sycl::range{WGS}};

    const auto NDR = sycl::nd_range<1>{WGS, WGS};

    std::cout << "Testing for work size " << WGS << " and partition size "
              << PartitionSize << std::endl;
    Q.submit([&](sycl::handler &CGH) {
      sycl::accessor MatchAcc{MatchBuf, CGH, sycl::write_only};
      sycl::accessor LeaderAcc{LeaderBuf, CGH, sycl::write_only};
      const auto KernelFunc =
          [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
            auto WI = item.get_global_id();
            auto SG = item.get_sub_group();
            auto SGS = SG.get_local_linear_range();

            auto ChunkGroup = syclex::chunked_partition<PartitionSize>(SG);

            bool Match = true;
            Match &= (ChunkGroup.get_group_id() == (WI / PartitionSize));
            Match &= (ChunkGroup.get_local_id() == (WI % PartitionSize));
            Match &= (ChunkGroup.get_group_range() == (SGS / PartitionSize));
            Match &= (ChunkGroup.get_local_range() == PartitionSize);
            MatchAcc[WI] = Match;
            LeaderAcc[WI] = ChunkGroup.leader();
          };
      CGH.parallel_for<SubgroupTestKernel<PartitionSize>>(NDR, KernelFunc);
    });

    {
      sycl::host_accessor MatchAcc{MatchBuf, sycl::read_write};
      sycl::host_accessor LeaderAcc{LeaderBuf, sycl::read_write};
      for (int WI = 0; WI < WGS; ++WI) {
        assert(MatchAcc[WI] == true);
        assert(LeaderAcc[WI] == ((WI % PartitionSize) == 0));
        MatchAcc[WI] = false;
        LeaderAcc[WI] = false;
      }
    }

    std::cout << "Testing for work size " << WGS << " and partition size "
              << PartitionSize << " and subpartition size " << PartitionSize
              << std::endl;
    Q.submit([&](sycl::handler &CGH) {
      sycl::accessor MatchAcc{MatchBuf, CGH, sycl::write_only};
      sycl::accessor LeaderAcc{LeaderBuf, CGH, sycl::write_only};
      const auto KernelFunc =
          [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
            auto WI = item.get_global_id();
            auto SG = item.get_sub_group();

            auto ParentChunkGroup =
                syclex::chunked_partition<PartitionSize>(SG);
            auto ChunkGroup =
                syclex::chunked_partition<PartitionSize>(ParentChunkGroup);

            bool Match = true;
            Match &= (ChunkGroup.get_group_id() == 0);
            Match &= (ChunkGroup.get_local_id() == (WI % PartitionSize));
            Match &= (ChunkGroup.get_group_range() == 1);
            Match &= (ChunkGroup.get_local_range() == PartitionSize);
            MatchAcc[WI] = Match;
            LeaderAcc[WI] = ChunkGroup.leader();
          };
      CGH.parallel_for<ChunkTestKernel<PartitionSize, PartitionSize>>(
          NDR, KernelFunc);
    });

    {
      sycl::host_accessor MatchAcc{MatchBuf, sycl::read_write};
      sycl::host_accessor LeaderAcc{LeaderBuf, sycl::read_write};
      for (int WI = 0; WI < WGS; ++WI) {
        assert(MatchAcc[WI] == true);
        assert(LeaderAcc[WI] == ((WI % PartitionSize) == 0));
        MatchAcc[WI] = false;
        LeaderAcc[WI] = false;
      }
    }

    constexpr size_t HalfPartitionSize = PartitionSize / 2;
    if constexpr (HalfPartitionSize != 0) {
      std::cout << "Testing for work size " << WGS << " and partition size "
                << PartitionSize << " and subpartition size "
                << HalfPartitionSize << std::endl;
      const auto NDR = sycl::nd_range<1>{WGS, WGS};
      Q.submit([&](sycl::handler &CGH) {
        sycl::accessor MatchAcc{MatchBuf, CGH, sycl::write_only};
        sycl::accessor LeaderAcc{LeaderBuf, CGH, sycl::write_only};
        const auto KernelFunc =
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
              auto WI = item.get_global_id();
              auto SG = item.get_sub_group();

              auto ParentChunkGroup =
                  syclex::chunked_partition<PartitionSize>(SG);
              auto ChunkGroup = syclex::chunked_partition<HalfPartitionSize>(
                  ParentChunkGroup);

              bool Match = true;
              Match &= (ChunkGroup.get_group_id() ==
                        (ParentChunkGroup.get_local_id() / HalfPartitionSize));
              Match &= (ChunkGroup.get_local_id() ==
                        (ParentChunkGroup.get_local_id() % HalfPartitionSize));
              Match &= (ChunkGroup.get_group_range() ==
                        (ParentChunkGroup.get_local_linear_range() /
                         HalfPartitionSize));
              Match &= (ChunkGroup.get_local_range() == HalfPartitionSize);
              MatchAcc[WI] = Match;
              LeaderAcc[WI] = ChunkGroup.leader();
            };
        CGH.parallel_for<ChunkTestKernel<PartitionSize, HalfPartitionSize>>(
            NDR, KernelFunc);
      });

      {
        sycl::host_accessor MatchAcc{MatchBuf, sycl::read_write};
        sycl::host_accessor LeaderAcc{LeaderBuf, sycl::read_write};
        for (int WI = 0; WI < WGS; ++WI) {
          assert(MatchAcc[WI] == true);
          assert(LeaderAcc[WI] == ((WI % HalfPartitionSize) == 0));
          MatchAcc[WI] = false;
          LeaderAcc[WI] = false;
        }
      }
    }
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
