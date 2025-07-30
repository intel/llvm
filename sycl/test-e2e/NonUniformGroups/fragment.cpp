// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
// Test CPU AOT as well when possible.
// RUN: %if any-device-is-cpu && opencl-aot %{ %{run-aux} %clangxx -fsycl -fsycl-targets=spir64_x86_64 -o %t.x86.out %s %}
// RUN: %if cpu && opencl-aot %{ %{run} %t.x86.out %}
//
// REQUIRES: cpu || gpu
// REQUIRES: aspect-ext_oneapi_fragment

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/chunk.hpp>
#include <sycl/ext/oneapi/experimental/fragment.hpp>
#include <vector>
namespace syclex = sycl::ext::oneapi::experimental;

class SubgroupTestKernel;
class FragmentTestKernel;
class ChunkTestKernel;

int main() {
  sycl::queue Q;

  auto SGSizes = Q.get_device().get_info<sycl::info::device::sub_group_sizes>();
  if (std::find(SGSizes.begin(), SGSizes.end(), 32) == SGSizes.end()) {
    std::cout << "Test skipped due to missing support for sub-group size 32."
              << std::endl;
    return 0;
  }

  // Test for both the full sub-group size and a case with less work than a full
  // sub-group.
  for (size_t WGS : std::array<size_t, 2>{32, 16}) {
    std::cout << "Testing sub_group partition for work size " << WGS
              << std::endl;

    sycl::buffer<bool, 1> MatchBuf{sycl::range{WGS}};
    sycl::buffer<bool, 1> LeaderBuf{sycl::range{WGS}};

    const auto NDR = sycl::nd_range<1>{WGS, WGS};
    Q.submit([&](sycl::handler &CGH) {
      sycl::accessor MatchAcc{MatchBuf, CGH, sycl::write_only};
      sycl::accessor LeaderAcc{LeaderBuf, CGH, sycl::write_only};
      const auto KernelFunc =
          [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
            auto WI = item.get_global_id();
            auto SG = item.get_sub_group();

            // Split into odd and even work-items.
            bool Predicate = WI % 2 == 0;
            auto FragmentGroup = syclex::binary_partition(SG, Predicate);

            // Check function return values match Predicate.
            // NB: Test currently uses exactly one sub-group, but we use SG
            //     below in case this changes in future.
            bool Match = true;
            auto GroupID = (Predicate) ? 1 : 0;
            auto LocalID = SG.get_local_id() / 2;
            Match &= (FragmentGroup.get_group_id() == GroupID);
            Match &= (FragmentGroup.get_local_id() == LocalID);
            Match &= (FragmentGroup.get_group_range() == 2);
            Match &= (FragmentGroup.get_local_range() ==
                      SG.get_local_linear_range() / 2);
            MatchAcc[WI] = Match;
            LeaderAcc[WI] = FragmentGroup.leader();
          };
      CGH.parallel_for<SubgroupTestKernel>(NDR, KernelFunc);
    });

    sycl::host_accessor MatchAcc{MatchBuf, sycl::read_only};
    sycl::host_accessor LeaderAcc{LeaderBuf, sycl::read_only};
    for (int WI = 0; WI < WGS; ++WI) {
      assert(MatchAcc[WI] == true);
      assert(LeaderAcc[WI] == (WI < 2));
    }
  }

  // Test for fragment created from another fragment.
  {
    std::cout << "Testing for fragment from fragment" << std::endl;

    sycl::buffer<bool, 1> MatchBuf{sycl::range{32}};
    sycl::buffer<bool, 1> LeaderBuf{sycl::range{32}};

    const auto NDR = sycl::nd_range<1>{32, 32};
    Q.submit([&](sycl::handler &CGH) {
      sycl::accessor MatchAcc{MatchBuf, CGH, sycl::write_only};
      sycl::accessor LeaderAcc{LeaderBuf, CGH, sycl::write_only};
      const auto KernelFunc =
          [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
            auto WI = item.get_global_id();
            auto SG = item.get_sub_group();

            // Split into odd and even work-items.
            bool ParentPredicate = WI % 2 == 0;
            auto ParentFragmentGroup =
                syclex::binary_partition(SG, ParentPredicate);

            // Split parent fragment into odd and even participants.
            bool Predicate = ParentFragmentGroup.get_local_linear_id() % 2 == 0;
            auto FragmentGroup =
                syclex::binary_partition(ParentFragmentGroup, Predicate);

            // Check function return values match Predicate and ParentPredicate.
            bool Match = true;
            auto GroupID = Predicate ? 1 : 0;
            auto LocalID = ParentFragmentGroup.get_local_id() / 2;
            Match &= (FragmentGroup.get_group_id() == GroupID);
            Match &= (FragmentGroup.get_local_id() == LocalID);
            Match &= (FragmentGroup.get_group_range() == 2);
            Match &= (FragmentGroup.get_local_range() ==
                      ParentFragmentGroup.get_local_linear_range() / 2);
            MatchAcc[WI] = Match;
            LeaderAcc[WI] = FragmentGroup.leader();
          };
      CGH.parallel_for<FragmentTestKernel>(NDR, KernelFunc);
    });

    sycl::host_accessor MatchAcc{MatchBuf, sycl::read_only};
    sycl::host_accessor LeaderAcc{LeaderBuf, sycl::read_only};
    for (int WI = 0; WI < 32; ++WI) {
      assert(MatchAcc[WI] == true);
      assert(LeaderAcc[WI] == (WI < 4));
    }
  }

  // Test for fragment created from a chunk.
  if (Q.get_device().has(sycl::aspect::ext_oneapi_chunk)) {
    std::cout << "Testing for fragment from chunk" << std::endl;

    constexpr size_t ChunkSize = 8;

    sycl::buffer<bool, 1> MatchBuf{sycl::range{32}};
    sycl::buffer<bool, 1> LeaderBuf{sycl::range{32}};

    const auto NDR = sycl::nd_range<1>{32, 32};
    Q.submit([&](sycl::handler &CGH) {
      sycl::accessor MatchAcc{MatchBuf, CGH, sycl::write_only};
      sycl::accessor LeaderAcc{LeaderBuf, CGH, sycl::write_only};
      const auto KernelFunc =
          [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
            auto WI = item.get_global_id();
            auto SG = item.get_sub_group();

            // Split into chunks.
            auto ParentChunkGroup = syclex::chunked_partition<ChunkSize>(SG);

            // Split parent fragment into odd and even participants.
            bool Predicate = ParentChunkGroup.get_local_linear_id() % 2 == 0;
            auto FragmentGroup =
                syclex::binary_partition(ParentChunkGroup, Predicate);

            // Check function return values match Predicate.
            bool Match = true;
            auto GroupID = Predicate ? 1 : 0;
            auto LocalID = ParentChunkGroup.get_local_id() / 2;
            Match &= (FragmentGroup.get_group_id() == GroupID);
            Match &= (FragmentGroup.get_local_id() == LocalID);
            Match &= (FragmentGroup.get_group_range() == 2);
            Match &= (FragmentGroup.get_local_range() == ChunkSize / 2);
            MatchAcc[WI] = Match;
            LeaderAcc[WI] = FragmentGroup.leader();
          };
      CGH.parallel_for<ChunkTestKernel>(NDR, KernelFunc);
    });

    sycl::host_accessor MatchAcc{MatchBuf, sycl::read_only};
    sycl::host_accessor LeaderAcc{LeaderBuf, sycl::read_only};
    for (int WI = 0; WI < 32; ++WI) {
      assert(MatchAcc[WI] == true);
      assert(LeaderAcc[WI] == (WI % ChunkSize < 2));
    }
  }

  return 0;
}
