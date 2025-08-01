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
#include <sycl/ext/oneapi/experimental/fragment.hpp>
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

  // Test for both the full sub-group size and a case with less work than a full
  // sub-group.
  for (size_t WGS : std::array<size_t, 2>{32, 16}) {
    std::cout << "Testing for work size " << WGS << std::endl;

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

            // opportunistic has unpredictable runtime behavior due to:
            // - dynamic formation from work-items
            // - hardware scheduling
            // - runtime factors that affect grouping:
            // - platform differences (SPIR-V vs. NVIDIA - group_ballot vs
            // activemask) groups
            // so need to check some values may change from run to run
            // for expected ranges and consistent with other groups.
            if (item.get_global_id() % 2 == 0) {
              auto OpportunisticGroup =
                  syclex::this_work_item::get_opportunistic_group();

              bool Match = true;
              Match &= (OpportunisticGroup.get_group_id() == 0);
              Match &= (OpportunisticGroup.get_local_id() <
                        OpportunisticGroup.get_local_range());
              Match &= (OpportunisticGroup.get_group_range() == 1);
              Match &= (OpportunisticGroup.get_local_linear_range() <=
                        SG.get_local_linear_range());
              MatchAcc[WI] = Match;
              LeaderAcc[WI] = OpportunisticGroup.leader();
            }
          };
      CGH.parallel_for<TestKernel>(NDR, KernelFunc);
    });

    sycl::host_accessor MatchAcc{MatchBuf, sycl::read_only};
    sycl::host_accessor LeaderAcc{LeaderBuf, sycl::read_only};
    uint32_t NumLeaders = 0;
    for (size_t WI{}; WI != WGS; ++WI) {
      if (WI % 2 == 0) {
        assert(MatchAcc[WI] == true);
        if (LeaderAcc[WI])
          NumLeaders++;
      }
    }
    assert(NumLeaders > 0);
  }
  return 0;
}
