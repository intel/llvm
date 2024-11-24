// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
// RUN: %if any-device-is-cpu && opencl-aot %{ %clangxx -fsycl -fsycl-targets=spir64_x86_64 -o %t.x86.out %s %}
// RUN: %if cpu %{ %{run} %t.x86.out %}
//
// REQUIRES: cpu || gpu
// UNSUPPORTED: hip
// REQUIRES: sg-32
// REQUIRES: aspect-ext_oneapi_chunk

#include <vector>

// #ifdef __SYCL_DEVICE_ONLY__
//[[__sycl_detail__::__uses_aspects__(sycl::aspect::ext_oneapi_chunk)]]

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/chunk.hpp>

namespace syclex = sycl::ext::oneapi::experimental;

template <size_t ChunkSize> class TestKernel;

template <size_t ChunkSize> void test() {
  sycl::queue Q;

  // Test for both the full sub-group size and a case with less work than a full
  // sub-group.
  for (size_t WGS : std::array<size_t, 2>{32, 16}) {
    if (WGS < ChunkSize)
      continue;

    std::cout << "Testing for work size " << WGS << " and partition size "
              << ChunkSize << std::endl;

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
            auto SGS = SG.get_local_linear_range();

            auto Partition = syclex::chunked_partition<ChunkSize>(SG);

            bool Match = true;
            Match &= (Partition.get_group_id() == (WI / ChunkSize));
            Match &= (Partition.get_local_id() == (WI % ChunkSize));
            Match &= (Partition.get_group_range() == (SGS / ChunkSize));
            Match &= (Partition.get_local_range() == ChunkSize);
            MatchAcc[WI] = Match;
            LeaderAcc[WI] = Partition.leader();
          };
      CGH.parallel_for<TestKernel<ChunkSize>>(NDR, KernelFunc);
    });

    sycl::host_accessor MatchAcc{MatchBuf, sycl::read_only};
    sycl::host_accessor LeaderAcc{LeaderBuf, sycl::read_only};
    for (int WI = 0; WI < WGS; ++WI) {
      assert(MatchAcc[WI] == true);
      assert(LeaderAcc[WI] == ((WI % ChunkSize) == 0));
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

// #endif
