// Regression test for https://github.com/intel/llvm/issues/23204
//
// group_ballot() and this_work_item::get_opportunistic_group() wrap a
// convergent SPIR-V ballot builtin whose result depends on the set of
// work-items active in the block that executes it. If the wrapper is not
// inlined into the (potentially divergent) calling block -- as happens at -O0,
// where nothing is inlined -- the builtin ends up in a separate function where
// the backend treats every work-item as active, and returns a mask/range that
// wrongly includes work-items that already returned from the kernel.
//
// Compile at -O0 specifically to pin down the inlining-dependent behaviour.
// RUN: %{build} -O0 -o %t.out
// RUN: %{run} %t.out
//
// REQUIRES: cpu || gpu
// REQUIRES: aspect-ext_oneapi_fragment

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/fragment.hpp>
#include <sycl/ext/oneapi/sub_group_mask.hpp>
#include <sycl/usm.hpp>
#include <vector>

namespace syclex = sycl::ext::oneapi::experimental;

// Run a kernel over 2 sub-groups of size SGSize. Work-items with a sub-group
// local id >= SGSize / 2 return early; the rest call group_ballot(sg, true) and
// get_opportunistic_group(). The expected ballot mask has the lower SGSize / 2
// bits set, and the expected opportunistic group size is SGSize / 2.
template <size_t SGSize> bool run(sycl::queue &Q, bool ConvergedFirst) {
  constexpr size_t Groups = 2;
  const size_t N = SGSize * Groups;
  const std::uint32_t ExpectedMask = (std::uint32_t{1} << (SGSize / 2)) - 1u;
  const std::uint32_t ExpectedCount = SGSize / 2;

  auto *Mask = sycl::malloc_device<std::uint32_t>(N, Q);
  auto *Count = sycl::malloc_device<std::uint32_t>(N, Q);
  Q.fill(Mask, std::uint32_t{0xdeadbeef}, N).wait();
  Q.fill(Count, std::uint32_t{0}, N).wait();

  Q.parallel_for(
       sycl::nd_range<1>{N, SGSize},
       [=](sycl::nd_item<1> Item) [[sycl::reqd_sub_group_size(SGSize)]] {
         auto SG = Item.get_sub_group();
         const size_t I = Item.get_global_linear_id();
         const size_t Lid = SG.get_local_linear_id();

         // Optionally perform a group operation in converged control flow
         // before diverging.
         if (ConvergedFirst) {
           [[maybe_unused]] auto All =
               sycl::ext::oneapi::group_ballot(SG, true);
         }

         // Half of the work-items leave the kernel.
         if (Lid >= SGSize / 2)
           return;

         auto M = sycl::ext::oneapi::group_ballot(SG, true);
         std::uint32_t Bits = 0;
         M.extract_bits(Bits);
         Mask[I] = Bits;

         auto OG = syclex::this_work_item::get_opportunistic_group();
         Count[I] = static_cast<std::uint32_t>(OG.get_local_linear_range());
       })
      .wait();

  std::vector<std::uint32_t> Masks(N), Counts(N);
  Q.memcpy(Masks.data(), Mask, N * sizeof(std::uint32_t)).wait();
  Q.memcpy(Counts.data(), Count, N * sizeof(std::uint32_t)).wait();
  sycl::free(Mask, Q);
  sycl::free(Count, Q);

  bool Ok = true;
  for (size_t I = 0; I < N; ++I) {
    if (I % SGSize >= SGSize / 2)
      continue;
    bool Good = Masks[I] == ExpectedMask && Counts[I] == ExpectedCount;
    Ok = Ok && Good;
    if (!Good)
      std::cout << "  FAIL item " << I << ": mask 0x" << std::hex << Masks[I]
                << std::dec << " (expected 0x" << std::hex << ExpectedMask
                << std::dec << "), opportunistic group size " << Counts[I]
                << " (expected " << ExpectedCount << ")\n";
  }
  return Ok;
}

int main() {
  sycl::queue Q;
  auto SGSizes = Q.get_device().get_info<sycl::info::device::sub_group_sizes>();

  bool Ok = true;
  for (bool ConvergedFirst : {false, true}) {
    if (std::find(SGSizes.begin(), SGSizes.end(), 16) != SGSizes.end())
      Ok = run<16>(Q, ConvergedFirst) && Ok;
    if (std::find(SGSizes.begin(), SGSizes.end(), 32) != SGSizes.end())
      Ok = run<32>(Q, ConvergedFirst) && Ok;
  }

  assert(Ok && "returned work-items must not be part of the ballot / "
               "opportunistic group");
  return Ok ? 0 : 1;
}
