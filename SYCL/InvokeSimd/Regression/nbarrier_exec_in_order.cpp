// TODO: enable on Windows once driver is ready
// NOTE: named barrier supported only since PVC
// REQUIRES: gpu-intel-pvc && linux
// UNSUPPORTED: cuda || hip
//
// TODO: enable when Jira issue resolved, currently fail with VISALTO enable
// XFAIL: gpu-intel-pvc
//
// RUN: %clangxx -fsycl -fno-sycl-device-code-split-esimd -Xclang -fsycl-allow-func-ptr %s -o %t.out
// RUN: env IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %GPU_RUN_PLACEHOLDER %t.out
//
// VISALTO enable run
// RUN: env IGC_VISALTO=63 IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %GPU_RUN_PLACEHOLDER %t.out

/*
 * Test checks basic support for named barriers in invoke_simd context.
 * Threads are executed in ascending order of their local ID and each thread
 * stores data to addresses that partially overlap with addresses used by
 * previous thread.
 */

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/oneapi/experimental/invoke_simd.hpp>
#include <sycl/sycl.hpp>

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>
#include <sycl/ext/oneapi/experimental/invoke_simd.hpp>
#include <sycl/sycl.hpp>

#include <functional>
#include <iostream>
#include <type_traits>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental;
namespace esimd = sycl::ext::intel::esimd;
namespace experimental_esimd = sycl::ext::intel::experimental::esimd;

constexpr int VL = 16;
template <int case_num> class KernelID;

template <unsigned Threads, unsigned Size, bool UseSLM>
ESIMD_INLINE void ESIMD_CALLEE_nbarrier(int localID,
                                        int *o) SYCL_ESIMD_FUNCTION {
  // Threads - 1 named barriers required
  // but id 0 reserved for unnamed
  experimental_esimd::named_barrier_init<Threads>();

  int flag = 0; // producer-consumer mode
  int producers = 2;
  int consumers = 2;

  // overlaping offsets
  unsigned int off = VL * localID / 2;
  unsigned int off_slm = off * sizeof(int);
  esimd::simd<int, VL> val(localID);

  if constexpr (UseSLM) {
    /* TODO: SLM has to be allocated of outside invoke_simd, but propper
     * intarface is not yet ready. Current test implementation in this regard
     * is a subject to future changes.
     */
    esimd::slm_init(Size * sizeof(int));
    esimd::simd<int, VL> zero(0);
    experimental_esimd::lsc_slm_block_store<int, VL>(2 * off_slm, zero);
  }
  esimd::barrier();

  // Threads are executed in ascending order of their local ID and
  // each thread stores data to addresses that partially overlap with
  // addresses used by previous thread.

  // localID == 0 skips this branch and goes straight to lsc_surf_store
  // localID == 1 signals barrier 1
  // localID == 2 signals barrier 2
  // localID == 3 signals barrier 3
  // and so on
  if (localID > 0) {
    int barrier_id = localID;
    __ESIMD_ENS::named_barrier_signal(barrier_id, flag, producers, consumers);
    __ESIMD_ENS::named_barrier_wait(barrier_id);
  }

  if constexpr (UseSLM)
    experimental_esimd::lsc_slm_block_store<int, VL>(off_slm, val);
  else
    experimental_esimd::lsc_block_store<int, VL>(o + off, val);

  experimental_esimd::lsc_fence();

  // localID == 0 arrives here first and signals barrier 1
  // localID == 1 arrives here next and signals barrier 2
  // localID == 2 arrives here next and signals barrier 3
  // and so on, but last thread skipped this block
  if (localID < Threads - 1) {
    int barrier_id = localID + 1;
    __ESIMD_ENS::named_barrier_signal(barrier_id, flag, producers, consumers);
    __ESIMD_ENS::named_barrier_wait(barrier_id);
  }

  esimd::barrier();
  if constexpr (UseSLM) {
    auto res = experimental_esimd::lsc_slm_block_load<int, VL>(2 * off_slm);
    experimental_esimd::lsc_block_store<int, VL>(o + 2 * off, res);
  }
}

template <unsigned Threads, unsigned Size, bool UseSLM>
[[intel::device_indirectly_callable]] SYCL_EXTERNAL void __regcall SIMD_CALLEE_nbarrier(
    int localID, int *o) SYCL_ESIMD_FUNCTION {
  ESIMD_CALLEE_nbarrier<Threads, Size, UseSLM>(localID, o);
}

template <int case_num, unsigned Threads, bool UseSLM, class QueueTY>
bool test(QueueTY q) {
  // number of ints stored by each thread
  constexpr unsigned Size = VL * Threads;

  static_assert(Threads % 2 == 0, "Number of threads must be even");
  std::cout << "Case #" << case_num << "\n\tTreads: " << Threads
            << "\n\tInts per thread: " << VL
            << "\n\tMemory: " << (UseSLM ? "local\n" : "global\n");

  auto *out = malloc_shared<int>(Size, q);
  for (int i = 0; i < Size; i++) {
    out[i] = -1;
  }

  try {
    // workgroups
    sycl::range<1> GlobalRange{Size};
    // threads in each group
    sycl::range<1> LocalRange{Threads};
    sycl::nd_range<1> Range{GlobalRange * LocalRange, LocalRange};

    auto e = q.submit([&](handler &cgh) {
      cgh.parallel_for<KernelID<case_num>>(
          nd_range<1>(GlobalRange, LocalRange),
          // This test requires an explicit specification of the subgroup size
          [=](nd_item<1> item) [[intel::reqd_sub_group_size(VL)]] {
            sycl::group<1> g = item.get_group();
            sycl::sub_group sg = item.get_sub_group();
            uint32_t i =
                sg.get_group_linear_id() * VL + g.get_linear_id() * Threads;
            uint32_t wi_id = i + sg.get_local_id();
            // Thread local ID in ESIMD context
            int localID = wi_id / VL;
            invoke_simd(sg, SIMD_CALLEE_nbarrier<Threads, Size, UseSLM>,
                        uniform{localID}, uniform{out});
          });
    });
    e.wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    free(out, q);
    return false;
  }

  bool passed = true;
  for (int i = 0; i < Size; i++) {
    int etalon = i * 2 * Threads / Size;
    if (etalon == Threads) // last stored chunk
      etalon -= 1;
    if (etalon > Threads) // excessive part of surface
      etalon = 0;
    if (out[i] != etalon) {
      passed = false;
      std::cout << "out[" << i << "]=" << out[i] << " vs " << etalon << "\n";
    }
  }

  free(out, q);

  std::cout << (passed ? " Passed\n" : " FAILED\n");
  return passed;
}

int main() {
  auto q = queue{gpu_selector_v};
  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<sycl::info::device::name>()
            << "\n";

  bool passed = true;

  passed &= test<1, 2, false>(q);
  passed &= test<2, 4, false>(q);
  passed &= test<3, 8, false>(q);
  passed &= test<4, 16, false>(q);

  /* TODO: Enable the sub-tests with UseSLM=true after the issue with SLM
   * initialization is fixed.
   *
     passed &= test<5, 2, true>(q);
     passed &= test<6, 4, true>(q);
     passed &= test<7, 8, true>(q);
     passed &= test<8, 16, true>(q);
   */

  return passed ? 0 : 1;
}
