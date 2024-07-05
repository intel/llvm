// NOTE: named barrier supported only since PVC
// REQUIRES: gpu-intel-pvc
//
// RUN: %{build} -fno-sycl-device-code-split-esimd -Xclang -fsycl-allow-func-ptr -o %t.out
// RUN: env IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %{run} %t.out
//
// vISA LTO run
// RUN: env IGC_VISALTO=63 IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %{run} %t.out

/*
 * Test checks basic support for named barriers in invoke_simd context.
 * Threads are executed in ascending order of their local ID and each thread
 * stores data to addresses that partially overlap with addresses used by
 * previous thread.
 */

#include <sycl/detail/core.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>
#include <sycl/ext/oneapi/experimental/invoke_simd.hpp>
#include <sycl/usm.hpp>

#include <functional>
#include <iostream>
#include <type_traits>

// TODO: When gpu driver can pass/accept accessor by value,
// the work-around undef #ifdef USE_ACC_PTR should be removed.
#define USE_ACC_PTR

using namespace sycl;
using namespace sycl::ext::oneapi::experimental;
namespace esimd = sycl::ext::intel::esimd;
namespace experimental_esimd = sycl::ext::intel::experimental::esimd;

constexpr int VL = 16;
template <int CaseNum> class KernelID;

template <unsigned Threads, unsigned Size, bool UseSLM>
ESIMD_INLINE void ESIMD_CALLEE_nbarrier(local_accessor<int, 1> local_acc,
                                        int local_id,
                                        int *o) SYCL_ESIMD_FUNCTION {
  // Threads - 1 named barriers required
  // but id 0 reserved for unnamed
  experimental_esimd::named_barrier_init<Threads>();

  int flag = 0; // producer-consumer mode
  int producers = 2;
  int consumers = 2;

  esimd::simd<int, VL> val(local_id);

  unsigned int slm_base =
      static_cast<uint32_t>(reinterpret_cast<std::uintptr_t>(
          local_acc.get_multi_ptr<access::decorated::no>().get_raw()));

  /* Each thread operates on a region of memory (global or slm) that overlaps
   * with that of the previous and next threads.
   */
  unsigned int off = VL * local_id / 2;

  esimd::barrier();

  /* Threads are executed in ascending order of their local ID, so there should
   * be no race conditions due to overlapping memory regions.
   *
   * local_id == 0 skips this branch and goes straight to LSC store
   * local_id == 1 signals barrier 1
   * local_id == 2 signals barrier 2
   * local_id == 3 signals barrier 3
   * and so on
   */
  if (local_id > 0) {
    int barrier_id = local_id;
    __ESIMD_ENS::named_barrier_signal(barrier_id, flag, producers, consumers);
    __ESIMD_ENS::named_barrier_wait(barrier_id);
  }

  /* This is the payload store with overlapping offset. Since threads are
   * executed in ascending order, each next thread will rewrite some amount of
   * data that the previous one wrote.
   */
  if constexpr (UseSLM)
    experimental_esimd::lsc_slm_block_store<int, VL>(slm_base + off * sizeof(int), val);
  else
    experimental_esimd::lsc_block_store<int, VL>(o + off, val);

  experimental_esimd::lsc_fence();

  /* local_id == 0 arrives here first and signals barrier 1
   * local_id == 1 arrives here next and signals barrier 2
   * local_id == 2 arrives here next and signals barrier 3
   * and so on, but last thread skips this block
   */
  if (local_id < Threads - 1) {
    int barrier_id = local_id + 1;
    __ESIMD_ENS::named_barrier_signal(barrier_id, flag, producers, consumers);
    __ESIMD_ENS::named_barrier_wait(barrier_id);
  }

  esimd::barrier();

  /* This section is only needed to copy the content of the SLM to the global
   * buffer for self-check purposes. Here we wait for all threads to sync, and
   * each thread now copies the a region from the SLM to the global buffer. To
   * do this, we double the value in off variable.
   */
  if constexpr (UseSLM) {
    off *= 2;
    auto res = experimental_esimd::lsc_slm_block_load<int, VL>(slm_base + off * sizeof(int));
    experimental_esimd::lsc_block_store<int, VL>(o + off, res);
  }
}

template <unsigned Threads, unsigned Size, bool UseSLM>
[[intel::device_indirectly_callable]] SYCL_EXTERNAL void __regcall SIMD_CALLEE_nbarrier(
#ifdef USE_ACC_PTR
    local_accessor<int, 1> *local_acc,
#else
    local_accessor<int, 1> local_acc,
#endif
    int local_id, int *o) SYCL_ESIMD_FUNCTION {
#ifdef USE_ACC_PTR
  ESIMD_CALLEE_nbarrier<Threads, Size, UseSLM>(*local_acc, local_id, o);
#else
  ESIMD_CALLEE_nbarrier<Threads, Size, UseSLM>(local_acc, local_id, o);
#endif
}

template <int CaseNum, unsigned Threads, bool UseSLM = false>
bool test(queue q) {
  std::cout << "Case #" << CaseNum << "\n Treads: " << Threads
            << ", Memory: " << (UseSLM ? "local\n" : "global\n");

  static_assert(Threads % 2 == 0, "Number of threads must be even");

  constexpr unsigned Size = VL * Threads;

  if constexpr (UseSLM) {
    auto dev = q.get_device();
    auto device_slm_size =
        dev.template get_info<sycl::info::device::local_mem_size>();

    // The test is going to use Size elements of int type.
    if (device_slm_size < Size * sizeof(int)) {
      // Report an error - the test needs a fix.
      std::cerr << "Error: Test needs more SLM memory than device has"
                << std::endl;
      return false;
    }
  }

  auto *out = malloc_shared<int>(Size, q);
  for (int i = 0; i < Size; i++) {
    out[i] = -1;
  }

  try {
    // workgroups
    sycl::range<1> global_range{Size};
    // threads in each group
    sycl::range<1> local_range{Size};

    auto e = q.submit([&](handler &cgh) {
      local_accessor<int, 1> local_acc;

      /* We only need to initialize local accessor for cases that use SLM,
       * other cases use global buffer and initialize local accessor with size
       * of 0 for compitability.
       */
      if constexpr (UseSLM)
        local_acc = local_accessor<int, 1>(Size, cgh);
      else
        local_acc = local_accessor<int, 1>(0, cgh); // dummy local accessor

      cgh.parallel_for<KernelID<CaseNum>>(
          nd_range<1>(global_range, local_range),
          // This test requires an explicit specification of the subgroup size
          [=](nd_item<1> item) [[intel::reqd_sub_group_size(VL)]] {
            sycl::group<1> g = item.get_group();
            sycl::sub_group sg = item.get_sub_group();

            // Thread's ID in ESIMD context
            int local_id = sg.get_group_linear_id();

            auto local_acc_copy = local_acc;
            // SLM init
            if constexpr (UseSLM) {
              uint32_t slm_id = item.get_local_id(0);
              local_acc_copy[slm_id] = -1;
	    }
            item.barrier();

#ifdef USE_ACC_PTR
            auto local_acc_arg = uniform{&local_acc_copy};
#else
            auto local_acc_arg = uniform{local_acc_copy};
#endif

            invoke_simd(sg, SIMD_CALLEE_nbarrier<Threads, Size, UseSLM>,
                        local_acc_arg, uniform{local_id}, uniform{out});
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
    /* Each thread stores its ID. Effectively, each thread stores VL / 2 number
     * of elements with an offset equal to ID * (VL / 2). The only exception is
     * the last thread, it effectively stores VL elements with the same offset.
     * So ID, the reference, can be calculated from i in the following way:
     */
    int ref = i * 2 / VL;
    if (ref == Threads) // last stored chunk
      ref -= 1;
    if (ref > Threads) // excessive part of buffer, not used
      ref = -1;
    if (out[i] != ref) {
      passed = false;
      std::cout << "out[" << i << "]=" << out[i] << " vs " << ref << "\n";
    }
  }

  free(out, q);

  std::cout << (passed ? " Passed\n" : " FAILED\n");
  return passed;
}

int main() {
  queue q;
  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<sycl::info::device::name>()
            << "\n";
  auto device_slm_size =
      dev.get_info<sycl::info::device::local_mem_size>();
  std::cout << "Local Memory Size: " << device_slm_size << std::endl;

  bool passed = true;

  passed &= test<1, 2>(q);
  passed &= test<2, 4>(q);
  passed &= test<3, 8>(q);
  passed &= test<4, 16>(q);
  passed &= test<5, 32>(q);

  constexpr bool UseSLM = true;
  passed &= test<6, 2, UseSLM>(q);
  passed &= test<7, 4, UseSLM>(q);
  passed &= test<8, 8, UseSLM>(q);
  passed &= test<9, 16, UseSLM>(q);
  passed &= test<10, 32, UseSLM>(q);

  return passed ? 0 : 1;
}
