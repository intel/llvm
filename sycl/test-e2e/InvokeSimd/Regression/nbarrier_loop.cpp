// TODO: enable on Windows once driver is ready
// NOTE: named barrier supported only since PVC
// REQUIRES: gpu-intel-pvc && linux
// UNSUPPORTED: cuda || hip
//
// TODO: enable when Jira issue resolved, currently fail with VISALTO enable
// XFAIL: gpu-intel-pvc
//
// RUN: %{build} -fno-sycl-device-code-split-esimd -Xclang -fsycl-allow-func-ptr -o %t.out
// RUN: env IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %{run} %t.out
//
// VISALTO enable run
// RUN: env IGC_VISALTO=63 IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %{run} %t.out

/*
 * Test checks support of named barrier in a loop in invoke_simd context.
 * First iteration has 1 barrier and 1 producer, second - 2 barriers and 2
 * producers. Producer stores data to SLM, then all threads read SLM and store
 * data to surface.
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

// TODO: When gpu driver can pass/accept accessor by value,
// the work-around undef #ifdef US_ACC_PTR should be removed.
#define USE_ACC_PTR

using namespace sycl;
using namespace sycl::ext::oneapi::experimental;
namespace esimd = sycl::ext::intel::esimd;
namespace experimental_esimd = sycl::ext::intel::experimental::esimd;

constexpr int VL = 16;
constexpr unsigned Groups = 1;
constexpr unsigned Threads = 8;
constexpr unsigned Size = Groups * Threads * VL;

class KernelID;

ESIMD_INLINE void ESIMD_CALLEE_nbarrier(local_accessor<int, 1> LocalAcc,
		                        int localID,
					int *o) SYCL_ESIMD_FUNCTION {
  // 2 named barriers, id 0 reserved for unnamed
  constexpr unsigned bnum = 3;

  experimental_esimd::named_barrier_init<bnum>();

  // slm size used in kernel
  constexpr unsigned slm_size = Size / 2;

  // 2 producers on first iteration, 1 producer on second
  unsigned int indexes[2][2] = {{1, 2}, {3, 3}}; // local ids of producers
  unsigned int prods[2] = {2, 1};                // number of producers

  unsigned int off = localID * VL;
  // producer writes to SLM, consumer reads what producer wrote
  unsigned int slm_base =
      static_cast<uint32_t>(
          reinterpret_cast<std::uintptr_t>(LocalAcc.get_pointer()));

  esimd::barrier();

  for (int b = bnum - 1; b > 0; b--) {
    int j = bnum - b - 1; // iteration index

    bool is_producer = localID == indexes[j][0] || localID == indexes[j][1];
    bool is_consumer = !is_producer;
    // only-consumer or only-producer modes
    unsigned int flag = is_producer ? 0x1 : 0x2;

    unsigned int producers = prods[j];
    unsigned int consumers = Threads - producers;

    if (is_producer) {
      unsigned int slm_store_off = j * sizeof(int) * slm_size / 4;
      // second iteration store partialy overlaps first iteration stores
      unsigned int dx = producers == 2 ? (localID - 1) : 0;
      slm_store_off += dx * sizeof(int) * slm_size / 2;
      simd<int, slm_size / 2> init(localID);
      // producer stores to SLM
      experimental_esimd::lsc_slm_block_store<int, slm_size / 2>(slm_base + slm_store_off, init);
    }

    __ESIMD_ENS::named_barrier_signal(b, flag, producers, consumers);

    if (is_consumer)
      __ESIMD_ENS::named_barrier_wait(b);

    auto val = experimental_esimd::lsc_slm_block_load<int, VL>(slm_base + off * sizeof(int));
    // and storing it to output surface
    experimental_esimd::lsc_fence();
    experimental_esimd::lsc_block_store<int, VL>(o + off + j * slm_size, val);
    experimental_esimd::lsc_fence();
  }
}

[[intel::device_indirectly_callable]] SYCL_EXTERNAL void __regcall SIMD_CALLEE_nbarrier(
#ifdef USE_ACC_PTR
    local_accessor<int, 1> *LocalAcc,
#else
    local_accessor<int, 1> LocalAcc,
#endif
    int localID, int *o) SYCL_ESIMD_FUNCTION {
#ifdef USE_ACC_PTR
  ESIMD_CALLEE_nbarrier(*LocalAcc, localID, o);
#else
  ESIMD_CALLEE_nbarrier(LocalAcc, localID, o);
#endif
}

int main() {
  auto q = queue{gpu_selector_v};
  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<sycl::info::device::name>()
            << "\n";

  auto deviceSLMSize = dev.template get_info<sycl::info::device::local_mem_size>();
  std::cout << "Local Memory Size: " << deviceSLMSize << std::endl;

  // The test is going to use Size elements of int type.
  if (deviceSLMSize < Size * sizeof(int)) {
    // Report an error - the test needs a fix.
    std::cerr << "Error: Test needs more SLM memory than device has"
              << std::endl;
    return -1;
  }

  auto *out = malloc_shared<int>(Size, q);
  for (int i = 0; i < Size; i++) {
    out[i] = -1;
  }

  try {
    // workgroups
    sycl::range<1> GlobalRange{Size};
    // threads in each group
    sycl::range<1> LocalRange{Size / Groups};
    sycl::nd_range<1> Range{GlobalRange * LocalRange, LocalRange};

    auto e = q.submit([&](handler &cgh) {
      auto LocalAcc = local_accessor<int, 1>(Size, cgh);
      cgh.parallel_for<KernelID>(
          nd_range<1>(GlobalRange, LocalRange),
          // This test requires an explicit specification of the subgroup size
          [=](nd_item<1> item) [[intel::reqd_sub_group_size(VL)]] {
            sycl::group<1> g = item.get_group();
            sycl::sub_group sg = item.get_sub_group();

            // Thread local ID in ESIMD context
            int localID = sg.get_group_linear_id();

	    // SLM init
            uint32_t slmID = item.get_local_id(0);
            auto LocalAccCopy = LocalAcc;
            LocalAccCopy[slmID] = -1;
            item.barrier();

#ifdef USE_ACC_PTR
            auto LocalAccArg = uniform{&LocalAccCopy};
#else
            auto LocalAccArg = uniform{LocalAccCopy};
#endif

            invoke_simd(sg, SIMD_CALLEE_nbarrier, LocalAccArg, uniform{localID}, uniform{out});
          });
    });
    e.wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    free(out, q);
    return -1;
  }

  bool passed = true;
  for (int i = 0; i < Size; i++) {
    int etalon = 2;
    if (i < Size / 4)
      etalon = 1;
    if (i >= Size / 2) {
      if (i < (7 * Size / 8)) {
        if (i < (5 * Size / 8))
          etalon = 1;
        else
          etalon = 3;
      }
    }
    if (out[i] != etalon) {
      passed = false;
      std::cout << "out[" << i << "]=" << std::hex << out[i] << " vs " << etalon
                << std::dec << std::endl;
    }
  }

  free(out, q);

  std::cout << (passed ? " Passed\n" : " FAILED\n");
  return passed ? 0 : 1;
}

