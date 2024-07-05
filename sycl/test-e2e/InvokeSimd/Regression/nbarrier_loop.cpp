// NOTE: named barrier supported only since PVC
// REQUIRES: gpu-intel-pvc
//
// RUN: %{build} -fno-sycl-device-code-split-esimd -Xclang -fsycl-allow-func-ptr -o %t.out
// RUN: env IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %{run} %t.out
//
// vISA LTO run
// RUN: env IGC_VISALTO=63 IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %{run} %t.out

/*
 * Test checks support of named barrier in a loop in invoke_simd context.
 * First iteration has 1 barrier and 1 producer, second - 2 barriers and 2
 * producers. Producer stores data to SLM, then all threads read SLM and store
 * data to surface.
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
constexpr unsigned Groups = 1;
constexpr unsigned Threads = 4;
constexpr unsigned Items = Groups * Threads * VL;

// The size is chosen so that the entire surface can be covered in 2 stores with
// SIMD size of VL
constexpr unsigned Size = Items / 2;

class KernelID;

ESIMD_INLINE void ESIMD_CALLEE_nbarrier(local_accessor<int, 1> local_acc,
                                        int local_id,
                                        int *o) SYCL_ESIMD_FUNCTION {
  // 2 named barriers, id 0 reserved for unnamed
  constexpr unsigned bnum = 3;

  experimental_esimd::named_barrier_init<bnum>();

  // 2 producers on first iteration, 1 producer on second
  unsigned int indexes[2][2] = {{1, 2}, {3, 3}}; // local ids of producers
  unsigned int prods[2] = {2, 1};                // number of producers

  // Producer writes to SLM, consumer reads what producer wrote.
  unsigned int slm_base =
      static_cast<uint32_t>(reinterpret_cast<std::uintptr_t>(
          local_acc.get_multi_ptr<access::decorated::no>().get_raw()));

  esimd::barrier();

  // Iterate over barrier ids, starting with 1, since id 0 is reserved for
  // unnamed barriers.
  for (int b = 1; b < bnum; b++) {
    // index to use in arrays and calculate offsets.
    int index = b - 1;

    bool is_producer = local_id == indexes[index][0] || local_id == indexes[index][1];
    bool is_consumer = !is_producer;
    // Modes: only-producer or only-consumer
    unsigned int flag = is_producer ? 0x1 : 0x2;

    unsigned int producers = prods[index];
    unsigned int consumers = Threads - producers;

    if (is_producer) {
      /* The first iteration has 2 producers that cover all allocated SLM in
       * equal parts. The second iteration has only 1 producer that stores data
       * in the middle of the region, partially rewriting the results of the
       * first iteration.
       */
      unsigned int slm_off = slm_base;
      if (index == 0) {
        // Here local_id is 1 or 2.
	// local_id 1 stores to lower half of SLM
	// local_id 2 to upper half
        slm_off += (local_id - 1) * VL * sizeof(int);
      } else {
        // Here only one producer stores right in the middle.
        slm_off += (Size / 4) * sizeof(int);
      }

      simd<int, VL> init(local_id);
      experimental_esimd::lsc_slm_block_store<int, VL>(slm_off, init);
    }

    __ESIMD_ENS::named_barrier_signal(b, flag, producers, consumers);

    if (is_consumer)
      __ESIMD_ENS::named_barrier_wait(b);
  }

  experimental_esimd::lsc_fence();

  // Copying SLM content to buffer.
  if (local_id == 0) {
    for (int off = 0; off < Size; off += VL) {
      auto val = experimental_esimd::lsc_slm_block_load<int, VL>(
          slm_base + off * sizeof(int));
      experimental_esimd::lsc_block_store<int, VL>(o + off, val);
    }
  }
}

[[intel::device_indirectly_callable]] SYCL_EXTERNAL void __regcall SIMD_CALLEE_nbarrier(
#ifdef USE_ACC_PTR
    local_accessor<int, 1> *local_acc,
#else
    local_accessor<int, 1> local_acc,
#endif
    int local_id, int *o) SYCL_ESIMD_FUNCTION {
#ifdef USE_ACC_PTR
  ESIMD_CALLEE_nbarrier(*local_acc, local_id, o);
#else
  ESIMD_CALLEE_nbarrier(local_acc, local_id, o);
#endif
}

int main() {
  queue q;
  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<sycl::info::device::name>()
            << "\n";

  auto device_slm_size =
      dev.get_info<sycl::info::device::local_mem_size>();
  std::cout << "Local Memory Size: " << device_slm_size << std::endl;

  // The test is going to use Size elements of int type.
  if (device_slm_size < Size * sizeof(int)) {
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
    sycl::range<1> GlobalRange{Items};
    // threads in each group
    sycl::range<1> LocalRange{Items / Groups};

    auto e = q.submit([&](handler &cgh) {
      auto local_acc = local_accessor<int, 1>(Size, cgh);
      cgh.parallel_for<KernelID>(
          nd_range<1>(GlobalRange, LocalRange),
          // This test requires an explicit specification of the subgroup size
          [=](nd_item<1> item) [[intel::reqd_sub_group_size(VL)]] {
            sycl::group<1> g = item.get_group();
            sycl::sub_group sg = item.get_sub_group();

            // Thread's ID in ESIMD context
            int local_id = sg.get_group_linear_id();

            // SLM init
            uint32_t slm_id = item.get_local_id(0);
            auto local_acc_copy = local_acc;
            local_acc_copy[slm_id] = -1;
            item.barrier();

#ifdef USE_ACC_PTR
            auto local_acc_arg = uniform{&local_acc_copy};
#else
            auto local_acc_arg = uniform{local_acc_copy};
#endif

            invoke_simd(sg, SIMD_CALLEE_nbarrier, local_acc_arg, uniform{local_id},
                        uniform{out});
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
    int ref = -1;
    if (i < Size / 4) // lower quarter
      ref = 1;
    else if (i >= 3 * Size / 4) // upper quarter
      ref = 2;
    else // middle part
      ref = 3;

    if (out[i] != ref) {
      passed = false;
      std::cout << "out[" << i << "]=" << std::hex << out[i] << " vs " << ref
                << std::dec << std::endl;
    }
  }

  free(out, q);

  std::cout << (passed ? " Passed\n" : " FAILED\n");
  return passed ? 0 : 1;
}
