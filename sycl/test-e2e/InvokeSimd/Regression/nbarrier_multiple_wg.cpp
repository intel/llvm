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
 * Test checks support of named barrier in invoke_simd context with multiple
 * ESIMD work-groups. Producers store to SLM; consumers read SLM and store
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
template <int case_num> class KernelID;

template <unsigned Groups, unsigned Threads, unsigned Size>
ESIMD_INLINE void ESIMD_CALLEE_nbarrier(local_accessor<int, 1> LocalAcc,
                                        int groupID, int localID,
                                        int *o) SYCL_ESIMD_FUNCTION {
  // 1 named barrier, id 0 reserved for unnamed
  constexpr unsigned bnum = 2;
  constexpr unsigned bid = 1;
  experimental_esimd::named_barrier_init<bnum>();

  unsigned int group_off = VL * groupID * Threads;
  unsigned int globalID = groupID * Threads + localID;
  unsigned int global_off = VL * globalID;

  esimd::simd<int, VL> val(localID);

  constexpr unsigned producers = Threads / 2;
  constexpr unsigned consumers = Threads / 2;

  // thread with even local id is producer in each work-group
  bool is_producer = localID % 2 == 0;
  bool is_consumer = !is_producer;
  // only-producer or only-comsumer modes
  unsigned int flag = is_producer ? 0x1 : 0x2;

  // producer writes to SLM, consumer reads what producer wrote
  unsigned int off_slm =
      static_cast<uint32_t>(
          reinterpret_cast<std::uintptr_t>(LocalAcc.get_pointer())) +
      (is_producer ? global_off : (global_off - VL)) * sizeof(int);

  esimd::barrier();

  if (is_producer) {
    esimd::simd<int, VL> v(globalID);
    // producer stores data to SLM
    experimental_esimd::lsc_slm_block_store<int, VL>(off_slm, v);
  }

  // signaling after data stored
  __ESIMD_ENS::named_barrier_signal(bid, flag, producers, consumers);

  if (is_consumer) {
    // consumers waiting here for signal from producer
    __ESIMD_ENS::named_barrier_wait(bid);
    // read SLM and store to output
    auto ret = experimental_esimd::lsc_slm_block_load<int, VL>(off_slm);
    // store SLM to output
    experimental_esimd::lsc_block_store<int, VL>(o + global_off - VL, ret);
    experimental_esimd::lsc_block_store<int, VL>(o + global_off, ret);
  }
}

template <unsigned Groups, unsigned Threads, unsigned Size>
[[intel::device_indirectly_callable]] SYCL_EXTERNAL void __regcall SIMD_CALLEE_nbarrier(
#ifdef USE_ACC_PTR
    local_accessor<int, 1> *LocalAcc,
#else
    local_accessor<int, 1> LocalAcc,
#endif
    int groupID, int localID, int *o) SYCL_ESIMD_FUNCTION {
#ifdef USE_ACC_PTR
  ESIMD_CALLEE_nbarrier<Groups, Threads, Size>(*LocalAcc, groupID, localID, o);
#else
  ESIMD_CALLEE_nbarrier<Groups, Threads, Size>(LocalAcc, groupID, localID, o);
#endif
}

template <int case_num, unsigned Groups, unsigned Threads, class QueueTY>
bool test(QueueTY q) {
  constexpr unsigned Size = VL * Threads * Groups;

  static_assert(Threads > 1, "Threads number must be greater than 1");
  static_assert(Threads % 2 == 0, "Threads number expect to be even");
  static_assert(Groups > 1, "Threads number must be greater than 1");

  auto *out = malloc_shared<int>(Size, q);
  for (int i = 0; i < Size; i++) {
    out[i] = -1;
  }

  auto dev = q.get_device();
  auto deviceSLMSize =
      dev.template get_info<sycl::info::device::local_mem_size>();
  std::cout << "Local Memory Size: " << deviceSLMSize << std::endl;

  // The test is going to use Size elements of int type.
  if (deviceSLMSize < Size * sizeof(int)) {
    // Report an error - the test needs a fix.
    std::cerr << "Error: Test needs more SLM memory than device has"
              << std::endl;
    return false;
  }

  try {
    // workgroups
    sycl::range<1> GlobalRange{Size};
    // threads in each group
    sycl::range<1> LocalRange{Size / Groups};
    sycl::nd_range<1> Range{GlobalRange * LocalRange, LocalRange};

    auto e = q.submit([&](handler &cgh) {
      auto LocalAcc = local_accessor<int, 1>(Size, cgh);
      cgh.parallel_for<KernelID<case_num>>(
          nd_range<1>(GlobalRange, LocalRange),
          // This test requires an explicit specification of the subgroup size
          [=](nd_item<1> item) [[intel::reqd_sub_group_size(VL)]] {
            sycl::group<1> g = item.get_group();
            sycl::sub_group sg = item.get_sub_group();

            // Thread local ID in ESIMD context
            int localID = sg.get_group_linear_id();
            int groupID = g.get_group_linear_id();

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

            invoke_simd(sg, SIMD_CALLEE_nbarrier<Groups, Threads, Size>,
                        LocalAccArg, uniform{groupID}, uniform{localID},
                        uniform{out});
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
    int etalon = (i / (2 * VL)) * 2;
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

  passed &= test<1, 2, 8>(q);
  passed &= test<2, 4, 8>(q);
  passed &= test<3, 8, 8>(q);
  passed &= test<4, 4, 32>(q);
  passed &= test<5, 16, 16>(q);
  passed &= test<6, 32, 32>(q);

  return passed ? 0 : 1;
}
