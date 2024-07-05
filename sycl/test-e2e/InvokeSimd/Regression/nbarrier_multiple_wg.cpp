// NOTE: named barrier supported only since PVC
// REQUIRES: gpu-intel-pvc
//
// RUN: %{build} -fno-sycl-device-code-split-esimd -Xclang -fsycl-allow-func-ptr -o %t.out
// RUN: env IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %{run} %t.out
//
// vISA LTO run
// RUN: env IGC_VISALTO=63 IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %{run} %t.out

/*
 * Test checks support of named barrier in invoke_simd context with multiple
 * ESIMD work-groups. Producers store to SLM; consumers read SLM and store
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
template <int CaseNum> class KernelID;

template <unsigned Groups, unsigned Threads, unsigned Size>
ESIMD_INLINE void ESIMD_CALLEE_nbarrier(local_accessor<int, 1> local_acc,
                                        int groupID, int local_id,
                                        int *o) SYCL_ESIMD_FUNCTION {
  // 1 named barrier, id 0 reserved for unnamed
  constexpr unsigned bnum = 2;
  constexpr unsigned bid = 1;
  experimental_esimd::named_barrier_init<bnum>();

  constexpr unsigned producers = Threads / 2;
  constexpr unsigned consumers = Threads / 2;

  // Thread with even local id is producer in each work-group.
  bool is_producer = local_id % 2 == 0;
  bool is_consumer = !is_producer;
  // Modes: only-producer or only-comsumer.
  unsigned int flag = is_producer ? 0x1 : 0x2;

  /* Global offset depends on if the thread is producer or consumer. Producer
   * is a thread with even id and it stores data to SLM. Consumer reads what
   * producer wrote, so consumer's offset has to be adjusted.
   */
  unsigned int global_id = groupID * Threads + local_id;
  unsigned int global_off = VL * (is_producer ? global_id : (global_id - 1));

  unsigned int slm_base =
      static_cast<uint32_t>(reinterpret_cast<std::uintptr_t>(
          local_acc.get_multi_ptr<access::decorated::no>().get_raw()));
  unsigned int slm_off = slm_base + global_off * sizeof(int);

  esimd::barrier();

  if (is_producer) {
    esimd::simd<int, VL> val(global_id);
    // Producer stores data to SLM.
    experimental_esimd::lsc_slm_block_store<int, VL>(slm_off, val);
  }

  __ESIMD_ENS::named_barrier_signal(bid, flag, producers, consumers);

  if (is_consumer) {
    // Consumers waiting here for signal from producer.
    __ESIMD_ENS::named_barrier_wait(bid);
    // Consumers simply copying producers data from SLM to global buffer.
    auto ret = experimental_esimd::lsc_slm_block_load<int, VL>(slm_off);
    experimental_esimd::lsc_block_store<int, VL>(o + global_off, ret);
  }
}

template <unsigned Groups, unsigned Threads, unsigned Size>
[[intel::device_indirectly_callable]] SYCL_EXTERNAL void __regcall SIMD_CALLEE_nbarrier(
#ifdef USE_ACC_PTR
    local_accessor<int, 1> *local_acc,
#else
    local_accessor<int, 1> local_acc,
#endif
    int groupID, int local_id, int *o) SYCL_ESIMD_FUNCTION {
#ifdef USE_ACC_PTR
  ESIMD_CALLEE_nbarrier<Groups, Threads, Size>(*local_acc, groupID, local_id, o);
#else
  ESIMD_CALLEE_nbarrier<Groups, Threads, Size>(local_acc, groupID, local_id, o);
#endif
}

template <int CaseNum, unsigned Groups, unsigned Threads>
bool test(queue q) {
  constexpr unsigned Size = VL * Threads * Groups;

  std::cout << "Case #" << CaseNum << "\n Treads: " << Threads
            << ", Groups: " << Groups << ", Size: " << Size << "\n";

  static_assert(Threads > 1, "Threads number must be greater than 1");
  static_assert(Threads % 2 == 0, "Threads number expect to be even");
  static_assert(Groups > 1, "Threads number must be greater than 1");

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

  auto *out = malloc_shared<int>(Size, q);
  for (int i = 0; i < Size; i++) {
    out[i] = -1;
  }

  try {
    // workgroups
    sycl::range<1> global_range{Size};
    // threads in each group
    sycl::range<1> local_range{Size / Groups};

    auto e = q.submit([&](handler &cgh) {
      auto local_acc = local_accessor<int, 1>(Size, cgh);
      cgh.parallel_for<KernelID<CaseNum>>(
          nd_range<1>(global_range, local_range),
          // This test requires an explicit specification of the subgroup size
          [=](nd_item<1> item) [[intel::reqd_sub_group_size(VL)]] {
            sycl::group<1> g = item.get_group();
            sycl::sub_group sg = item.get_sub_group();

            // Thread's ID in ESIMD context
            int local_id = sg.get_group_linear_id();
            int groupID = g.get_group_linear_id();

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

            invoke_simd(sg, SIMD_CALLEE_nbarrier<Groups, Threads, Size>,
                        local_acc_arg, uniform{groupID}, uniform{local_id},
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
    int ref = i / VL;
    /* Since the producers have an even thread id and are the only ones doing a
     * write, any odd ref must be skipped.
     */
    if (ref % 2 == 1)
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
      dev.template get_info<sycl::info::device::local_mem_size>();
  std::cout << "Local Memory Size: " << device_slm_size << std::endl;

  bool passed = true;

  passed &= test<1, 2, 8>(q);
  passed &= test<2, 4, 8>(q);
  passed &= test<3, 8, 8>(q);
  passed &= test<4, 4, 32>(q);
  passed &= test<5, 16, 16>(q);
  passed &= test<6, 32, 32>(q);

  return passed ? 0 : 1;
}
