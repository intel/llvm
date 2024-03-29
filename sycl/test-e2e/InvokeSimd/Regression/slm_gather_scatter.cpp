// GPU driver had an error in handling of SLM aligned block_loads/stores,
// which has been fixed only in "1.3.26816", and in win/opencl version going
// _after_ 101.4575.
// REQUIRES-INTEL-DRIVER: lin: 26816, win: 101.4576
//
// RUN: %{build} -fno-sycl-device-code-split-esimd -Xclang -fsycl-allow-func-ptr -o %t.out
// RUN: env IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %{run} %t.out
//
// VISALTO enable run
// RUN: env IGC_VISALTO=63 IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %{run} %t.out

/*
 * Test check basic support of local memory access in invoke_simd.
 */

#include "../invoke_simd_utils.hpp"

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/oneapi/experimental/invoke_simd.hpp>
#include <sycl/sycl.hpp>

#include <functional>
#include <iostream>
#include <type_traits>

// TODO: When gpu driver can pass/accept accessor by value,
// the work-around undef #ifdef US_ACC_PTR should be removed.
#define USE_ACC_PTR

/* Subgroup size attribute is optional
 * In case it is absent compiler decides what subgroup size to use
 */
#ifdef IMPL_SUBGROUP
#define SUBGROUP_ATTR
#else
#define SUBGROUP_ATTR [[intel::reqd_sub_group_size(VL)]]
#endif

using namespace sycl;
using namespace sycl::ext::oneapi::experimental;
namespace esimd = sycl::ext::intel::esimd;

using dtype = int;

constexpr int VL = 16;
constexpr uint32_t LocalRange = VL * 2;          // 2 sub-groups per 1 group.
constexpr uint32_t GlobalRange = LocalRange * 2; // 2 groups.

ESIMD_INLINE void slm_load_store_test(
    local_accessor<dtype, 1> LocalAcc, uint32_t LAByteOffset, dtype *A,
    dtype *C, esimd::simd<uint32_t, VL> GlobalByteOffsets) SYCL_ESIMD_FUNCTION {

  uint32_t LocalAccOffset =
      static_cast<uint32_t>(
          reinterpret_cast<std::uintptr_t>(LocalAcc.get_pointer().get())) +
      LAByteOffset;
  esimd::simd<uint32_t, VL> Offsets(LocalAccOffset, sizeof(dtype));
  auto Local1 = esimd::slm_gather<dtype, VL>(Offsets);
  Offsets += static_cast<uint32_t>(LocalRange * sizeof(dtype));
  auto Local2 = esimd::slm_gather<dtype, VL>(Offsets);

  auto Global = esimd::gather(A, GlobalByteOffsets);
  auto Res = Global + Local1 + Local2;
  esimd::slm_scatter(Offsets, Res);
}

[[intel::device_indirectly_callable]] SYCL_EXTERNAL void __regcall invoke_slm_load_store_test(
#ifdef USE_ACC_PTR
    local_accessor<dtype, 1> *LocalAcc,
#else
    local_accessor<dtype, 1> LocalAcc,
#endif
    uint32_t SLMByteOffset, dtype *A, dtype *C,
    simd<uint32_t, VL> GlobalByteOffsets) SYCL_ESIMD_FUNCTION {
#ifdef USE_ACC_PTR
  slm_load_store_test(*LocalAcc, SLMByteOffset, A, C, GlobalByteOffsets);
#else
  slm_load_store_test(LocalAcc, SLMByteOffset, A, C, GlobalByteOffsets);
#endif
}

int main(void) {
  auto Q = queue{gpu_selector_v};
  auto Dev = Q.get_device();
  std::cout << "Running on " << Dev.get_info<sycl::info::device::name>()
            << std::endl;

  auto DeviceSLMSize = Dev.get_info<sycl::info::device::local_mem_size>();
  std::cout << "Local Memory Size: " << DeviceSLMSize << std::endl;

  sycl::nd_range<1> NDRange{range<1>{GlobalRange}, range<1>{LocalRange}};

  // The test is going to use (LocalRange * 2) elements of dtype type.
  if (DeviceSLMSize < LocalRange * 2 * sizeof(dtype)) {
    // Report an error - the test needs a fix.
    std::cerr << "Error: Test needs more SLM memory than device has"
              << std::endl;
    return 1;
  }

  auto *A = malloc_shared<dtype>(GlobalRange, Q);
  auto *C = malloc_shared<dtype>(GlobalRange, Q);

  for (auto i = 0; i < GlobalRange; i++) {
    A[i] = i;
    C[i] = 0;
  }
  try {
    Q.submit([&](handler &CGH) {
       auto LocalAcc = local_accessor<dtype, 1>(LocalRange * 2, CGH);
       CGH.parallel_for(NDRange, [=](nd_item<1> Item) SUBGROUP_ATTR {
         uint32_t GlobalId = Item.get_global_id(0);
         uint32_t LocalId = Item.get_local_id(0);
         auto LocalAccCopy = LocalAcc;
         LocalAccCopy[LocalId] = GlobalId * 100;
         LocalAccCopy[LocalId + LocalRange] = GlobalId * 10000;
         Item.barrier();

         uint32_t LAByteOffset = (LocalId / VL) * VL * sizeof(dtype);
         uint32_t GlobalByteOffset = GlobalId * sizeof(dtype);
         sycl::sub_group SG = Item.get_sub_group();
#ifdef USE_ACC_PTR
         auto LocalAccArg = uniform{&LocalAccCopy};
#else
         auto LocalAccArg = uniform{LocalAccCopy};
#endif
         invoke_simd(SG, invoke_slm_load_store_test, LocalAccArg,
                     uniform{LAByteOffset}, uniform{A}, uniform{C},
                     GlobalByteOffset);
         C[GlobalId] = LocalAccCopy[LocalId + LocalRange];
       });
     }).wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    free(A, Q);
    free(C, Q);
    return e.code().value();
  }

  bool Pass = true;
  for (auto i = 0; i < GlobalRange; i++) {
    dtype Expected = A[i] + i * (10000 + 100);
    if (C[i] != Expected) {
      std::cout << "Error: C[" << i << "]:" << C[i]
                << " != [expected]:" << Expected << std::endl;
      Pass = false;
    }
  }

  free(A, Q);
  free(C, Q);

  std::cout << "Test result: " << (Pass ? "Pass" : "Fail") << std::endl;
  return Pass ? 0 : 1;
}
