// REQUIRES: level_zero, level_zero_dev_kit
// RUN: %{build} %level_zero_options -o %t.out
// RUN: %{run} %t.out

// This is a variant of max_malloc.cpp that bypasses the SYCL/UR USM
// allocation APIs and calls the raw Level Zero zeMemAllocDevice()/
// zeMemAllocShared() functions directly instead. Unlike malloc_device()/
// malloc_shared(), which always return nullptr on any failure without
// exposing the specific error code (see
// sycl/source/detail/usm/usm_impl.cpp), the raw Level Zero API returns the
// actual ze_result_t, which lets this test tell a genuine, expected
// "out of device memory" failure near the device's memory ceiling (see
// https://github.com/intel/llvm/issues/22227) apart from any other,
// unexpected allocation failure.
//
// This test also explicitly calls zeContextMakeMemoryResident() after each
// zeMemAllocDevice(), mirroring what the UR Level Zero adapter's
// malloc_device() implementation does by default (see
// USMAllocationMakeResident() in
// unified-runtime/source/adapters/level_zero/usm.cpp), and times that call
// with std::chrono. This lets this raw-L0 test be used to independently
// verify whether zeContextMakeMemoryResident()'s per-call latency grows as
// more allocations accumulate over the course of the test, which is a
// suspected root cause of the max_malloc.cpp timeouts (see
// https://github.com/intel/llvm/issues/22405).

#include <chrono>
#include <iostream>
#include <level_zero/ze_api.h>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>

using namespace sycl;

const double Gb = 1024 * 1024 * 1024;

// Older Level Zero headers only have the (deprecated) "exp" relaxed
// allocation limits extension; newer ones have the non-deprecated "ext"
// version. Support both so this test builds against either.
#ifdef ZE_STRUCTURE_TYPE_RELAXED_ALLOCATION_LIMITS_EXT_DESC
using ze_relaxed_alloc_desc_t = ze_relaxed_allocation_limits_ext_desc_t;
constexpr auto ZeRelaxedAllocStype =
    ZE_STRUCTURE_TYPE_RELAXED_ALLOCATION_LIMITS_EXT_DESC;
constexpr auto ZeRelaxedAllocFlagMaxSize =
    ZE_RELAXED_ALLOCATION_LIMITS_EXT_FLAG_MAX_SIZE;
#else
using ze_relaxed_alloc_desc_t = ze_relaxed_allocation_limits_exp_desc_t;
constexpr auto ZeRelaxedAllocStype =
    ZE_STRUCTURE_TYPE_RELAXED_ALLOCATION_LIMITS_EXP_DESC;
constexpr auto ZeRelaxedAllocFlagMaxSize =
    ZE_RELAXED_ALLOCATION_LIMITS_EXP_FLAG_MAX_SIZE;
#endif

int main() {
  auto D = device(gpu_selector_v);

  std::cout << "name = " << D.get_info<info::device::name>() << std::endl;

  auto global_mem_size = D.get_info<info::device::global_mem_size>() / Gb;
  std::cout << "global_mem_size = " << global_mem_size << std::endl;
  std::cout << "max_mem_alloc_size = "
            << D.get_info<info::device::max_mem_alloc_size>() / Gb << std::endl;

  auto Q = queue(D);
  auto C = Q.get_context();

  ze_context_handle_t ZeContext = get_native<backend::ext_oneapi_level_zero>(C);
  ze_device_handle_t ZeDevice = get_native<backend::ext_oneapi_level_zero>(D);

  // Allow allocations larger than the device's default max_mem_alloc_size,
  // equivalent to what UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1 does for
  // malloc_device()/malloc_shared() in max_malloc.cpp.
  ze_relaxed_alloc_desc_t RelaxedLimits = {};
  RelaxedLimits.stype = ZeRelaxedAllocStype;
  RelaxedLimits.pNext = nullptr;
  RelaxedLimits.flags = ZeRelaxedAllocFlagMaxSize;

  for (int I = 1; I < global_mem_size; I++) {
    ze_device_mem_alloc_desc_t DeviceDesc = {};
    DeviceDesc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    DeviceDesc.pNext = &RelaxedLimits;
    DeviceDesc.flags = 0;
    DeviceDesc.ordinal = 0;

    void *p = nullptr;
    ze_result_t Res = zeMemAllocDevice(ZeContext, &DeviceDesc, I * Gb,
                                       /*alignment=*/0, ZeDevice, &p);
    std::cout << "zeMemAllocDevice(" << I << "Gb) = " << p
              << ", result = " << std::hex << Res << std::dec << std::endl;
    if (Res != ZE_RESULT_SUCCESS) {
      if (Res == ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY &&
          I > global_mem_size / 2) {
        std::cout
            << "zeMemAllocDevice(" << I
            << "Gb) failed with ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY past "
               "half of global_mem_size, treating this as an expected, "
               "benign allocation-ceiling failure."
            << std::endl;
        return 0;
      }
      std::cout << "FAILED" << std::endl;
      return -1;
    }

    // Mirror what the UR Level Zero adapter's malloc_device() does by
    // default: force the allocation resident on its device right away (see
    // USMAllocationMakeResident() in
    // unified-runtime/source/adapters/level_zero/usm.cpp). Time the call
    // in-process with std::chrono to get a reliable per-call latency,
    // independent of any stdout-buffering artifacts in the CI logs.
    auto MakeResidentStart = std::chrono::steady_clock::now();
    Res = zeContextMakeMemoryResident(ZeContext, ZeDevice, p, I * Gb);
    auto MakeResidentEnd = std::chrono::steady_clock::now();
    auto MakeResidentMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                              MakeResidentEnd - MakeResidentStart)
                              .count();
    std::cout << "zeContextMakeMemoryResident(" << I
              << "Gb) result = " << std::hex << Res << std::dec << ", took "
              << MakeResidentMs << " ms" << std::endl;
    if (Res != ZE_RESULT_SUCCESS) {
      std::cout << "FAILED (zeContextMakeMemoryResident)" << std::endl;
      return -1;
    }

    zeMemFree(ZeContext, p);

    ze_host_mem_alloc_desc_t HostDesc = {};
    HostDesc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
    HostDesc.pNext = nullptr;
    HostDesc.flags = 0;

    Res = zeMemAllocShared(ZeContext, &DeviceDesc, &HostDesc, I * Gb,
                           /*alignment=*/0, ZeDevice, &p);
    std::cout << "zeMemAllocShared(" << I << "Gb) = " << p
              << ", result = " << std::hex << Res << std::dec << std::endl;
    if (Res != ZE_RESULT_SUCCESS) {
      if (Res == ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY &&
          I > global_mem_size / 2) {
        std::cout
            << "zeMemAllocShared(" << I
            << "Gb) failed with ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY past "
               "half of global_mem_size, treating this as an expected, "
               "benign allocation-ceiling failure."
            << std::endl;
        return 0;
      }
      std::cout << "FAILED" << std::endl;
      return -1;
    }
    zeMemFree(ZeContext, p);
  }

  return 0;
}
