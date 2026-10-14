// REQUIRES: hip

// RUN: %{build} -o %t.out %hip_options
// The opt-in launch is skipped when the opt-in ceiling equals the non-opt-in
// limit. Otherwise it uses one byte more than that limit.
// RUN: %{run} %t.out
// A cap below the device ceiling lowers the query and rejects a larger launch.
// RUN: %{run} UR_HIP_MAX_LOCAL_MEM_SIZE=32768 %t.out

//==------------------------ hip-local-mem-optin.cpp ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Dynamic local memory above the non-opt-in per-block limit has to opt in with
// hipFuncAttributeMaxDynamicSharedMemorySize. AMD devices report the same
// ceiling for both attributes (64KiB LDS on gfx942), so that launch is skipped
// there. The test still checks the reported ceiling, a launch of that size,
// and rejection of one byte more. UR_HIP_MAX_LOCAL_MEM_SIZE caps both.

#include <sycl/detail/core.hpp>
#include <sycl/ext/intel/info/device.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>

#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
#define __HIP_PLATFORM_AMD__
#endif
#include <hip/hip_runtime_api.h>

namespace {

int queryAttribute(int Device, hipDeviceAttribute_t Attr, bool *Ok) {
  int Value = 0;
  hipError_t Err = hipDeviceGetAttribute(&Value, Attr, Device);
  if (Err != hipSuccess) {
    (void)hipGetLastError();
    *Ok = false;
    return 0;
  }
  *Ok = true;
  return Value;
}

bool launchAndCheck(sycl::queue &Q, size_t Bytes) {
  sycl::buffer<unsigned char, 1> Out{sycl::range<1>(2)};
  Q.submit([&](sycl::handler &CGH) {
     sycl::accessor OutAcc{Out, CGH, sycl::write_only};
     sycl::local_accessor<unsigned char, 1> Local{Bytes, CGH};
     CGH.parallel_for(sycl::nd_range<1>{1, 1}, [=](sycl::nd_item<1>) {
       Local[0] = 0xA5;
       Local[Bytes - 1] = 0x5A;
       OutAcc[0] = Local[0];
       OutAcc[1] = Local[Bytes - 1];
     });
   }).wait();
  sycl::host_accessor Host{Out, sycl::read_only};
  if (Host[0] != 0xA5 || Host[1] != 0x5A) {
    std::printf("launch %zu read back %u %u\n", Bytes, Host[0], Host[1]);
    return false;
  }
  std::printf("launch %zu OK\n", Bytes);
  return true;
}

bool expectReject(sycl::queue &Q, size_t Bytes, const char *Expected) {
  try {
    launchAndCheck(Q, Bytes);
  } catch (const std::exception &E) {
    if (std::string(E.what()).find(Expected) == std::string::npos) {
      std::printf("launch %zu rejected with unexpected message: %s\n", Bytes,
                  E.what());
      return false;
    }
    std::printf("launch %zu rejected as expected\n", Bytes);
    return true;
  }
  std::printf("launch %zu was expected to fail\n", Bytes);
  return false;
}

} // namespace

int main() {
  sycl::queue Q;
  const std::string Pci =
      Q.get_device().get_info<sycl::ext::intel::info::device::pci_address>();
  int DeviceCount = 0;
  if (hipGetDeviceCount(&DeviceCount) != hipSuccess || DeviceCount <= 0) {
    std::printf("failed to enumerate HIP devices\n");
    return 1;
  }
  int HipDevice = -1;
  for (int Index = 0; Index < DeviceCount; ++Index) {
    char BusId[32] = {};
    if (hipDeviceGetPCIBusId(BusId, sizeof(BusId), Index) == hipSuccess &&
        Pci == BusId) {
      HipDevice = Index;
      break;
    }
  }
  if (HipDevice < 0) {
    std::printf("no HIP device matches PCI address %s\n", Pci.c_str());
    return 1;
  }

  bool DefaultOk = false;
  const int Default = queryAttribute(
      HipDevice, hipDeviceAttributeMaxSharedMemoryPerBlock, &DefaultOk);
  if (!DefaultOk || Default <= 0) {
    std::printf("failed to query non-opt-in shared memory on HIP device %d\n",
                HipDevice);
    return 1;
  }

  bool OptinOk = false;
  int Optin = queryAttribute(
      HipDevice, hipDeviceAttributeSharedMemPerBlockOptin, &OptinOk);
  // Match the adapter: a rejected or non-positive opt-in query keeps the
  // non-opt-in size.
  if (!OptinOk || Optin <= 0)
    Optin = Default;
  const int Ceiling = std::max(Default, Optin);

  const auto Reported = static_cast<size_t>(
      Q.get_device().get_info<sycl::info::device::local_mem_size>());

  const char *CapEnv = std::getenv("UR_HIP_MAX_LOCAL_MEM_SIZE");
  size_t Expected = static_cast<size_t>(Ceiling);
  if (CapEnv) {
    const int Cap = std::atoi(CapEnv);
    if (Cap <= 0) {
      std::printf("UR_HIP_MAX_LOCAL_MEM_SIZE is not a positive size\n");
      return 1;
    }
    Expected = static_cast<size_t>(std::min(Cap, Ceiling));
  }

  std::printf("device %d default %d opt-in %d ceiling %d reported %zu\n",
              HipDevice, Default, Optin, Ceiling, Reported);
  if (Reported != Expected) {
    std::printf("local_mem_size %zu, expected %zu\n", Reported, Expected);
    return 1;
  }

  if (Reported == 0 || !launchAndCheck(Q, Reported))
    return 1;

  const char *Reject =
      (Reported + 1 > static_cast<size_t>(Ceiling))
          ? "Excessive allocation of local memory on the device"
          : "UR_HIP_MAX_LOCAL_MEM_SIZE";
  if (!expectReject(Q, Reported + 1, Reject))
    return 1;

  // One byte past the non-opt-in limit is the hipFuncSetAttribute path. It is
  // unreachable when the ceiling equals that limit (gfx942 and the other AMD
  // ISAs) or when the env cap keeps the launch inside the default pool.
  const size_t OptInBytes = static_cast<size_t>(Default) + 1;
  if (Ceiling > Default && OptInBytes <= Reported && OptInBytes != Reported) {
    if (!launchAndCheck(Q, OptInBytes))
      return 1;
    std::printf("opt-in launch %zu OK\n", OptInBytes);
  } else if (Ceiling > Default && OptInBytes == Reported) {
    std::printf("opt-in launch covered by the %zu-byte launch\n", Reported);
  } else {
    std::printf("skipped opt-in launch: ceiling %d default %d reported %zu\n",
                Ceiling, Default, Reported);
  }
  return 0;
}
