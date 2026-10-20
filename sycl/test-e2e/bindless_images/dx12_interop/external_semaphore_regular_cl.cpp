// REQUIRES: aspect-ext_oneapi_external_semaphore_import
// REQUIRES: aspect-ext_intel_device_info_luid
// REQUIRES: windows, level_zero

// Regular command-list external semaphore support requires this driver
// version.
// REQUIRES-INTEL-DRIVER: win: 101.9030

// XFAIL: windows && run-mode && gpu-intel-gen12
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/23249

// RUN: %{build} %link-directx -o %t.exe %if target-spir %{ -Wno-ignored-attributes %}
// RUN: env UR_L0_BATCH_SIZE=8 SYCL_UR_TRACE=2 UR_L0_DEBUG=1 %{run} %t.exe \
// RUN:   2>&1 | FileCheck --check-prefixes=CHECK%if level_zero_v2_adapter %{,CHECK-V2%} %else %{,CHECK-V1%} %s
//
// CHECK-LABEL: SUBMISSION TRACE BEGIN
// CHECK-V1: zeCommandQueueExecuteCommandLists
// CHECK-V1: zeCommandQueueExecuteCommandLists
// CHECK-V2: zeCommandListImmediateAppendCommandListsExp
// CHECK-V2: zeCommandListImmediateAppendCommandListsExp
// CHECK-LABEL: SUBMISSION TRACE END

// Verify that external semaphore wait and signal operations work on an
// in-order queue backed by regular (non-immediate) command lists. The trace
// checks ensure that the status query does not hide a wait left in the batch.

#include "d3d12_setup.hpp"
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <sycl/detail/core.hpp>
#include <sycl/ext/intel/info/device.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>
#include <sycl/properties/queue_properties.hpp>
#include <thread>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

namespace syclexp = sycl::ext::oneapi::experimental;

D3D12Context createSyclD3D12Context(const sycl::device &device) {
  auto syclLuid = device.get_info<sycl::ext::intel::info::device::luid>();
  static_assert(sizeof(LUID) == sizeof(syclLuid));
  LUID adapterLuid{};
  std::memcpy(&adapterLuid, syclLuid.data(), sizeof(adapterLuid));

  ComPtr<IDXGIFactory4> factory;
  ThrowIfFailed(CreateDXGIFactory1(IID_PPV_ARGS(&factory)),
                "Failed to create DXGI factory");

  ComPtr<IDXGIAdapter1> adapter;
  ThrowIfFailed(factory->EnumAdapterByLuid(adapterLuid, IID_PPV_ARGS(&adapter)),
                "Failed to find the DXGI adapter for the SYCL device");
  return createD3D12Context(adapter.Get());
}

[[noreturn]] void failWithoutCleanup(const std::string &message) {
  std::cerr << message << std::endl;
  std::_Exit(EXIT_FAILURE);
}

bool remainsIncomplete(const sycl::event &event) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
  do {
    if (event.get_info<sycl::info::event::command_execution_status>() ==
        sycl::info::event_command_status::complete)
      return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  } while (std::chrono::steady_clock::now() < deadline);
  return true;
}

bool waitForCompletion(const sycl::event &event) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (event.get_info<sycl::info::event::command_execution_status>() !=
         sycl::info::event_command_status::complete) {
    if (std::chrono::steady_clock::now() >= deadline)
      return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return true;
}

int main() {
  sycl::queue q{
      {sycl::property::queue::in_order{},
       sycl::ext::intel::property::queue::no_immediate_command_list{}}};
  auto device = q.get_device();
  auto context = q.get_context();

  D3D12Context d3dCtx = createSyclD3D12Context(device);
  D3D12ExportableFence extFence = createExportableFence(d3dCtx);

  auto semDesc =
      syclexp::external_semaphore_descriptor<syclexp::resource_win32_handle>{
          extFence.sharedHandle,
          syclexp::external_semaphore_handle_type::win32_nt_dx12_fence};
  syclexp::external_semaphore syclSem =
      syclexp::import_external_semaphore(semDesc, device, context);

  try {
    const uint64_t D3DSignalValue = extFence.fenceValue + 1;
    std::cerr << "SUBMISSION TRACE BEGIN" << std::endl;
    q.single_task([]() {});
    sycl::event waitEvent =
        q.ext_oneapi_wait_external_semaphore(syclSem, D3DSignalValue);

    if (!remainsIncomplete(waitEvent))
      failWithoutCleanup(
          "The external semaphore wait completed before D3D12 signaled it.");
    std::cerr << "SUBMISSION TRACE END" << std::endl;

    signalExportableFence(d3dCtx, extFence);
    if (!waitForCompletion(waitEvent))
      failWithoutCleanup(
          "The external semaphore wait did not complete after D3D12 signaled "
          "it.");

    const uint64_t SyclSignalValue = D3DSignalValue + 1;
    q.ext_oneapi_signal_external_semaphore(syclSem, SyclSignalValue);

    ThrowIfFailed(extFence.fence->SetEventOnCompletion(SyclSignalValue,
                                                       d3dCtx.fenceEvent),
                  "Failed to wait for shared fence");
    constexpr DWORD TimeoutMs = 5000;
    DWORD waitResult = WaitForSingleObject(d3dCtx.fenceEvent, TimeoutMs);
    if (waitResult == WAIT_TIMEOUT)
      failWithoutCleanup(
          "Timed out waiting for the regular command-list semaphore signal.");
    if (waitResult != WAIT_OBJECT_0)
      failWithoutCleanup("Waiting for the external semaphore failed with "
                         "error " +
                         std::to_string(GetLastError()));

    q.wait_and_throw();
  } catch (const std::exception &e) {
    failWithoutCleanup(e.what());
  }

  syclexp::release_external_semaphore(syclSem, device, context);
  cleanupExportableFence(extFence);
  if (d3dCtx.fenceEvent)
    CloseHandle(d3dCtx.fenceEvent);
  return 0;
}
