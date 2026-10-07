// REQUIRES: aspect-ext_oneapi_external_semaphore_import, aspect-usm_shared_allocations
// REQUIRES: windows, level_zero

// Regular command-list external semaphore support requires this driver
// version.
// REQUIRES-INTEL-DRIVER: win: 101.9030

// XFAIL: windows && run-mode && gpu-intel-gen12
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/23249

// RUN: %{build} %link-directx -o %t.exe %if target-spir %{ -Wno-ignored-attributes %}
// RUN: env UR_L0_BATCH_SIZE=8 %{run} %t.exe

// Verify that external semaphore wait and signal operations work on an
// in-order queue backed by regular (non-immediate) command lists.

#include "d3d12_setup.hpp"
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <sycl/atomic_ref.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>
#include <sycl/properties/queue_properties.hpp>
#include <sycl/usm.hpp>
#include <thread>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

namespace syclexp = sycl::ext::oneapi::experimental;

[[noreturn]] void failWithoutCleanup(const std::string &message) {
  std::cerr << message << std::endl;
  std::_Exit(EXIT_FAILURE);
}

using MarkerAtomicRef =
    sycl::atomic_ref<int, sycl::memory_order::relaxed,
                     sycl::memory_scope::system,
                     sycl::access::address_space::global_space>;

bool waitForMarker(int *marker) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (MarkerAtomicRef(*marker).load() == 0) {
    if (std::chrono::steady_clock::now() >= deadline)
      return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return true;
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

  D3D12Context d3dCtx = createD3D12Context();
  D3D12ExportableFence extFence = createExportableFence(d3dCtx);

  auto semDesc =
      syclexp::external_semaphore_descriptor<syclexp::resource_win32_handle>{
          extFence.sharedHandle,
          syclexp::external_semaphore_handle_type::win32_nt_dx12_fence};
  syclexp::external_semaphore syclSem =
      syclexp::import_external_semaphore(semDesc, device, context);
  int *marker = sycl::malloc_shared<int>(1, q);
  MarkerAtomicRef(*marker).store(0);

  try {
    constexpr uint64_t D3DSignalValue = 1;
    q.single_task([=]() { MarkerAtomicRef(*marker).store(1); });
    sycl::event waitEvent =
        q.ext_oneapi_wait_external_semaphore(syclSem, D3DSignalValue);

    if (!waitForMarker(marker))
      failWithoutCleanup(
          "The external semaphore wait did not submit the preceding batch.");
    if (!remainsIncomplete(waitEvent))
      failWithoutCleanup(
          "The external semaphore wait completed before D3D12 signaled it.");

    signalExportableFence(d3dCtx, extFence);
    if (!waitForCompletion(waitEvent))
      failWithoutCleanup(
          "The external semaphore wait did not complete after D3D12 signaled "
          "it.");

    constexpr uint64_t SyclSignalValue = 2;
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

  sycl::free(marker, q);
  syclexp::release_external_semaphore(syclSem, device, context);
  cleanupExportableFence(extFence);
  if (d3dCtx.fenceEvent)
    CloseHandle(d3dCtx.fenceEvent);
  return 0;
}
