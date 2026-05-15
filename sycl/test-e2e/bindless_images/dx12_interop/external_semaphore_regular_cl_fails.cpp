
//
// REQUIRES: aspect-ext_oneapi_external_semaphore_import, windows
//
// RUN: %{build} %link-directx -o %t.exe %if target-spir %{ -Wno-ignored-attributes %}
// RUN: %{run} %t.exe

// Waiting on a DX12-fence external semaphore from a queue backed by a
// regular (non-immediate) command list must throw sycl::exception.
//
// Mirrors vulkan_interop/external_semaphore_regular_cl_fails.cpp but uses
// the win32_nt_dx12_fence handle-type path. The Level Zero adapter
// rejects external_semaphore wait/signal at submission time when the
// queue is not using immediate command lists; this test verifies the
// rejection still fires for the DX12 fence handle type.
//
// Flow:
//   1. Create a real, exportable D3D12 timeline fence and signal it.
//   2. Import it into SYCL via a (lawful) immediate-CL queue.
//   3. Call ext_oneapi_wait_external_semaphore on a separate queue that
//      explicitly opts into no_immediate_command_list, and expect a
//      sycl::exception.

#include "d3d12_setup.hpp"
#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>
#include <sycl/properties/queue_properties.hpp>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

namespace syclexp = sycl::ext::oneapi::experimental;

int main() {
  D3D12Context d3dCtx = createD3D12Context();
  D3D12ExportableFence extFence = createExportableFence(d3dCtx);

  // Make the fence reach a known value so wait(value=1) is satisfiable
  // if it ever gets that far.
  signalExportableFence(d3dCtx, extFence);

  // Lawful queue: import the semaphore here.
  sycl::queue immQ{
      {sycl::property::queue::in_order{},
       sycl::ext::intel::property::queue::immediate_command_list{}}};
  auto device = immQ.get_device();
  auto context = immQ.get_context();

  auto semDesc =
      syclexp::external_semaphore_descriptor<syclexp::resource_win32_handle>{
          extFence.sharedHandle,
          syclexp::external_semaphore_handle_type::win32_nt_dx12_fence};
  syclexp::external_semaphore syclSem =
      syclexp::import_external_semaphore(semDesc, device, context);

  // The non-immediate-CL queue is what should trigger rejection on use.
  sycl::queue regQ{
      context, device,
      sycl::property_list{
          sycl::property::queue::in_order{},
          sycl::ext::intel::property::queue::no_immediate_command_list{}}};

  int ret = 1;
  try {
    regQ.ext_oneapi_wait_external_semaphore(syclSem, extFence.fenceValue);
    regQ.wait_and_throw();
    std::cerr << "FAIL: ext_oneapi_wait_external_semaphore (dx12_fence) on a "
                 "non-immediate-CL queue did not throw."
              << std::endl;
  } catch (const sycl::exception &e) {
    std::cout << "Got expected sycl::exception: " << e.what() << std::endl;
    ret = 0;
  }

  syclexp::release_external_semaphore(syclSem, device, context);
  cleanupExportableFence(extFence);
  return ret;
}
