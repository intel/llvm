// REQUIRES: aspect-ext_oneapi_external_memory_import
// REQUIRES: aspect-ext_oneapi_external_semaphore_import
// REQUIRES: windows

// UNSUPPORTED: gpu-intel-dg2
// UNSUPPORTED-TRACKER: GSD-12428
// semaphores-do-not-work-dg2

// UNSUPPORTED: gpu-intel-gen12
// UNSUPPORTED-TRACKER: GSD-12427
// Gen12-semaphores-work-but-this-test-hangs.

// UNSUPPORTED: arch-intel_gpu_bmg_g21
// UNSUPPORTED-TRACKER: GSD-12436
// this test works on BMG, but if run in parallel with itself, or with  other
// semaphore tests, it can hang.

// RUN: %{build} %link-directx -o %t.exe %if target-spir %{ -Wno-ignored-attributes %}
// RUN: %{run} %t.exe --no-sem
// RUN: %{run} %t.exe

// clang-format off
/*
  DirectX 12 / SYCL Buffer + Fence (Timeline) Interop Stress Test

  clang++.exe -fsycl -o dsbt.exe D3D12_sycl_buffer_timeline_semaphore.cpp -ld3d12 -ldxgi -ld3dcompiler

  Iteratively round-trips data through a D3D12 buffer and a SYCL kernel,
  synchronized via a single ID3D12Fence with monotonically increasing
  values.  No image APIs are used — only D3D12 buffers exported and mapped
  to SYCL USM pointers.

  Semaphore protocol (ID3D12Fence):
    One timeline fence, initial value 0.
    D3D12 signals odd values: 2*i - 1  (for iteration i = 1..N)
    SYCL  signals even values: 2*i

  Flow (per iteration i):
    1. D3D12: fill upload buffer with value i, copy to default buffer,
              signal fence = 2*i-1 on the command queue
    2. SYCL:  wait fence >= 2*i-1, kernel: out[j] = in[j] * 2,
              signal fence = 2*i
    3. D3D12: device-wait fence >= 2*i on the command queue,
              copy to readback buffer, CPU wait on fence, verify out[j] == i*2
*/
// clang-format on

#include "d3d12_setup.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>
#include <sycl/properties/queue_properties.hpp>
#include <vector>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

namespace syclexp = sycl::ext::oneapi::experimental;

//----------------------------------------------------------------------------//
void pause() {
  std::cin.clear(); // Clear any potential previous input left in the buffer.
  // std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

  std::cout << "\n\nPress Enter to continue . . .\n";   
  std::cin.get(); 
}
//----------------------------------------------------------------------------//
int main(int argc, char **argv) {
  bool   useSemaphores = true;
  bool   useTags       = false;
  int    iterations    = 100;
  size_t numElements   = 1024;

  // pause();

  DWORD pid = GetCurrentProcessId();
  std::ostringstream oss, pidStr;
  pidStr << "[" << std::setw(6) << pid << ":  0a] ";
  std::string pidTag = pidStr.str();
# define INCR_PID_TAG pidTag[11]++

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if      (arg == "--no-sem") useSemaphores = false;
    else if (arg == "--use-tags") useTags = true;
    else if (i + 1 < argc) {
      if      (arg == "--iterations") iterations = std::stoi(argv[++i]);
      else if (arg == "--size")      numElements = std::stoi(argv[++i]);
    }
  }

  const char *stop = (useTags ? "\n" : "");

  size_t bufferSize = numElements * sizeof(uint32_t);

  oss.str("");  oss.clear();
  oss << pidTag << "Running SYCL D3D12 Buffer + Timeline Fence Stress Test\n";
  INCR_PID_TAG;

  oss << pidTag << "Elements: " << numElements << " | Iterations: " << iterations
      << " | Semaphores: " << (useSemaphores ? "ON" : "OFF") << "\n";
  INCR_PID_TAG;

  // D3D12 SETUP
  D3D12Context d3dCtx = createD3D12Context();

  // Exportable device-local buffers
  D3D12BufferResources srcBuf = createExportableBuffer(d3dCtx, bufferSize);
  D3D12BufferResources dstBuf = createExportableBuffer(d3dCtx, bufferSize);

  // Host-visible staging buffers
  D3D12BufferResources srcStage = createUploadBuffer(d3dCtx, bufferSize);
  D3D12BufferResources dstStage = createReadbackBuffer(d3dCtx, bufferSize);

  // Interop Timeline Fence
  D3D12ExportableFence extFence;
  if (useSemaphores) {
    extFence = createExportableFence(d3dCtx);
    oss << pidTag << "Created exportable fence\n";  INCR_PID_TAG;
  }

  // Set initial buffer states explicitly to COPY_DEST to avoid generic read
  // promotion issues
  d3dCtx.cmdAlloc->Reset();
  d3dCtx.cmdList->Reset(d3dCtx.cmdAlloc.Get(), nullptr);
  D3D12_RESOURCE_BARRIER initialBarriers[2] = {};
  initialBarriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  initialBarriers[0].Transition.pResource = srcBuf.resource.Get();
  initialBarriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
  initialBarriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
  initialBarriers[0].Transition.Subresource =
      D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

  initialBarriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  initialBarriers[1].Transition.pResource = dstBuf.resource.Get();
  initialBarriers[1].Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
  initialBarriers[1].Transition.StateAfter =
      D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
  initialBarriers[1].Transition.Subresource =
      D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
  d3dCtx.cmdList->ResourceBarrier(2, initialBarriers);
  d3dCtx.cmdList->Close();
  executeAndWait(d3dCtx);
  oss << pidTag << "Initial buffer state setup done\n";  INCR_PID_TAG;
  std::cout << oss.str();  oss.str("");  oss.clear();

  // SYCL INTEROP
  try {
    sycl::queue q{sycl::ext::intel::property::queue::immediate_command_list{}};
    // sycl::queue q;
    auto device = q.get_device();
    auto context = q.get_context();

    oss << pidTag << "[SYCL] Device: " << device.get_info<sycl::info::device::name>() << "\n";
    INCR_PID_TAG;
    std::cout << oss.str();  oss.str("");  oss.clear();

    // Import buffers
    syclexp::external_mem_descriptor<syclexp::resource_win32_handle> inDesc{
        srcBuf.sharedHandle, syclexp::external_mem_handle_type::win32_nt_handle,
        bufferSize};
    syclexp::external_mem inExtMem =
        syclexp::import_external_memory(inDesc, device, context);

    syclexp::external_mem_descriptor<syclexp::resource_win32_handle> outDesc{
        dstBuf.sharedHandle, syclexp::external_mem_handle_type::win32_nt_handle,
        bufferSize};
    syclexp::external_mem outExtMem =
        syclexp::import_external_memory(outDesc, device, context);
    oss << pidTag << "[SYCL] Imported external memory\n";
    INCR_PID_TAG;
    std::cout << oss.str();  oss.str("");  oss.clear();

    // Import timeline fence
    syclexp::external_semaphore syclSem{};
    if (useSemaphores) {
      auto semDesc = syclexp::external_semaphore_descriptor<
          syclexp::resource_win32_handle>{
          extFence.sharedHandle,
          syclexp::external_semaphore_handle_type::win32_nt_dx12_fence};
      syclSem = syclexp::import_external_semaphore(semDesc, device, context);
    }

    uint32_t *inPtr = static_cast<uint32_t *>(
        syclexp::map_external_linear_memory(inExtMem, 0, bufferSize, q));
    uint32_t *outPtr = static_cast<uint32_t *>(
        syclexp::map_external_linear_memory(outExtMem, 0, bufferSize, q));

    oss << pidTag << "[Test] Starting " << iterations<< " iteration stress test...\n";
    INCR_PID_TAG;
    std::cout << oss.str();  oss.str("");  oss.clear();

    for (int i = 1; i <= iterations; ++i) {
      if (useTags) std::cout << "------------------------- Iteration " << std::setw(4) << i << " -------------------------\n";
      uint64_t dx12SignalVal = (uint64_t)(2 * i - 1);
      uint64_t syclSignalVal = (uint64_t)(2 * i);
      std::ostringstream idStream;
      std::ostringstream dx12ValStream;
      std::ostringstream syclValStream;
      idStream      << "[" << std::setw(6) << pid << ":" << std::setw(3) << i << "a] ";
      dx12ValStream << std::setw(4) << dx12SignalVal;
      syclValStream << std::setw(4) << syclSignalVal;
      std::string      idStr =      idStream.str();
      std::string dx12ValStr = dx12ValStream.str();
      std::string syclValStr = syclValStream.str();
#     define INCR_LID_TAG (useTags ? idStr[11]++ : 0)

      // D3D12: Upload and copy
      void *mapped;
      srcStage.resource->Map(0, nullptr, &mapped);
      auto *data = static_cast<uint32_t *>(mapped);
      for (size_t j = 0; j < numElements; ++j)
        data[j] = (uint32_t)i;
      srcStage.resource->Unmap(0, nullptr);

      d3dCtx.cmdAlloc->Reset();
      d3dCtx.cmdList->Reset(d3dCtx.cmdAlloc.Get(), nullptr);

      d3dCtx.cmdList->CopyBufferRegion(srcBuf.resource.Get(), 0,
                                       srcStage.resource.Get(), 0, bufferSize);

      // Barrier: CopyDest -> UAV
      D3D12_RESOURCE_BARRIER barrier = {};
      barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
      barrier.Transition.pResource = srcBuf.resource.Get();
      barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
      barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
      barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
      d3dCtx.cmdList->ResourceBarrier(1, &barrier);

      d3dCtx.cmdList->Close();
      ID3D12CommandList *ppCommandLists[] = {d3dCtx.cmdList.Get()};
      d3dCtx.cmdQueue->ExecuteCommandLists(1, ppCommandLists);

      if (useSemaphores) {
        oss << idStr << "D3D12 context command queue signal(" << dx12ValStr << "), " << stop;  INCR_LID_TAG;
        std::cout << oss.str() << std::flush;
        oss.str("");  oss.clear();
        if (!useTags) idStr = "";
        d3dCtx.cmdQueue->Signal(extFence.fence.Get(), dx12SignalVal);
        oss << idStr <<"ok, " << stop;  INCR_LID_TAG;
        std::cout << oss.str() << std::flush;
        oss.str("");  oss.clear();
      }

      // Host wait for upload to finish using the generic context fence
      d3dCtx.fenceValue++;
      d3dCtx.cmdQueue->Signal(d3dCtx.fence.Get(), d3dCtx.fenceValue);
      d3dCtx.fence->SetEventOnCompletion(d3dCtx.fenceValue, d3dCtx.fenceEvent);
      WaitForSingleObject(d3dCtx.fenceEvent, INFINITE);

      oss << idStr << "D3D12 upload done, " << stop;  INCR_LID_TAG;
      std::cout << oss.str() << std::flush;
      oss.str("");  oss.clear();
      if (!useTags) idStr = "";

      // SYCL: Wait, execute, signal
      if (useSemaphores) {
        oss << idStr << "SYCL sem-wait(" << dx12ValStr << ")..." << stop;  INCR_LID_TAG;
        std::cout << oss.str() << std::flush;
        oss.str("");  oss.clear();
        q.ext_oneapi_wait_external_semaphore(syclSem, dx12SignalVal);
        oss << idStr <<"ok, " << stop;  INCR_LID_TAG;
        std::cout << oss.str() << std::flush;
        oss.str("");  oss.clear();
      }

      q.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::range<1>(numElements), [=](sycl::item<1> item) {
          size_t id = item.get_id(0);
          outPtr[id] = inPtr[id] * 2;
        });
      });

      if (useSemaphores) {
        oss << idStr << "SYCL sem-signal(" << syclValStr << ")..." << stop;  INCR_LID_TAG;
        std::cout << oss.str() << std::flush;
        oss.str("");  oss.clear();
        q.ext_oneapi_signal_external_semaphore(syclSem, syclSignalVal);
        oss << idStr << "ok, " << stop;  INCR_LID_TAG;
        std::cout << oss.str() << std::flush;
        oss.str("");  oss.clear();
      }
      q.wait();
      oss << idStr << "SYCL done, " << stop;  INCR_LID_TAG;
      std::cout << oss.str() << std::flush;
      oss.str("");  oss.clear();

      // D3D12: Readback and verify
      if (useSemaphores) {
        d3dCtx.cmdQueue->Wait(extFence.fence.Get(), syclSignalVal);
      }

      d3dCtx.cmdAlloc->Reset();
      d3dCtx.cmdList->Reset(d3dCtx.cmdAlloc.Get(), nullptr);

      // Barrier: UAV -> CopySource
      barrier.Transition.pResource = dstBuf.resource.Get();
      barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
      barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
      d3dCtx.cmdList->ResourceBarrier(1, &barrier);

      d3dCtx.cmdList->CopyBufferRegion(dstStage.resource.Get(), 0,
                                       dstBuf.resource.Get(), 0, bufferSize);

      // Barrier: revert dstBuf back to UAV for next iteration
      barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
      barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
      d3dCtx.cmdList->ResourceBarrier(1, &barrier);

      d3dCtx.cmdList->Close();
      d3dCtx.cmdQueue->ExecuteCommandLists(1, ppCommandLists);

      // Host wait for readback
      oss << idStr << "D3D12 fence..." << stop;  INCR_LID_TAG;
      std::cout << oss.str() << std::flush;
      oss.str("");  oss.clear();
      d3dCtx.fenceValue++;
      d3dCtx.cmdQueue->Signal(d3dCtx.fence.Get(), d3dCtx.fenceValue);
      d3dCtx.fence->SetEventOnCompletion(d3dCtx.fenceValue, d3dCtx.fenceEvent);

      if (WaitForSingleObject(d3dCtx.fenceEvent, 5000) == WAIT_TIMEOUT) {
        std::cerr << "\n" << idStr << "TIMEOUT on host wait!\n";
        return 1;
      }
      oss << idStr << "ok --> " << stop;  INCR_LID_TAG;
      std::cout << oss.str() << std::flush;
      oss.str("");  oss.clear();
      // Verify data
      dstStage.resource->Map(0, nullptr, &mapped);
      auto *outData = static_cast<uint32_t *>(mapped);
      uint32_t expected = (uint32_t)i * 2;
      int errors = 0;
      for (size_t j = 0; j < numElements; ++j) {
        if (outData[j] != expected) {
          if (errors++ < 5)
            std::cerr << idStr << "(" << j << "): got " << outData[j] << " expected "
                      << expected << "\n";
        }
      }
      dstStage.resource->Unmap(0, nullptr);

      if (errors > 0) {
        std::cerr << "\n" << pidTag << "FAILURE at iteration " << i << ": " << errors
                  << " mismatches\n";  INCR_PID_TAG;
        return 1;
      }

      // Reset srcBuf state to COPY_DEST for next iteration
      d3dCtx.cmdAlloc->Reset();
      d3dCtx.cmdList->Reset(d3dCtx.cmdAlloc.Get(), nullptr);
      barrier.Transition.pResource = srcBuf.resource.Get();
      barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
      barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
      d3dCtx.cmdList->ResourceBarrier(1, &barrier);
      d3dCtx.cmdList->Close();
      executeAndWait(d3dCtx); // Use generic helper to submit and wait

      oss << idStr << "PASS\n";
      std::cout << oss.str();  oss.str("");  oss.clear();
    }

    oss << pidTag << "SUCCESS! All " << iterations << " iterations passed.\n";  INCR_PID_TAG;
    std::cout << oss.str();  oss.str("");  oss.clear();

    // SYCL Cleanup
    syclexp::unmap_external_linear_memory(inPtr, q);
    syclexp::unmap_external_linear_memory(outPtr, q);
    if (useSemaphores)
      syclexp::release_external_semaphore(syclSem, device, context);
    syclexp::release_external_memory(inExtMem, device, context);
    syclexp::release_external_memory(outExtMem, device, context);

  } catch (sycl::exception &e) {
    std::cerr << pidTag << "SYCL Exception: " << e.what() << "\n";
    return 1;
  }

  // D3D12 Cleanup
  if (useSemaphores)
    cleanupExportableFence(extFence);
  cleanupBuffer(srcBuf);
  cleanupBuffer(dstBuf);
  cleanupBuffer(srcStage);
  cleanupBuffer(dstStage);
  // Clean up the generic context event directly
  if (d3dCtx.fenceEvent)
    CloseHandle(d3dCtx.fenceEvent);

  oss << pidTag << "**** TEST COMPLETE **** -- Elements: " << numElements 
      << " | Iterations: " << iterations
      << " | Semaphores: " << (useSemaphores ? "ON" : "OFF") << "\n";
  std::cout << oss.str();  oss.str("");  oss.clear();

  return 0;
}
//----------------------------------------------------------------------------//
