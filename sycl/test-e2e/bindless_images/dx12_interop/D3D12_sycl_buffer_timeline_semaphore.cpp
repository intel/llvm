// REQUIRES: aspect-ext_oneapi_external_memory_import
// REQUIRES: aspect-ext_oneapi_external_semaphore_import
// REQUIRES: windows

// UNSUPPORTED: windows
// UNSUPPORTED-TRACKER:

// RUN: %{build} -ld3d12 -ldxgi -ld3dcompiler -o %t.exe %if target-spir %{ -Wno-ignored-attributes %}
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
#include <string>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>
#include <vector>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

namespace syclexp = sycl::ext::oneapi::experimental;

int main(int argc, char **argv) {
  bool useSemaphores = true;
  int iterations = 100;
  size_t numElements = 1024;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--no-sem")
      useSemaphores = false;
    if (arg == "--iterations" && i + 1 < argc)
      iterations = std::stoi(argv[++i]);
    if (arg == "--size" && i + 1 < argc)
      numElements = std::stoi(argv[++i]);
  }

  size_t bufferSize = numElements * sizeof(uint32_t);

  std::cout << "Running SYCL D3D12 Buffer + Timeline Fence Stress Test\n";
  std::cout << "Elements: " << numElements << " | Iterations: " << iterations
            << " | Semaphores: " << (useSemaphores ? "ON" : "OFF") << "\n";

  // D3D12 SETUP
  D3D12Context d3dCtx = createD3D12Context();

  // Exportable device-local buffers
  D3D12BufferResources inBuf = createExportableBuffer(d3dCtx, bufferSize);
  D3D12BufferResources outBuf = createExportableBuffer(d3dCtx, bufferSize);

  // Host-visible staging buffers
  D3D12BufferResources inStaging = createUploadBuffer(d3dCtx, bufferSize);
  D3D12BufferResources outStaging = createReadbackBuffer(d3dCtx, bufferSize);

  // Interop Timeline Fence
  D3D12ExportableFence extFence;
  if (useSemaphores) {
    extFence = createExportableFence(d3dCtx);
  }

  // Set initial buffer states explicitly to COPY_DEST to avoid generic read
  // promotion issues
  d3dCtx.cmdAlloc->Reset();
  d3dCtx.cmdList->Reset(d3dCtx.cmdAlloc.Get(), nullptr);
  D3D12_RESOURCE_BARRIER initialBarriers[2] = {};
  initialBarriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  initialBarriers[0].Transition.pResource = inBuf.resource.Get();
  initialBarriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
  initialBarriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
  initialBarriers[0].Transition.Subresource =
      D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

  initialBarriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  initialBarriers[1].Transition.pResource = outBuf.resource.Get();
  initialBarriers[1].Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
  initialBarriers[1].Transition.StateAfter =
      D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
  initialBarriers[1].Transition.Subresource =
      D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
  d3dCtx.cmdList->ResourceBarrier(2, initialBarriers);
  d3dCtx.cmdList->Close();
  executeAndWait(d3dCtx);

  // SYCL INTEROP
  try {
    sycl::queue q;
    auto device = q.get_device();
    auto context = q.get_context();

    std::cout << "[SYCL] Device: "
              << device.get_info<sycl::info::device::name>() << "\n";

    // Import buffers
    syclexp::external_mem_descriptor<syclexp::resource_win32_handle> inDesc{
        inBuf.sharedHandle, syclexp::external_mem_handle_type::win32_nt_handle,
        bufferSize};
    syclexp::external_mem inExtMem =
        syclexp::import_external_memory(inDesc, device, context);

    syclexp::external_mem_descriptor<syclexp::resource_win32_handle> outDesc{
        outBuf.sharedHandle, syclexp::external_mem_handle_type::win32_nt_handle,
        bufferSize};
    syclexp::external_mem outExtMem =
        syclexp::import_external_memory(outDesc, device, context);

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

    std::cout << "[Test] Starting " << iterations
              << " iteration stress test...\n";

    for (int i = 1; i <= iterations; ++i) {
      uint64_t d3dSignalVal = (uint64_t)(2 * i - 1);
      uint64_t syclSignalVal = (uint64_t)(2 * i);

      // D3D12: Upload and copy
      void *mapped;
      inStaging.resource->Map(0, nullptr, &mapped);
      auto *data = static_cast<uint32_t *>(mapped);
      for (size_t j = 0; j < numElements; ++j)
        data[j] = (uint32_t)i;
      inStaging.resource->Unmap(0, nullptr);

      d3dCtx.cmdAlloc->Reset();
      d3dCtx.cmdList->Reset(d3dCtx.cmdAlloc.Get(), nullptr);

      d3dCtx.cmdList->CopyBufferRegion(inBuf.resource.Get(), 0,
                                       inStaging.resource.Get(), 0, bufferSize);

      // Barrier: CopyDest -> UAV
      D3D12_RESOURCE_BARRIER barrier = {};
      barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
      barrier.Transition.pResource = inBuf.resource.Get();
      barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
      barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
      barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
      d3dCtx.cmdList->ResourceBarrier(1, &barrier);

      d3dCtx.cmdList->Close();
      ID3D12CommandList *ppCommandLists[] = {d3dCtx.cmdList.Get()};
      d3dCtx.cmdQueue->ExecuteCommandLists(1, ppCommandLists);

      if (useSemaphores) {
        d3dCtx.cmdQueue->Signal(extFence.fence.Get(), d3dSignalVal);
      }

      // Host wait for upload to finish using the generic context fence
      d3dCtx.fenceValue++;
      d3dCtx.cmdQueue->Signal(d3dCtx.fence.Get(), d3dCtx.fenceValue);
      d3dCtx.fence->SetEventOnCompletion(d3dCtx.fenceValue, d3dCtx.fenceEvent);
      WaitForSingleObject(d3dCtx.fenceEvent, INFINITE);

      std::cout << "  [" << i << "] D3D12 upload done" << std::flush;

      // SYCL: Wait, execute, signal
      if (useSemaphores) {
        std::cout << ", SYCL sem-wait(" << d3dSignalVal << ")..." << std::flush;
        q.ext_oneapi_wait_external_semaphore(syclSem, d3dSignalVal);
        std::cout << "ok" << std::flush;
      }

      q.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::range<1>(numElements), [=](sycl::item<1> item) {
          size_t id = item.get_id(0);
          outPtr[id] = inPtr[id] * 2;
        });
      });

      if (useSemaphores) {
        std::cout << ", SYCL sem-signal(" << syclSignalVal << ")..."
                  << std::flush;
        q.ext_oneapi_signal_external_semaphore(syclSem, syclSignalVal);
        std::cout << "ok" << std::flush;
      }
      q.wait();
      std::cout << ", SYCL done" << std::flush;

      // D3D12: Readback and verify
      if (useSemaphores) {
        d3dCtx.cmdQueue->Wait(extFence.fence.Get(), syclSignalVal);
      }

      d3dCtx.cmdAlloc->Reset();
      d3dCtx.cmdList->Reset(d3dCtx.cmdAlloc.Get(), nullptr);

      // Barrier: UAV -> CopySource
      barrier.Transition.pResource = outBuf.resource.Get();
      barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
      barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
      d3dCtx.cmdList->ResourceBarrier(1, &barrier);

      d3dCtx.cmdList->CopyBufferRegion(outStaging.resource.Get(), 0,
                                       outBuf.resource.Get(), 0, bufferSize);

      // Barrier: revert outBuf back to UAV for next iteration
      barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
      barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
      d3dCtx.cmdList->ResourceBarrier(1, &barrier);

      d3dCtx.cmdList->Close();
      d3dCtx.cmdQueue->ExecuteCommandLists(1, ppCommandLists);

      // Host wait for readback
      std::cout << ", d3d-fence..." << std::flush;
      d3dCtx.fenceValue++;
      d3dCtx.cmdQueue->Signal(d3dCtx.fence.Get(), d3dCtx.fenceValue);
      d3dCtx.fence->SetEventOnCompletion(d3dCtx.fenceValue, d3dCtx.fenceEvent);

      if (WaitForSingleObject(d3dCtx.fenceEvent, 5000) == WAIT_TIMEOUT) {
        std::cerr << "\nTIMEOUT on host wait!\n";
        return 1;
      }
      std::cout << "ok" << std::flush;

      // Verify data
      outStaging.resource->Map(0, nullptr, &mapped);
      auto *outData = static_cast<uint32_t *>(mapped);
      uint32_t expected = (uint32_t)i * 2;
      int errors = 0;
      for (size_t j = 0; j < numElements; ++j) {
        if (outData[j] != expected) {
          if (errors++ < 5)
            std::cerr << "  [" << j << "]: got " << outData[j] << " expected "
                      << expected << "\n";
        }
      }
      outStaging.resource->Unmap(0, nullptr);

      if (errors > 0) {
        std::cerr << "\nFAILURE at iteration " << i << ": " << errors
                  << " mismatches\n";
        return 1;
      }

      // Reset inBuf state to COPY_DEST for next iteration
      d3dCtx.cmdAlloc->Reset();
      d3dCtx.cmdList->Reset(d3dCtx.cmdAlloc.Get(), nullptr);
      barrier.Transition.pResource = inBuf.resource.Get();
      barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
      barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
      d3dCtx.cmdList->ResourceBarrier(1, &barrier);
      d3dCtx.cmdList->Close();
      executeAndWait(d3dCtx); // Use generic helper to submit and wait

      if (i % 25 == 0 || i == 1)
        std::cout << " PASS\n";
      else
        std::cout << " ok\n";
    }

    std::cout << "SUCCESS! All " << iterations << " iterations passed.\n";

    // SYCL Cleanup
    syclexp::unmap_external_linear_memory(inPtr, q);
    syclexp::unmap_external_linear_memory(outPtr, q);
    if (useSemaphores)
      syclexp::release_external_semaphore(syclSem, device, context);
    syclexp::release_external_memory(inExtMem, device, context);
    syclexp::release_external_memory(outExtMem, device, context);

  } catch (sycl::exception &e) {
    std::cerr << "SYCL Exception: " << e.what() << "\n";
    return 1;
  }

  // D3D12 Cleanup
  if (useSemaphores)
    cleanupExportableFence(extFence);
  cleanupBuffer(inBuf);
  cleanupBuffer(outBuf);
  cleanupBuffer(inStaging);
  cleanupBuffer(outStaging);
  // Clean up the generic context event directly
  if (d3dCtx.fenceEvent)
    CloseHandle(d3dCtx.fenceEvent);

  return 0;
}