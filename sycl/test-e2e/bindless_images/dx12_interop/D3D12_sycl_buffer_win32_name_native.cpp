// REQUIRES: aspect-ext_oneapi_external_memory_import
// REQUIRES: aspect-ext_oneapi_external_semaphore_import
// REQUIRES: windows

// UNSUPPORTED: gpu-intel-dg2
// UNSUPPORTED-TRACKER: GSD-12428

// UNSUPPORTED: gpu-intel-gen12
// UNSUPPORTED-TRACKER: GSD-12427

// UNSUPPORTED: arch-intel_gpu_bmg_g21
// UNSUPPORTED-TRACKER: GSD-12436

// RUN: %{build} %link-directx -o %t.exe %if target-spir %{ -Wno-ignored-attributes %}
// RUN: %{run} %t.exe --no-sem
// RUN: %{run} %t.exe

// clang-format off
/*
  DirectX 12 / SYCL Buffer + Fence (Timeline) Interop Test - resource_win32_name

  Tests native resource_win32_name support in SYCL.
  SYCL runtime internally converts named handles to regular handles.
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

// Named resource structure
struct D3D12NamedBuffer {
  Microsoft::WRL::ComPtr<ID3D12Resource> resource;
  std::wstring name;
  HANDLE keepAliveHandle; // Must keep at least one handle open for the name to
                          // persist
};

struct D3D12NamedFence {
  Microsoft::WRL::ComPtr<ID3D12Fence> fence;
  std::wstring name;
  HANDLE keepAliveHandle;
};

// Create a buffer with a named shared handle
D3D12NamedBuffer createNamedExportableBuffer(D3D12Context &ctx, size_t size,
                                             const wchar_t *name) {
  D3D12NamedBuffer result;
  result.name = name;

  D3D12_HEAP_PROPERTIES heapProps = {};
  heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

  D3D12_RESOURCE_DESC desc = {};
  desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  desc.Width = size;
  desc.Height = 1;
  desc.DepthOrArraySize = 1;
  desc.MipLevels = 1;
  desc.SampleDesc.Count = 1;
  desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

  HRESULT hr = ctx.device->CreateCommittedResource(
      &heapProps, D3D12_HEAP_FLAG_SHARED, &desc, D3D12_RESOURCE_STATE_COMMON,
      nullptr, IID_PPV_ARGS(&result.resource));
  if (FAILED(hr)) {
    throw std::runtime_error("Failed to create named buffer");
  }

  // Create a NAMED shared handle and keep it open so the name persists
  hr = ctx.device->CreateSharedHandle(result.resource.Get(), nullptr,
                                      GENERIC_ALL, name,
                                      &result.keepAliveHandle);
  if (FAILED(hr)) {
    throw std::runtime_error("Failed to create named shared handle");
  }

  std::wcout << L"[D3D12] Created named buffer: " << name << L"\n";

  return result;
}

// Create a fence with a named shared handle
D3D12NamedFence createNamedExportableFence(D3D12Context &ctx,
                                           const wchar_t *name) {
  D3D12NamedFence result;
  result.name = name;

  HRESULT hr = ctx.device->CreateFence(0, D3D12_FENCE_FLAG_SHARED,
                                       IID_PPV_ARGS(&result.fence));
  if (FAILED(hr)) {
    throw std::runtime_error("Failed to create fence");
  }

  // Create a NAMED shared handle and keep it open so the name persists
  hr = ctx.device->CreateSharedHandle(result.fence.Get(), nullptr, GENERIC_ALL,
                                      name, &result.keepAliveHandle);
  if (FAILED(hr)) {
    throw std::runtime_error("Failed to create named shared fence handle");
  }

  std::wcout << L"[D3D12] Created named fence: " << name << L"\n";

  return result;
}

int main(int argc, char **argv) {
  bool useSemaphores = true;
  int iterations = 10;
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

  std::cout << "Running SYCL D3D12 resource_win32_name Native Test\n";
  std::cout << "Elements: " << numElements << " | Iterations: " << iterations
            << " | Semaphores: " << (useSemaphores ? "ON" : "OFF") << "\n";

  // D3D12 SETUP
  D3D12Context d3dCtx = createD3D12Context();

  // Create NAMED exportable buffers
  D3D12NamedBuffer inBuf = createNamedExportableBuffer(
      d3dCtx, bufferSize, L"Global\\SYCLTestInputBuffer3");
  D3D12NamedBuffer outBuf = createNamedExportableBuffer(
      d3dCtx, bufferSize, L"Global\\SYCLTestOutputBuffer3");

  // Host-visible staging buffers (not named, not exported)
  D3D12BufferResources inStaging = createUploadBuffer(d3dCtx, bufferSize);
  D3D12BufferResources outStaging = createReadbackBuffer(d3dCtx, bufferSize);

  // Interop Timeline Fence - NAMED
  D3D12NamedFence extFence;
  if (useSemaphores) {
    extFence = createNamedExportableFence(d3dCtx, L"Global\\SYCLTestFence3");
  }

  // Set initial buffer states
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

  // SYCL INTEROP - using resource_win32_name NATIVELY
  try {
    sycl::queue q;
    auto device = q.get_device();
    auto context = q.get_context();

    std::cout << "[SYCL] Device: "
              << device.get_info<sycl::info::device::name>() << "\n";

    // Import buffers BY NAME using resource_win32_name
    // Pass D3D12 device pointer so SYCL can open the named handle
    std::cout << "[SYCL] Importing input buffer by name (native support)\n";
    syclexp::external_mem_descriptor<syclexp::resource_win32_name> inDesc{
        {(const void *)inBuf.name.c_str(), d3dCtx.device.Get()},
        syclexp::external_mem_handle_type::win32_nt_handle,
        bufferSize};
    syclexp::external_mem inExtMem =
        syclexp::import_external_memory(inDesc, device, context);

    std::cout << "[SYCL] Importing output buffer by name (native support)\n";
    syclexp::external_mem_descriptor<syclexp::resource_win32_name> outDesc{
        {(const void *)outBuf.name.c_str(), d3dCtx.device.Get()},
        syclexp::external_mem_handle_type::win32_nt_handle,
        bufferSize};
    syclexp::external_mem outExtMem =
        syclexp::import_external_memory(outDesc, device, context);

    // Import timeline fence BY NAME
    syclexp::external_semaphore syclSem{};
    if (useSemaphores) {
      std::cout << "[SYCL] Importing fence by name (native support)\n";
      auto semDesc =
          syclexp::external_semaphore_descriptor<syclexp::resource_win32_name>{
              {(const void *)extFence.name.c_str(), d3dCtx.device.Get()},
              syclexp::external_semaphore_handle_type::win32_nt_dx12_fence};
      syclSem = syclexp::import_external_semaphore(semDesc, device, context);
    }

    uint32_t *inPtr = static_cast<uint32_t *>(
        syclexp::map_external_linear_memory(inExtMem, 0, bufferSize, q));
    uint32_t *outPtr = static_cast<uint32_t *>(
        syclexp::map_external_linear_memory(outExtMem, 0, bufferSize, q));

    std::cout << "[Test] Starting " << iterations << " iteration test...\n";

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

      // Host wait for upload to finish
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
        std::cout << ", SYCL sem-signal(" << syclSignalVal << ")" << std::flush;
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
      executeAndWait(d3dCtx);

      if (i % 5 == 0 || i == 1)
        std::cout << " PASS\n";
      else
        std::cout << " ok\n";
    }

    std::cout << "SUCCESS! All " << iterations << " iterations passed.\n";

    // SYCL Cleanup - handles are automatically closed by release functions
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
  CloseHandle(inBuf.keepAliveHandle);
  CloseHandle(outBuf.keepAliveHandle);
  if (useSemaphores)
    CloseHandle(extFence.keepAliveHandle);
  cleanupBuffer(inStaging);
  cleanupBuffer(outStaging);
  if (d3dCtx.fenceEvent)
    CloseHandle(d3dCtx.fenceEvent);

  return 0;
}
