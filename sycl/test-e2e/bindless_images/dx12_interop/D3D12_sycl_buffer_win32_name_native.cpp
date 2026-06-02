// REQUIRES: aspect-ext_oneapi_external_memory_import
// REQUIRES: aspect-ext_oneapi_external_semaphore_import
// REQUIRES: windows

// XFAIL: windows
// XFAIL-TRACKER: GSD-12837

// RUN: %{build} %link-directx -o %t.exe %if target-spir %{ -Wno-ignored-attributes %}
// RUN: %{run} %t.exe --no-sem
// RUN: %{run} %t.exe

// clang-format off
/*
  DirectX 12 / SYCL Buffer + Fence (Timeline) Interop Test - resource_win32_name

  clang++.exe -fsycl -o dsbwnn.exe D3D12_sycl_buffer_win32_name_native.cpp -ld3d12 -ldxgi -ld3dcompiler

  Tests native resource_win32_name support in SYCL. The NT object name
  is passed through SYCL -> UR -> L0, which opens the named object on
  the caller's behalf via ze_external_memory_import_win32_handle_t::name.

  FLAGS: --no-sem        Don't use semaphores for SYCL/D3D12 synchronization
         --iterations N  Number of iterations to run (default: 10)
         --size M        Number of uint32_t elements in the buffer (default: 1024)
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

struct D3D12NamedBuffer {
  Microsoft::WRL::ComPtr<ID3D12Resource> resource;
  std::wstring name;
  // Must keep at least one handle open for the name to persist.
  HANDLE keepAliveHandle;
};

struct D3D12NamedFence {
  Microsoft::WRL::ComPtr<ID3D12Fence> fence;
  std::wstring name;
  HANDLE keepAliveHandle;
};

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

  hr = ctx.device->CreateSharedHandle(result.resource.Get(), nullptr,
                                      GENERIC_ALL, name,
                                      &result.keepAliveHandle);
  if (FAILED(hr)) {
    throw std::runtime_error("Failed to create named shared handle");
  }

  std::wcout << L"[D3D12] Created named buffer: " << name << std::endl;

  return result;
}

D3D12NamedFence createNamedExportableFence(D3D12Context &ctx,
                                           const wchar_t *name) {
  D3D12NamedFence result;
  result.name = name;

  HRESULT hr = ctx.device->CreateFence(0, D3D12_FENCE_FLAG_SHARED,
                                       IID_PPV_ARGS(&result.fence));
  if (FAILED(hr)) {
    throw std::runtime_error("Failed to create fence");
  }

  hr = ctx.device->CreateSharedHandle(result.fence.Get(), nullptr, GENERIC_ALL,
                                      name, &result.keepAliveHandle);
  if (FAILED(hr)) {
    throw std::runtime_error("Failed to create named shared fence handle");
  }

  std::wcout << L"[D3D12] Created named fence: " << name << std::endl;

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

  D3D12Context d3dCtx = createD3D12Context();

  D3D12NamedBuffer inBuf = createNamedExportableBuffer(
      d3dCtx, bufferSize, L"Global\\SYCLTestInputBuffer3");
  D3D12NamedBuffer outBuf = createNamedExportableBuffer(
      d3dCtx, bufferSize, L"Global\\SYCLTestOutputBuffer3");

  D3D12BufferResources inStaging = createUploadBuffer(d3dCtx, bufferSize);
  D3D12BufferResources outStaging = createReadbackBuffer(d3dCtx, bufferSize);

  D3D12NamedFence extFence;
  if (useSemaphores) {
    extFence = createNamedExportableFence(d3dCtx, L"Global\\SYCLTestFence3");
  }

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

  try {
    sycl::queue q;
    auto device = q.get_device();
    auto context = q.get_context();

    std::cout << "[SYCL] Device: "
              << device.get_info<sycl::info::device::name>() << std::endl;

    std::cout << "[SYCL] Importing input buffer by name (native support)"
              << std::endl;
    syclexp::external_mem_descriptor<syclexp::resource_win32_name> inDesc{
        {(const void *)inBuf.name.c_str()},
        syclexp::external_mem_handle_type::win32_nt_handle,
        bufferSize};
    syclexp::external_mem inExtMem =
        syclexp::import_external_memory(inDesc, device, context);

    std::cout << "[SYCL] Importing output buffer by name (native support)"
              << std::endl;
    syclexp::external_mem_descriptor<syclexp::resource_win32_name> outDesc{
        {(const void *)outBuf.name.c_str()},
        syclexp::external_mem_handle_type::win32_nt_handle,
        bufferSize};
    syclexp::external_mem outExtMem =
        syclexp::import_external_memory(outDesc, device, context);

    syclexp::external_semaphore syclSem{};
    if (useSemaphores) {
      std::cout << "[SYCL] Importing fence by name (native support)"
                << std::endl;
      auto semDesc =
          syclexp::external_semaphore_descriptor<syclexp::resource_win32_name>{
              {(const void *)extFence.name.c_str()},
              syclexp::external_semaphore_handle_type::win32_nt_dx12_fence};
      syclSem = syclexp::import_external_semaphore(semDesc, device, context);
    }

    uint32_t *inPtr = static_cast<uint32_t *>(
        syclexp::map_external_linear_memory(inExtMem, 0, bufferSize, q));
    uint32_t *outPtr = static_cast<uint32_t *>(
        syclexp::map_external_linear_memory(outExtMem, 0, bufferSize, q));

    std::cout << "[Test] Starting " << iterations << " iteration test..."
              << std::endl;

    for (int i = 1; i <= iterations; ++i) {
      uint64_t d3dSignalVal = (uint64_t)(2 * i - 1);
      uint64_t syclSignalVal = (uint64_t)(2 * i);

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

      d3dCtx.fenceValue++;
      d3dCtx.cmdQueue->Signal(d3dCtx.fence.Get(), d3dCtx.fenceValue);
      d3dCtx.fence->SetEventOnCompletion(d3dCtx.fenceValue, d3dCtx.fenceEvent);
      WaitForSingleObject(d3dCtx.fenceEvent, INFINITE);

      std::cout << "  [" << i << "] D3D12 upload done" << std::endl;

      if (useSemaphores) {
        std::cout << ", SYCL sem-wait(" << d3dSignalVal << ")..." << std::endl;
        q.ext_oneapi_wait_external_semaphore(syclSem, d3dSignalVal);
        std::cout << "ok" << std::endl;
      }

      q.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::range<1>(numElements), [=](sycl::item<1> item) {
          size_t id = item.get_id(0);
          outPtr[id] = inPtr[id] * 2;
        });
      });

      if (useSemaphores) {
        std::cout << ", SYCL sem-signal(" << syclSignalVal << ")" << std::endl;
        q.ext_oneapi_signal_external_semaphore(syclSem, syclSignalVal);
        std::cout << "ok" << std::endl;
      }
      q.wait();
      std::cout << ", SYCL done" << std::endl;

      if (useSemaphores) {
        d3dCtx.cmdQueue->Wait(extFence.fence.Get(), syclSignalVal);
      }

      d3dCtx.cmdAlloc->Reset();
      d3dCtx.cmdList->Reset(d3dCtx.cmdAlloc.Get(), nullptr);

      barrier.Transition.pResource = outBuf.resource.Get();
      barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
      barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
      d3dCtx.cmdList->ResourceBarrier(1, &barrier);

      d3dCtx.cmdList->CopyBufferRegion(outStaging.resource.Get(), 0,
                                       outBuf.resource.Get(), 0, bufferSize);

      barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
      barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
      d3dCtx.cmdList->ResourceBarrier(1, &barrier);

      d3dCtx.cmdList->Close();
      d3dCtx.cmdQueue->ExecuteCommandLists(1, ppCommandLists);

      std::cout << ", d3d-fence..." << std::endl;
      d3dCtx.fenceValue++;
      d3dCtx.cmdQueue->Signal(d3dCtx.fence.Get(), d3dCtx.fenceValue);
      d3dCtx.fence->SetEventOnCompletion(d3dCtx.fenceValue, d3dCtx.fenceEvent);

      if (WaitForSingleObject(d3dCtx.fenceEvent, 5000) == WAIT_TIMEOUT) {
        std::cerr << "\nTIMEOUT on host wait!\n";
        return 1;
      }
      std::cout << "ok" << std::flush;

      outStaging.resource->Map(0, nullptr, &mapped);
      auto *outData = static_cast<uint32_t *>(mapped);
      uint32_t expected = (uint32_t)i * 2;
      int errors = 0;
      for (size_t j = 0; j < numElements; ++j) {
        if (outData[j] != expected) {
          if (errors++ < 5)
            std::cerr << "  [" << j << "]: got " << outData[j] << " expected "
                      << expected << std::endl;
        }
      }
      outStaging.resource->Unmap(0, nullptr);

      if (errors > 0) {
        std::cerr << "\nFAILURE at iteration " << i << ": " << errors
                  << " mismatches" << std::endl;
        return 1;
      }

      d3dCtx.cmdAlloc->Reset();
      d3dCtx.cmdList->Reset(d3dCtx.cmdAlloc.Get(), nullptr);
      barrier.Transition.pResource = inBuf.resource.Get();
      barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
      barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
      d3dCtx.cmdList->ResourceBarrier(1, &barrier);
      d3dCtx.cmdList->Close();
      executeAndWait(d3dCtx);

      if (i % 5 == 0 || i == 1)
        std::cout << " PASS" << std::endl;
      else
        std::cout << " ok" << std::endl;
    }

    std::cout << "SUCCESS! All " << iterations << " iterations passed."
              << std::endl;

    syclexp::unmap_external_linear_memory(inPtr, q);
    syclexp::unmap_external_linear_memory(outPtr, q);
    if (useSemaphores)
      syclexp::release_external_semaphore(syclSem, device, context);
    syclexp::release_external_memory(inExtMem, device, context);
    syclexp::release_external_memory(outExtMem, device, context);

  } catch (sycl::exception &e) {
    std::cerr << "SYCL Exception: " << e.what() << std::endl;
    return 1;
  }

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
