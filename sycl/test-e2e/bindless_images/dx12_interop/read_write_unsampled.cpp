// REQUIRES: cuda
// REQUIRES: windows

// RUN: %{build} -l d3d12 -l dxgi -l dxguid -o %t.out
// RUN: %t.out

#pragma clang diagnostic ignored "-Waddress-of-temporary"

#include "read_write_unsampled.h"

void DX12InteropTest::initDX12Device() {
  // Create DXGI factory.
  ThrowIfFailed(CreateDXGIFactory2(0 /* dxgiFactoryFlags */,
                                   IID_PPV_ARGS(&m_dx12Factory)));

  // Get the hardware adapter for a suitable GPU.
  getDX12Adapter(m_dx12Factory.Get(), &m_dx12HardwareAdapter);

  // Create a device from our hardware adapter.
  ThrowIfFailed(D3D12CreateDevice(m_dx12HardwareAdapter.Get(),
                                  D3D_FEATURE_LEVEL_12_0,
                                  IID_PPV_ARGS(&m_dx12Device)));
}

void DX12InteropTest::initDX12CommandList() {
  // Describe and create the command queue.
  D3D12_COMMAND_QUEUE_DESC queueDesc = {D3D12_COMMAND_LIST_TYPE_DIRECT, 0,
                                        D3D12_COMMAND_QUEUE_FLAG_NONE, 0};
  ThrowIfFailed(m_dx12Device->CreateCommandQueue(
      &queueDesc, IID_PPV_ARGS(&m_dx12CommandQueue)));

  // Create the command allocator.
  ThrowIfFailed(m_dx12Device->CreateCommandAllocator(
      D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_dx12CommandAllocator)));

  // Create the command list.
  ThrowIfFailed(m_dx12Device->CreateCommandList(
      0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_dx12CommandAllocator.Get(), NULL,
      IID_PPV_ARGS(&m_dx12CommandList)));
}

void DX12InteropTest::initDX12Resources() {

  // Define default heap properties.
  D3D12_HEAP_PROPERTIES defaultHeapProperties = {};
  defaultHeapProperties.Type = D3D12_HEAP_TYPE_DEFAULT;
  defaultHeapProperties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
  defaultHeapProperties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
  defaultHeapProperties.CreationNodeMask = 1;
  defaultHeapProperties.VisibleNodeMask = 1;

  // Define texture resource descriptor (1D, 32-bit integer).
  D3D12_RESOURCE_DESC textureResourceDesc = {};
  textureResourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE1D;
  textureResourceDesc.Alignment = 0;
  textureResourceDesc.Width = m_width;
  textureResourceDesc.Height = 1;
  textureResourceDesc.DepthOrArraySize = 1;
  textureResourceDesc.MipLevels = 0;
  textureResourceDesc.Format = DXGI_FORMAT_R32_UINT;
  textureResourceDesc.SampleDesc = DXGI_SAMPLE_DESC{1, 0};
  textureResourceDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
  textureResourceDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

  // Create the DX12 texture.
  ThrowIfFailed(m_dx12Device->CreateCommittedResource(
      &defaultHeapProperties, D3D12_HEAP_FLAG_SHARED, &textureResourceDesc,
      D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&m_dx12Texture)));

  // Create a shared handle for our texture.
  ThrowIfFailed(m_dx12Device->CreateSharedHandle(m_dx12Texture.Get(), nullptr,
                                                 GENERIC_ALL, nullptr,
                                                 &m_sharedMemoryHandle));

  D3D12_RESOURCE_ALLOCATION_INFO textureAllocationInfo;
  textureAllocationInfo =
      m_dx12Device->GetResourceAllocationInfo(1, 1, &textureResourceDesc);
  size_t allocationSize = textureAllocationInfo.SizeInBytes;

  // Import our shared DX12 texture resource to SYCL.
  importDX12SharedMemoryHandle(allocationSize);

  // Create the DX12 fence and map to a SYCL semaphore.
  ThrowIfFailed(m_dx12Device->CreateFence(
      m_sharedFenceValue, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&m_dx12Fence)));

  ThrowIfFailed(m_dx12Device->CreateSharedHandle(m_dx12Fence.Get(), nullptr,
                                                 GENERIC_ALL, nullptr,
                                                 &m_sharedSemaphoreHandle));

  // Import our shared DX12 fence resource to SYCL.
  importDX12SharedSemaphoreHandle();

  // Create an event handle to use for synchronization.
  m_dx12FenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
  if (m_dx12FenceEvent == nullptr) {
    ThrowIfFailed(HRESULT_FROM_WIN32(GetLastError()));
  }

  populateDX12Texture();
}

void DX12InteropTest::importDX12SharedMemoryHandle(size_t allocationSize) {
  syclexp::external_mem_descriptor<syclexp::resource_win32_handle> extMemDesc{
      m_sharedMemoryHandle,
      syclexp::external_mem_handle_type::win32_nt_dx12_resource,
      allocationSize};

  m_syclInteropMemHandle =
      syclexp::import_external_memory(extMemDesc, m_syclQueue);

  m_syclImageMemHandle = syclexp::map_external_image_memory(
      m_syclInteropMemHandle, m_syclImageDesc, m_syclQueue);

  m_syclImageHandle =
      syclexp::create_image(m_syclImageMemHandle, m_syclImageDesc, m_syclQueue);
}

void DX12InteropTest::importDX12SharedSemaphoreHandle() {
  syclexp::external_semaphore_descriptor<syclexp::resource_win32_handle>
      extSemDesc{m_sharedSemaphoreHandle,
                 syclexp::external_semaphore_handle_type::win32_nt_dx12_fence};

  m_syclInteropSemaphoreHandle =
      syclexp::import_external_semaphore(extSemDesc, m_syclQueue);
}

void DX12InteropTest::callSYCLKernel() {

  // Wait for imported semaphore. This semaphore was signalled at the
  // end of `populateDX12Texture`.
  m_syclQueue.ext_oneapi_wait_external_semaphore(m_syclInteropSemaphoreHandle,
                                                 m_sharedFenceValue);

  // We can't capture the image handle through `this` in the lambda.
  // If we do the kernel will crash.
  auto imgHandle = m_syclImageHandle;

  // Submit our SYCL kernel. All we do is double the value of each pixel in the
  // texture.
  try {
    m_syclQueue.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<class TestKernel>(
          sycl::nd_range<1>{{m_width}, {1}}, [=](sycl::nd_item<1> it) {
            size_t dim0 = it.get_global_id(0);

            uint32_t px = syclexp::fetch_image<uint32_t>(imgHandle, int(dim0));

            px *= 2;

            syclexp::write_image(imgHandle, int(dim0), px);
          });
    });
  } catch (sycl::exception e) {
    std::cerr << "\tKernel submission failed! " << e.what() << std::endl;
    exit(-1);
  } catch (...) {
    std::cerr << "\tKernel submission failed!" << std::endl;
    exit(-1);
  }

  // Increment the fence value.
  m_sharedFenceValue++;

  // Signal imported semaphore.
  m_syclQueue.submit([&](sycl::handler &cgh) {
    cgh.ext_oneapi_signal_external_semaphore(m_syclInteropSemaphoreHandle,
                                             m_sharedFenceValue);
  });

  // Use DX12 to wait for the semaphore signalled by SYCL above.
  waitDX12Fence();
}

void DX12InteropTest::populateDX12Texture() {

  // Set our texture data to upload.
  std::vector<uint32_t> uploadData(m_width);
  for (int i = 0; i < m_width; ++i) {
    uploadData[i] = i;
  }

  // Get required staging buffer size.
  uint64_t stagingBufferSize = 0;
  m_dx12Device->GetCopyableFootprints(&m_dx12Texture->GetDesc(), 0, 1, 0,
                                      nullptr, nullptr, nullptr,
                                      &stagingBufferSize);

  // Define upload heap properties.
  D3D12_HEAP_PROPERTIES uploadHeapProperties = {};
  uploadHeapProperties.Type = D3D12_HEAP_TYPE_UPLOAD;
  uploadHeapProperties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
  uploadHeapProperties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
  uploadHeapProperties.CreationNodeMask = 1;
  uploadHeapProperties.VisibleNodeMask = 1;

  // Define upload buffer resource descriptor.
  D3D12_RESOURCE_DESC uploadBufferResourceDesc = {};
  uploadBufferResourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  uploadBufferResourceDesc.Alignment = 0;
  uploadBufferResourceDesc.Width = stagingBufferSize;
  uploadBufferResourceDesc.Height = 1;
  uploadBufferResourceDesc.DepthOrArraySize = 1;
  uploadBufferResourceDesc.MipLevels = 1;
  uploadBufferResourceDesc.Format = DXGI_FORMAT_UNKNOWN;
  uploadBufferResourceDesc.SampleDesc = DXGI_SAMPLE_DESC{1, 0};
  uploadBufferResourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  uploadBufferResourceDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

  // Allocate the staging upload buffer.
  ComPtr<ID3D12Resource> stagingBuffer;
  ThrowIfFailed(m_dx12Device->CreateCommittedResource(
      &uploadHeapProperties, D3D12_HEAP_FLAG_NONE, &uploadBufferResourceDesc,
      D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
      IID_PPV_ARGS(&stagingBuffer)));

  // Map the upload staging buffer to host visible memory.
  D3D12_RANGE stagingBufferRange{0, stagingBufferSize};
  uint32_t *pStagingBufferData{};
  ThrowIfFailed(stagingBuffer->Map(
      0, &stagingBufferRange, reinterpret_cast<void **>(&pStagingBufferData)));

  // Populate the staging buffer with our upload data.
  for (int i = 0; i < m_width; ++i) {
    pStagingBufferData[i] = uploadData[i];
  }

  // Unmap the staging buffer.
  D3D12_RANGE emptyRange{0, 0};
  stagingBuffer->Unmap(0, &emptyRange);

  // Reset command list to inital state if necessary.
  m_dx12CommandList->Reset(m_dx12CommandAllocator.Get(), nullptr);

  // Set the copy source and destination footprint/locations.
  D3D12_PLACED_SUBRESOURCE_FOOTPRINT bufferFootprint = {};
  bufferFootprint.Footprint.Width = m_width;
  bufferFootprint.Footprint.Height = 1;
  bufferFootprint.Footprint.Depth = 1;
  bufferFootprint.Footprint.RowPitch = static_cast<uint32_t>(stagingBufferSize);
  bufferFootprint.Footprint.Format = DXGI_FORMAT_R32_UINT;

  D3D12_TEXTURE_COPY_LOCATION copyDest = {};
  copyDest.pResource = m_dx12Texture.Get();
  copyDest.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  copyDest.SubresourceIndex = 0;

  D3D12_TEXTURE_COPY_LOCATION copySrc = {};
  copySrc.pResource = stagingBuffer.Get();
  copySrc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
  copySrc.PlacedFootprint = bufferFootprint;

  // Copy the upload buffer data to our texture.
  m_dx12CommandList->CopyTextureRegion(&copyDest, 0, 0, 0, &copySrc, nullptr);

  D3D12_RESOURCE_BARRIER transitionResourceBarrier = {};
  transitionResourceBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  transitionResourceBarrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
  transitionResourceBarrier.Transition.pResource = m_dx12Texture.Get();
  transitionResourceBarrier.Transition.StateBefore =
      D3D12_RESOURCE_STATE_COPY_DEST;
  transitionResourceBarrier.Transition.StateAfter =
      D3D12_RESOURCE_STATE_COPY_SOURCE;
  transitionResourceBarrier.Transition.Subresource =
      D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

  m_dx12CommandList->ResourceBarrier(1, &transitionResourceBarrier);

  // Execute the command list.
  ThrowIfFailed(m_dx12CommandList->Close());
  ID3D12CommandList *ppCommandLists[] = {m_dx12CommandList.Get()};
  m_dx12CommandQueue->ExecuteCommandLists(_countof(ppCommandLists),
                                          ppCommandLists);
  ThrowIfFailed(
      m_dx12CommandQueue->Signal(m_dx12Fence.Get(), m_sharedFenceValue));

  // Don't wait for the fence here. We will use the SYCL API to wait for this
  // fence in `callSYCLKernel`.
}

bool DX12InteropTest::validateOutput() {

  // Reset the command list.
  ThrowIfFailed(
      m_dx12CommandList->Reset(m_dx12CommandAllocator.Get(), nullptr));

  // Get intermediate readback buffer size.
  uint64_t readbackBufferSize = 0;
  m_dx12Device->GetCopyableFootprints(&m_dx12Texture->GetDesc(), 0, 1, 0,
                                      nullptr, nullptr, nullptr,
                                      &readbackBufferSize);

  // Define readback heap properties.
  D3D12_HEAP_PROPERTIES readbackHeapProperties = {};
  readbackHeapProperties.Type = D3D12_HEAP_TYPE_READBACK;
  readbackHeapProperties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
  readbackHeapProperties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
  readbackHeapProperties.CreationNodeMask = 1;
  readbackHeapProperties.VisibleNodeMask = 1;

  // Define readback buffer resource descriptor.
  D3D12_RESOURCE_DESC readbackBufferResourceDesc = {};
  readbackBufferResourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  readbackBufferResourceDesc.Alignment = 0;
  readbackBufferResourceDesc.Width = readbackBufferSize;
  readbackBufferResourceDesc.Height = 1;
  readbackBufferResourceDesc.DepthOrArraySize = 1;
  readbackBufferResourceDesc.MipLevels = 1;
  readbackBufferResourceDesc.Format = DXGI_FORMAT_UNKNOWN;
  readbackBufferResourceDesc.SampleDesc = DXGI_SAMPLE_DESC{1, 0};
  readbackBufferResourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  readbackBufferResourceDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

  // Create the readback buffer.
  ComPtr<ID3D12Resource> readbackBuffer;
  ThrowIfFailed(m_dx12Device->CreateCommittedResource(
      &readbackHeapProperties, D3D12_HEAP_FLAG_NONE,
      &readbackBufferResourceDesc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
      IID_PPV_ARGS(&readbackBuffer)));

  // Set the copy source and destination footprint/locations.
  D3D12_PLACED_SUBRESOURCE_FOOTPRINT bufferFootprint = {};
  bufferFootprint.Footprint.Width = m_width;
  bufferFootprint.Footprint.Height = 1;
  bufferFootprint.Footprint.Depth = 1;
  bufferFootprint.Footprint.RowPitch =
      static_cast<uint32_t>(readbackBufferSize);
  bufferFootprint.Footprint.Format = DXGI_FORMAT_R32_UINT;

  D3D12_TEXTURE_COPY_LOCATION copyDest = {};
  copyDest.pResource = readbackBuffer.Get();
  copyDest.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
  copyDest.PlacedFootprint = bufferFootprint;

  D3D12_TEXTURE_COPY_LOCATION copySrc = {};
  copySrc.pResource = m_dx12Texture.Get();
  copySrc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  copySrc.SubresourceIndex = 0;

  // Copy the texture to our readback buffer.
  m_dx12CommandList->CopyTextureRegion(&copyDest, 0, 0, 0, &copySrc, nullptr);

  // Execute the command list.
  ThrowIfFailed(m_dx12CommandList->Close());
  ID3D12CommandList *ppCommandLists[] = {m_dx12CommandList.Get()};
  m_dx12CommandQueue->ExecuteCommandLists(_countof(ppCommandLists),
                                          ppCommandLists);
  ThrowIfFailed(
      m_dx12CommandQueue->Signal(m_dx12Fence.Get(), m_sharedFenceValue));

  // Wait for the command list to finish execution and increment the fence
  // value.
  waitDX12Fence();
  m_sharedFenceValue++;

  // Map the readback buffer to host visible memory.
  D3D12_RANGE readbackBufferRange{0, m_width};
  uint32_t *pReadbackBufferData{};
  ThrowIfFailed(
      readbackBuffer->Map(0, &readbackBufferRange,
                          reinterpret_cast<void **>(&pReadbackBufferData)));

  // Wait for the GPU. Sometimes the Mapped memory isn't immediately visible to
  // the host
  ThrowIfFailed(
      m_dx12CommandQueue->Signal(m_dx12Fence.Get(), m_sharedFenceValue));
  waitDX12Fence();
  m_sharedFenceValue++;

  // Read back the updated texture data and validate it.
  bool validated = true;
  for (int i = 0; i < m_width; ++i) {
    bool mismatch = false;
    auto expected = i * 2;
    auto actual = pReadbackBufferData[i];

    if (actual != expected) {
      mismatch = true;
      validated = false;
    }

    if (mismatch) {
#ifdef VERBOSE_PRINT
      std::cout << "Result mismatch at " << i << "! Expected: " << expected
                << ", Actual: " << actual << std::endl;
#else
      break;
#endif
    }
  }

  // Unmap the readback buffer.
  D3D12_RANGE emptyRange{0, 0};
  readbackBuffer->Unmap(0, &emptyRange);

  // Signal the fence to wait upon before we can clean up DX12 later.
  ThrowIfFailed(
      m_dx12CommandQueue->Signal(m_dx12Fence.Get(), m_sharedFenceValue));

  return validated;
}

void DX12InteropTest::waitDX12Fence(DWORD timeoutMilliseconds) {
  // Check the current value of the fence to check if
  // GPU has finished executing the command list.
  if (m_dx12Fence->GetCompletedValue() < m_sharedFenceValue) {
    // If not, set value fence is to set on completion.
    ThrowIfFailed(m_dx12Fence->SetEventOnCompletion(m_sharedFenceValue,
                                                    m_dx12FenceEvent));
    // Wait for fence to be triggered.
    WaitForSingleObject(m_dx12FenceEvent, timeoutMilliseconds);
  }
}

void DX12InteropTest::cleanupDX12() {
  // Wait for the command list to finish execution.
  waitDX12Fence();

  // Clean up opened handles
  CloseHandle(m_sharedSemaphoreHandle);
  CloseHandle(m_sharedMemoryHandle);
  CloseHandle(m_dx12FenceEvent);

  // ComPtr handles will be destroyed automatically.
}

void DX12InteropTest::getDX12Adapter(IDXGIFactory2 *pFactory,
                                     IDXGIAdapter1 **ppAdapter) {
  ComPtr<IDXGIAdapter1> adapter;
  *ppAdapter = nullptr;

  // Find a suitable hardware adapter.
  uint32_t adapterIndex = 0;
  HRESULT adapterFound = pFactory->EnumAdapters1(adapterIndex, &adapter);
  while (adapterFound != DXGI_ERROR_NOT_FOUND) {
    DXGI_ADAPTER_DESC1 desc;
    adapter->GetDesc1(&desc);

    if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
      // We don't want a software adapter.
      continue;
    }

    // Check to see if the adapter supports Direct3D 12, but don't create the
    // actual device yet.
    if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0,
                                    _uuidof(ID3D12Device), nullptr))) {
      break;
    }

    // Increment adapter index and find the next adapter.
    adapterIndex++;
    pFactory->EnumAdapters1(adapterIndex, &adapter);
  }

  // Set the returned adapter.
  *ppAdapter = adapter.Detach();
}

int main() {

  bool validated = false;

  DX12InteropTest interopTestInstance(1024);
  interopTestInstance.initDX12Device();
  interopTestInstance.initDX12CommandList();
  interopTestInstance.initDX12Resources();
  interopTestInstance.callSYCLKernel();
  validated = interopTestInstance.validateOutput();
  interopTestInstance.cleanupDX12();

  if (validated) {
    std::cout << "Test passed!" << std::endl;
    return 0;
  }

  std::cerr << "Test failed!" << std::endl;
  return 1;
}
