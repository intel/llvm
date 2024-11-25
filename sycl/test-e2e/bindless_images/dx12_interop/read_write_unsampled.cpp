// REQUIRES: aspect-ext_oneapi_bindless_images
// REQUIRES: windows
// REQUIRES: build-and-run-mode

// DEFINE: %{link-flags}=%if cl_options %{ /clang:-ld3d12 /clang:-ldxgi /clang:-ldxguid %} %else %{ -ld3d12 -ldxgi -ldxguid %}
// RUN: %{build} %{link-flags} -o %t.out
// RUN: %{run-unfiltered-devices} env NEOReadDebugKeys=1 UseBindlessMode=1 UseExternalAllocatorForSshAndDsh=1 %t.out

#pragma clang diagnostic ignored "-Waddress-of-temporary"

#include "read_write_unsampled.h"
#include "../helpers/common.hpp"

static DXGI_FORMAT toDXGIFormat(int NChannels,
                                sycl::image_channel_type channelType) {
  switch (channelType) {
  case sycl::image_channel_type::snorm_int8:
    switch (NChannels) {
    case 1:
      return DXGI_FORMAT_R8_SNORM;
    case 2:
      return DXGI_FORMAT_R8G8_SNORM;
    case 4:
      return DXGI_FORMAT_R8G8B8A8_SNORM;
    default:
      break;
    }
  case sycl::image_channel_type::snorm_int16:
    switch (NChannels) {
    case 1:
      return DXGI_FORMAT_R16_SNORM;
    case 2:
      return DXGI_FORMAT_R16G16_SNORM;
    case 4:
      return DXGI_FORMAT_R16G16B16A16_SNORM;
    default:
      break;
    }
  case sycl::image_channel_type::unorm_int8:
    switch (NChannels) {
    case 1:
      return DXGI_FORMAT_R8_UNORM;
    case 2:
      return DXGI_FORMAT_R8G8_UNORM;
    case 4:
      return DXGI_FORMAT_R8G8B8A8_UNORM;
    default:
      break;
    }
  case sycl::image_channel_type::unorm_int16:
    switch (NChannels) {
    case 1:
      return DXGI_FORMAT_R16_UNORM;
    case 2:
      return DXGI_FORMAT_R16G16_UNORM;
    case 4:
      return DXGI_FORMAT_R16G16B16A16_UNORM;
    default:
      break;
    }
  case sycl::image_channel_type::unorm_short_565:
    return DXGI_FORMAT_B5G6R5_UNORM;
  case sycl::image_channel_type::unorm_short_555:
    return DXGI_FORMAT_B5G5R5A1_UNORM;
  case sycl::image_channel_type::unorm_int_101010:
    return DXGI_FORMAT_R10G10B10A2_UNORM;
  case sycl::image_channel_type::signed_int8:
    switch (NChannels) {
    case 1:
      return DXGI_FORMAT_R8_SINT;
    case 2:
      return DXGI_FORMAT_R8G8_SINT;
    case 4:
      return DXGI_FORMAT_R8G8B8A8_SINT;
    default:
      break;
    }
  case sycl::image_channel_type::signed_int16:
    switch (NChannels) {
    case 1:
      return DXGI_FORMAT_R16_SINT;
    case 2:
      return DXGI_FORMAT_R16G16_SINT;
    case 4:
      return DXGI_FORMAT_R16G16B16A16_SINT;
    default:
      break;
    }
  case sycl::image_channel_type::signed_int32:
    switch (NChannels) {
    case 1:
      return DXGI_FORMAT_R32_SINT;
    case 2:
      return DXGI_FORMAT_R32G32_SINT;
    case 4:
      return DXGI_FORMAT_R32G32B32A32_SINT;
    default:
      break;
    }
  case sycl::image_channel_type::unsigned_int8:
    switch (NChannels) {
    case 1:
      return DXGI_FORMAT_R8_UINT;
    case 2:
      return DXGI_FORMAT_R8G8_UINT;
    case 4:
      return DXGI_FORMAT_R8G8B8A8_UINT;
    default:
      break;
    }
  case sycl::image_channel_type::unsigned_int16:
    switch (NChannels) {
    case 1:
      return DXGI_FORMAT_R16_UINT;
    case 2:
      return DXGI_FORMAT_R16G16_UINT;
    case 4:
      return DXGI_FORMAT_R16G16B16A16_UINT;
    default:
      break;
    }
  case sycl::image_channel_type::unsigned_int32:
    switch (NChannels) {
    case 1:
      return DXGI_FORMAT_R32_UINT;
    case 2:
      return DXGI_FORMAT_R32G32_UINT;
    case 4:
      return DXGI_FORMAT_R32G32B32A32_UINT;
    default:
      break;
    }
  case sycl::image_channel_type::fp16:
    switch (NChannels) {
    case 1:
      return DXGI_FORMAT_R16_FLOAT;
    case 2:
      return DXGI_FORMAT_R16G16_FLOAT;
    case 4:
      return DXGI_FORMAT_R16G16B16A16_FLOAT;
    default:
      break;
    }
  case sycl::image_channel_type::fp32:
    switch (NChannels) {
    case 1:
      return DXGI_FORMAT_R32_FLOAT;
    case 2:
      return DXGI_FORMAT_R32G32_FLOAT;
    case 4:
      return DXGI_FORMAT_R32G32B32A32_FLOAT;
    default:
      break;
    }
  default:
    break;
  }
  std::cerr << "Unsupported image_channel_type in toDXGIFormat\n";
  exit(-1);
}

DX12SYCLDevice::DX12SYCLDevice() {
  m_syclQueue = sycl::queue{m_syclDevice, {sycl::property::queue::in_order{}}};
}

void DX12SYCLDevice::initDX12Device() {
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

void DX12SYCLDevice::initDX12CommandList() {
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

void DX12SYCLDevice::getDX12Adapter(IDXGIFactory2 *pFactory,
                                    IDXGIAdapter1 **ppAdapter) {
  ComPtr<IDXGIAdapter1> adapter;
  *ppAdapter = nullptr;

  // Find a suitable hardware adapter.
  uint32_t adapterIndex = 0;
  HRESULT adapterFound = pFactory->EnumAdapters1(adapterIndex, &adapter);
  while (adapterFound != DXGI_ERROR_NOT_FOUND) {
    DXGI_ADAPTER_DESC1 desc;
    adapter->GetDesc1(&desc);

    if (!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) {
      // We don't want a software adapter.

      // Check to see if the adapter supports Direct3D 12, but don't create the
      // actual device yet.
      if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0,
                                      _uuidof(ID3D12Device), nullptr))) {
        break;
      }
    }

    // Increment adapter index and find the next adapter.
    adapterIndex++;
    pFactory->EnumAdapters1(adapterIndex, &adapter);
  }

  // Set the returned adapter.
  *ppAdapter = adapter.Detach();
}

template <int NDims, typename DType, int NChannels>
DX12InteropTest<NDims, DType, NChannels>::DX12InteropTest(
    DX12SYCLDevice &device, sycl::image_channel_type channelType,
    sycl::range<NDims> globalSize, sycl::range<NDims> localSize)
    : m_device(device), m_channelType(channelType), m_globalSize(globalSize),
      m_localSize(localSize) {
  m_width = m_globalSize[0];
  m_height = 1;
  m_depth = 1;
  if constexpr (NDims > 1) {
    m_height = m_globalSize[1];
    if constexpr (NDims > 2)
      m_depth = m_globalSize[2];
  }
  m_numElems = m_width * m_height * m_depth * NChannels;
}

template <int NDims, typename DType, int NChannels>
void DX12InteropTest<NDims, DType, NChannels>::initDX12Resources() {

  // Define default heap properties.
  D3D12_HEAP_PROPERTIES defaultHeapProperties = {};
  defaultHeapProperties.Type = D3D12_HEAP_TYPE_DEFAULT;
  defaultHeapProperties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
  defaultHeapProperties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
  defaultHeapProperties.CreationNodeMask = 1;
  defaultHeapProperties.VisibleNodeMask = 1;

  // Define texture resource descriptor.
  D3D12_RESOURCE_DESC textureResourceDesc = {};
  if constexpr (NDims == 1)
    textureResourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE1D;
  else if constexpr (NDims == 2)
    textureResourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
  else
    textureResourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
  textureResourceDesc.Alignment = 0;
  textureResourceDesc.Width = m_width;
  textureResourceDesc.Height = m_height;
  textureResourceDesc.DepthOrArraySize = m_depth;
  textureResourceDesc.MipLevels = 0;
  textureResourceDesc.Format = toDXGIFormat(NChannels, m_channelType);
  textureResourceDesc.SampleDesc = DXGI_SAMPLE_DESC{1, 0};
  textureResourceDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
  textureResourceDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

  // Create the DX12 texture.
  auto *dx12Device = m_device.getDx12Device();
  ThrowIfFailed(dx12Device->CreateCommittedResource(
      &defaultHeapProperties, D3D12_HEAP_FLAG_SHARED, &textureResourceDesc,
      D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&m_dx12Texture)));

  // Create a shared handle for our texture.
  ThrowIfFailed(dx12Device->CreateSharedHandle(m_dx12Texture.Get(), nullptr,
                                               GENERIC_ALL, nullptr,
                                               &m_sharedMemoryHandle));

  D3D12_RESOURCE_ALLOCATION_INFO textureAllocationInfo;
  textureAllocationInfo =
      dx12Device->GetResourceAllocationInfo(1, 1, &textureResourceDesc);
  size_t allocationSize = textureAllocationInfo.SizeInBytes;

  // Import our shared DX12 texture resource to SYCL.
  importDX12SharedMemoryHandle(allocationSize);

  // Create the DX12 fence and map to a SYCL semaphore.
  ThrowIfFailed(dx12Device->CreateFence(
      m_sharedFenceValue, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&m_dx12Fence)));
  m_sharedFenceValue++;

#ifdef TEST_SEMAPHORE_IMPORT
  ThrowIfFailed(dx12Device->CreateSharedHandle(m_dx12Fence.Get(), nullptr,
                                               GENERIC_ALL, nullptr,
                                               &m_sharedSemaphoreHandle));

  // Import our shared DX12 fence resource to SYCL.
  importDX12SharedSemaphoreHandle();
#endif

  // Create an event handle to use for synchronization.
  m_dx12FenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
  if (m_dx12FenceEvent == nullptr) {
    ThrowIfFailed(HRESULT_FROM_WIN32(GetLastError()));
  }

  populateDX12Texture();
}

template <int NDims, typename DType, int NChannels>
void DX12InteropTest<NDims, DType, NChannels>::importDX12SharedMemoryHandle(
    size_t allocationSize) {
  syclexp::external_mem_descriptor<syclexp::resource_win32_handle> extMemDesc{
      m_sharedMemoryHandle,
      syclexp::external_mem_handle_type::win32_nt_dx12_resource,
      allocationSize};

  auto &syclQueue = m_device.getSyclQueue();
  m_syclExternalMemHandle =
      syclexp::import_external_memory(extMemDesc, syclQueue);

  syclexp::image_descriptor syclImageDesc{m_globalSize, NChannels,
                                          m_channelType};
  m_syclImageMemHandle = syclexp::map_external_image_memory(
      m_syclExternalMemHandle, syclImageDesc, syclQueue);

  m_syclImageHandle =
      syclexp::create_image(m_syclImageMemHandle, syclImageDesc, syclQueue);
}

template <int NDims, typename DType, int NChannels>
void DX12InteropTest<NDims, DType,
                     NChannels>::importDX12SharedSemaphoreHandle() {
  syclexp::external_semaphore_descriptor<syclexp::resource_win32_handle>
      extSemDesc{m_sharedSemaphoreHandle,
                 syclexp::external_semaphore_handle_type::win32_nt_dx12_fence};

  m_syclExternalSemaphoreHandle =
      syclexp::import_external_semaphore(extSemDesc, m_device.getSyclQueue());
}

template <int NDims, typename DType, int NChannels>
void DX12InteropTest<NDims, DType, NChannels>::callSYCLKernel() {
  auto &syclQueue = m_device.getSyclQueue();
#ifdef TEST_SEMAPHORE_IMPORT
  // Wait for imported semaphore. This semaphore was signalled at the
  // end of `populateDX12Texture`.
  syclQueue.ext_oneapi_wait_external_semaphore(m_syclExternalSemaphoreHandle,
                                               m_sharedFenceValue);
#endif

  // We can't capture the image handle through `this` in the lambda.
  // If we do the kernel will crash.
  auto imgHandle = m_syclImageHandle;

  using VecType = sycl::vec<DType, NChannels>;

  // Submit our SYCL kernel. All we do is double the value of each pixel in the
  // texture.
  try {
    syclQueue.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(
          sycl::nd_range<NDims>{m_globalSize, m_localSize},
          [=](sycl::nd_item<NDims> it) {
            if constexpr (NDims == 3) {
              size_t dim0 = it.get_global_id(0);
              size_t dim1 = it.get_global_id(1);
              size_t dim2 = it.get_global_id(2);
              auto px = syclexp::fetch_image<
                  std::conditional_t<NChannels == 1, DType, VecType>>(
                  imgHandle, sycl::int3(dim0, dim1, dim2));
              px *= static_cast<DType>(2);
              syclexp::write_image(imgHandle, sycl::int3(dim0, dim1, dim2), px);
            } else if constexpr (NDims == 2) {
              size_t dim0 = it.get_global_id(0);
              size_t dim1 = it.get_global_id(1);
              auto px = syclexp::fetch_image<
                  std::conditional_t<NChannels == 1, DType, VecType>>(
                  imgHandle, sycl::int2(dim0, dim1));
              px *= static_cast<DType>(2);
              syclexp::write_image(imgHandle, sycl::int2(dim0, dim1), px);
            } else {
              size_t dim0 = it.get_global_id(0);
              auto px = syclexp::fetch_image<
                  std::conditional_t<NChannels == 1, DType, VecType>>(
                  imgHandle, int(dim0));
              px *= static_cast<DType>(2);
              syclexp::write_image(imgHandle, int(dim0), px);
            }
          });
    });
  } catch (sycl::exception e) {
    std::cerr << "\tKernel submission failed! " << e.what() << std::endl;
    exit(-1);
  } catch (...) {
    std::cerr << "\tKernel submission failed!" << std::endl;
    exit(-1);
  }

#ifdef TEST_SEMAPHORE_IMPORT
  // Increment the fence value.
  m_sharedFenceValue++;

  // Signal imported semaphore.
  syclQueue.submit([&](sycl::handler &cgh) {
    cgh.ext_oneapi_signal_external_semaphore(m_syclExternalSemaphoreHandle,
                                             m_sharedFenceValue);
  });

  // Use DX12 to wait for the semaphore signalled by SYCL above.
  waitDX12Fence();
  m_sharedFenceValue++;
#else
  syclQueue.wait();
#endif
}

template <int NDims, typename DType, int NChannels>
void DX12InteropTest<NDims, DType, NChannels>::populateDX12Texture() {

  // Set our texture data to upload.
  m_inputData.resize(m_numElems);
  auto getInputValue = [&](int i) -> DType {
    if constexpr (std::is_integral_v<DType> ||
                  std::is_same_v<DType, sycl::half>)
      i = i % (static_cast<uint64_t>(std::numeric_limits<DType>::max()) / 2);
    return i;
  };
  for (int i = 0; i < m_numElems; ++i) {
    m_inputData[i] = getInputValue(i);
  }

  // Get required staging buffer size.
  uint64_t stagingBufferSize = 0;
  auto *dx12Device = m_device.getDx12Device();
  dx12Device->GetCopyableFootprints(&m_dx12Texture->GetDesc(), 0, 1, 0, nullptr,
                                    nullptr, nullptr, &stagingBufferSize);

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
  ThrowIfFailed(dx12Device->CreateCommittedResource(
      &uploadHeapProperties, D3D12_HEAP_FLAG_NONE, &uploadBufferResourceDesc,
      D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
      IID_PPV_ARGS(&stagingBuffer)));

  // Map the upload staging buffer to host visible memory.
  D3D12_RANGE stagingBufferRange{0, stagingBufferSize};
  DType *pStagingBufferData{};
  ThrowIfFailed(stagingBuffer->Map(
      0, &stagingBufferRange, reinterpret_cast<void **>(&pStagingBufferData)));

  // Populate the staging buffer with our upload data.
  for (int i = 0; i < m_numElems; ++i) {
    pStagingBufferData[i] = m_inputData[i];
  }

  // Unmap the staging buffer.
  D3D12_RANGE emptyRange{0, 0};
  stagingBuffer->Unmap(0, &emptyRange);

  // Reset command list to inital state if necessary.
  std::ignore = m_device.resetCommandList();

  // Set the copy source and destination footprint/locations.
  D3D12_PLACED_SUBRESOURCE_FOOTPRINT bufferFootprint = {};
  bufferFootprint.Footprint.Width = m_width;
  bufferFootprint.Footprint.Height = m_height;
  bufferFootprint.Footprint.Depth = m_depth;
  bufferFootprint.Footprint.RowPitch = m_width * sizeof(DType) * NChannels;
  bufferFootprint.Footprint.Format = toDXGIFormat(NChannels, m_channelType);

  D3D12_TEXTURE_COPY_LOCATION copyDest = {};
  copyDest.pResource = m_dx12Texture.Get();
  copyDest.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  copyDest.SubresourceIndex = 0;

  D3D12_TEXTURE_COPY_LOCATION copySrc = {};
  copySrc.pResource = stagingBuffer.Get();
  copySrc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
  copySrc.PlacedFootprint = bufferFootprint;

  // Copy the upload buffer data to our texture.
  auto *dx12CommandList = m_device.getDx12CommandList();
  dx12CommandList->CopyTextureRegion(&copyDest, 0, 0, 0, &copySrc, nullptr);

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

  dx12CommandList->ResourceBarrier(1, &transitionResourceBarrier);

  // Execute the command list.
  ThrowIfFailed(dx12CommandList->Close());
  ID3D12CommandList *ppCommandLists[] = {dx12CommandList};
  auto *dx12CommandQueue = m_device.getDx12CommandQueue();
  dx12CommandQueue->ExecuteCommandLists(_countof(ppCommandLists),
                                        ppCommandLists);
  ThrowIfFailed(
      dx12CommandQueue->Signal(m_dx12Fence.Get(), m_sharedFenceValue));

#ifdef TEST_SEMAPHORE_IMPORT
  // Don't wait for the fence here. We will use the SYCL API to wait for this
  // fence in `callSYCLKernel`.
#else
  waitDX12Fence();
  m_sharedFenceValue++;
#endif
}

template <int NDims, typename DType, int NChannels>
bool DX12InteropTest<NDims, DType, NChannels>::validateOutput() {

  // Reset the command list.
  ThrowIfFailed(m_device.resetCommandList());

  // Get intermediate readback buffer size.
  uint64_t readbackBufferSize = 0;
  auto *dx12Device = m_device.getDx12Device();
  dx12Device->GetCopyableFootprints(&m_dx12Texture->GetDesc(), 0, 1, 0, nullptr,
                                    nullptr, nullptr, &readbackBufferSize);

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
  ThrowIfFailed(dx12Device->CreateCommittedResource(
      &readbackHeapProperties, D3D12_HEAP_FLAG_NONE,
      &readbackBufferResourceDesc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
      IID_PPV_ARGS(&readbackBuffer)));

  // Set the copy source and destination footprint/locations.
  D3D12_PLACED_SUBRESOURCE_FOOTPRINT bufferFootprint = {};
  bufferFootprint.Footprint.Width = m_width;
  bufferFootprint.Footprint.Height = m_height;
  bufferFootprint.Footprint.Depth = m_depth;
  bufferFootprint.Footprint.RowPitch = m_width * sizeof(DType) * NChannels;
  bufferFootprint.Footprint.Format = toDXGIFormat(NChannels, m_channelType);

  D3D12_TEXTURE_COPY_LOCATION copyDest = {};
  copyDest.pResource = readbackBuffer.Get();
  copyDest.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
  copyDest.PlacedFootprint = bufferFootprint;

  D3D12_TEXTURE_COPY_LOCATION copySrc = {};
  copySrc.pResource = m_dx12Texture.Get();
  copySrc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  copySrc.SubresourceIndex = 0;

  // Copy the texture to our readback buffer.
  auto *dx12CommandList = m_device.getDx12CommandList();
  dx12CommandList->CopyTextureRegion(&copyDest, 0, 0, 0, &copySrc, nullptr);

  // Execute the command list.
  ThrowIfFailed(dx12CommandList->Close());
  ID3D12CommandList *ppCommandLists[] = {dx12CommandList};
  auto *dx12CommandQueue = m_device.getDx12CommandQueue();
  dx12CommandQueue->ExecuteCommandLists(_countof(ppCommandLists),
                                        ppCommandLists);
  ThrowIfFailed(
      dx12CommandQueue->Signal(m_dx12Fence.Get(), m_sharedFenceValue));

  // Wait for the command list to finish execution and increment the fence
  // value.
  waitDX12Fence();
  m_sharedFenceValue++;

  // Map the readback buffer to host visible memory.
  D3D12_RANGE readbackBufferRange{0, readbackBufferSize};
  DType *pReadbackBufferData{};
  ThrowIfFailed(
      readbackBuffer->Map(0, &readbackBufferRange,
                          reinterpret_cast<void **>(&pReadbackBufferData)));

  // Wait for the GPU. Sometimes the Mapped memory isn't immediately visible to
  // the host
  ThrowIfFailed(
      dx12CommandQueue->Signal(m_dx12Fence.Get(), m_sharedFenceValue));
  waitDX12Fence();
  m_sharedFenceValue++;

  // Read back the updated texture data and validate it.
  bool validated = true;
  for (int i = 0; i < m_numElems; ++i) {
    bool mismatch = false;
    auto expected = m_inputData[i] * 2;
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
      dx12CommandQueue->Signal(m_dx12Fence.Get(), m_sharedFenceValue));

  return validated;
}

template <int NDims, typename DType, int NChannels>
void DX12InteropTest<NDims, DType, NChannels>::waitDX12Fence(
    DWORD timeoutMilliseconds) {
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

template <int NDims, typename DType, int NChannels>
void DX12InteropTest<NDims, DType, NChannels>::cleanupDX12() {
  // Wait for the command list to finish execution.
  waitDX12Fence();

  // Clean up opened handles
  if (m_sharedSemaphoreHandle != INVALID_HANDLE_VALUE)
    CloseHandle(m_sharedSemaphoreHandle);
  CloseHandle(m_sharedMemoryHandle);
  CloseHandle(m_dx12FenceEvent);

  // ComPtr handles will be destroyed automatically.
}

template <int NDims, typename DType, int NChannels>
static bool
runTest(DX12SYCLDevice &device, sycl::image_channel_type channelType,
        sycl::range<NDims> globalSize, sycl::range<NDims> localSize) {
  DX12InteropTest<NDims, DType, NChannels> interopTestInstance(
      device, channelType, globalSize, localSize);
  interopTestInstance.initDX12Resources();
  interopTestInstance.callSYCLKernel();
  bool validated = interopTestInstance.validateOutput();
  interopTestInstance.cleanupDX12();

#ifdef VERBOSE_PRINT
  if (!validated) {
    std::cerr << "\tTest failed: NDims " << NDims << " NChannels " << NChannels
              << " image_channel_type "
              << bindless_helpers::channelTypeToString(channelType)
              << ", exiting\n";
    exit(-1);
  } else {
    std::cout << "\tTest passed: NDims " << NDims << " NChannels " << NChannels
              << " image_channel_type "
              << bindless_helpers::channelTypeToString(channelType) << "\n";
  }
#endif

  return validated;
}

int main() {
  DX12SYCLDevice device;
  device.initDX12Device();
  device.initDX12CommandList();

  bool validated = true;

  sycl::range<1> globalSize1{1024};
  sycl::range<1> localSize1{1024};
  validated &=
      runTest<1, uint32_t, 1>(device, sycl::image_channel_type::unsigned_int32,
                              globalSize1, localSize1);
  validated &= runTest<1, uint8_t, 4>(
      device, sycl::image_channel_type::unorm_int8, globalSize1, localSize1);
  validated &= runTest<1, float, 1>(device, sycl::image_channel_type::fp32,
                                    globalSize1, localSize1);
  validated &= runTest<1, sycl::half, 2>(device, sycl::image_channel_type::fp16,
                                         globalSize1, localSize1);
  validated &= runTest<1, sycl::half, 4>(device, sycl::image_channel_type::fp16,
                                         globalSize1, localSize1);

#ifdef TEST_SMALL_IMAGE_SIZE
  sycl::range<2> globalSize2[] = {
      {64, 64}, {64, 64}, {64, 64}, {64, 64}, {64, 64}};
#else
  sycl::range<2> globalSize2[] = {
      {1024, 1024}, {1920, 1080}, {1920, 1080}, {2048, 2048}, {2048, 2048}};
#endif
  validated &=
      runTest<2, uint32_t, 1>(device, sycl::image_channel_type::unsigned_int32,
                              globalSize2[0], {16, 16});
  validated &= runTest<2, uint8_t, 4>(
      device, sycl::image_channel_type::unorm_int8, globalSize2[1], {16, 8});
  validated &= runTest<2, float, 1>(device, sycl::image_channel_type::fp32,
                                    globalSize2[2], {16, 8});
  validated &= runTest<2, sycl::half, 2>(device, sycl::image_channel_type::fp16,
                                         globalSize2[3], {16, 16});
  validated &= runTest<2, sycl::half, 4>(device, sycl::image_channel_type::fp16,
                                         globalSize2[4], {16, 16});

#ifdef TEST_SMALL_IMAGE_SIZE
  sycl::range<3> globalSize3[] = {
      {64, 16, 4}, {64, 16, 4}, {64, 64, 4}, {64, 64, 4}, {64, 64, 4}};
#else
  sycl::range<3> globalSize3[] = {{1024, 1024, 16},
                                  {1920, 1080, 8},
                                  {1920, 1080, 8},
                                  {2048, 2048, 4},
                                  {2048, 2048, 4}};
#endif
  validated &=
      runTest<3, uint32_t, 1>(device, sycl::image_channel_type::unsigned_int32,
                              globalSize3[0], {16, 16, 1});
  validated &= runTest<3, uint8_t, 4>(
      device, sycl::image_channel_type::unorm_int8, globalSize3[1], {16, 8, 2});
  validated &= runTest<3, float, 1>(device, sycl::image_channel_type::fp32,
                                    globalSize3[2], {16, 8, 1});
  validated &= runTest<3, sycl::half, 2>(device, sycl::image_channel_type::fp16,
                                         globalSize3[3], {16, 16, 1});
  validated &= runTest<3, sycl::half, 4>(device, sycl::image_channel_type::fp16,
                                         globalSize3[4], {16, 16, 1});

  if (validated) {
    std::cout << "Test passed!" << std::endl;
    return 0;
  }

  std::cerr << "Test failed!" << std::endl;

  return 1;
}
