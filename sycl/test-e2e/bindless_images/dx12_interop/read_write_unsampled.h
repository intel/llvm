
#pragma once

#include <iostream>
#include <string>

#include "../helpers/common.hpp"
#include "../helpers/dx_interop_common.hpp"

#include <sycl/ext/oneapi/bindless_images.hpp>

#include <sycl/properties/queue_properties.hpp>

using namespace dx_helpers;
namespace syclexp = sycl::ext::oneapi::experimental;

class DX12SYCLDevice {
public:
  DX12SYCLDevice();

  DX12SYCLDevice(const DX12SYCLDevice &) = delete;
  DX12SYCLDevice &operator=(const DX12SYCLDevice &) = delete;

private:
  void initDX12Device();
  void initDX12CommandList();

public:
  ID3D12Device *getDx12Device() { return m_dx12Device.Get(); }
  ID3D12CommandQueue *getDx12CommandQueue() { return m_dx12CommandQueue.Get(); }
  ID3D12GraphicsCommandList *getDx12CommandList() {
    return m_dx12CommandList.Get();
  }
  sycl::queue &getSyclQueue() { return m_syclQueue; }

  HRESULT resetCommandList() {
    return m_dx12CommandList->Reset(m_dx12CommandAllocator.Get(), nullptr);
  }

private:
  // DX12 Objects
  ComPtr<IDXGIFactory4> m_dx12Factory;
  ComPtr<IDXGIAdapter1> m_dx12HardwareAdapter;
  ComPtr<ID3D12Device> m_dx12Device;
  ComPtr<ID3D12CommandQueue> m_dx12CommandQueue;
  ComPtr<ID3D12GraphicsCommandList> m_dx12CommandList;
  ComPtr<ID3D12CommandAllocator> m_dx12CommandAllocator;

  // SYCL Objects
  sycl::queue m_syclQueue;
  sycl::device m_syclDevice;
};

template <int NDims, typename DType, int NChannels> class DX12InteropTest {
public:
  DX12InteropTest(DX12SYCLDevice &device, sycl::image_channel_type channelType,
                  sycl::range<NDims> globalSize, sycl::range<NDims> localSize);

  ~DX12InteropTest() {}

  void initDX12Resources();
  void cleanupDX12();

  void callSYCLKernel();

  bool validateOutput();

private:
  void waitDX12Fence(DWORD timeoutMilliseconds = INFINITE);
  void populateDX12Texture();
  void importDX12SharedMemoryHandle(size_t allocSize);
  void importDX12SharedSemaphoreHandle();

  // Dimensions of image
  uint32_t m_width;
  uint32_t m_height;
  uint32_t m_depth;
  uint32_t m_numElems;

  std::vector<DType> m_inputData;

  sycl::image_channel_type m_channelType;

  sycl::range<NDims> m_globalSize;
  sycl::range<NDims> m_localSize;

  DX12SYCLDevice &m_device;

  // DX12 Objects
  ComPtr<ID3D12Resource> m_dx12Texture;
  ComPtr<ID3D12Fence> m_dx12Fence;
  HANDLE m_dx12FenceEvent;

  // Shared handles and values
  uint64_t m_sharedFenceValue = 0;
  HANDLE m_sharedMemoryHandle = INVALID_HANDLE_VALUE;
  HANDLE m_sharedSemaphoreHandle = INVALID_HANDLE_VALUE;

  // SYCL Objects
  syclexp::external_mem m_syclExternalMemHandle;
  syclexp::external_semaphore m_syclExternalSemaphoreHandle;
  syclexp::image_mem_handle m_syclImageMemHandle;
  syclexp::unsampled_image_handle m_syclImageHandle;
};
