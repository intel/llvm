
#pragma once

// Reduce the size of Win32 header files
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

// Define NOMINMAX to enable compilation on Windows
#define NOMINMAX

#include <windows.h>

#include <d3d12.h>
#include <dxgi1_4.h>

#include <iostream>
#include <string>
#include <wrl.h>

#include <sycl/ext/oneapi/bindless_images.hpp>

#include <sycl/properties/queue_properties.hpp>

using Microsoft::WRL::ComPtr;
namespace syclexp = sycl::ext::oneapi::experimental;

inline std::string ResultToString(HRESULT result) {
  char s_str[64] = {};
  sprintf_s(s_str, "Error result == 0x%08X", static_cast<uint32_t>(result));
  return std::string(s_str);
}

inline void ThrowIfFailed(HRESULT result) {
  if (result != S_OK) {
    throw std::runtime_error(ResultToString(result));
  }
}

class DX12InteropTest {
public:
  DX12InteropTest(uint32_t width)
      : m_width(width), m_sharedFenceValue(1),
        m_syclImageDesc({m_width}, 1,
                        sycl::image_channel_type::unsigned_int32) {
    m_syclQueue =
        sycl::queue{m_syclDevice, {sycl::property::queue::in_order{}}};
  }
  ~DX12InteropTest() {}

  void initDX12Device();
  void initDX12CommandList();
  void initDX12Resources();
  void cleanupDX12();

  void callSYCLKernel();

  bool validateOutput();

private:
  void waitDX12Fence(DWORD timeoutMilliseconds = INFINITE);
  void populateDX12Texture();
  void getDX12Adapter(IDXGIFactory2 *pFactory, IDXGIAdapter1 **ppAdapter);
  void importDX12SharedMemoryHandle(size_t allocSize);
  void importDX12SharedSemaphoreHandle();

  // Dimensions of image
  uint32_t m_width;

  // DX12 Objects
  ComPtr<IDXGIFactory4> m_dx12Factory;
  ComPtr<IDXGIAdapter1> m_dx12HardwareAdapter;
  ComPtr<ID3D12Device> m_dx12Device;
  ComPtr<ID3D12CommandQueue> m_dx12CommandQueue;
  ComPtr<ID3D12GraphicsCommandList> m_dx12CommandList;
  ComPtr<ID3D12CommandAllocator> m_dx12CommandAllocator;
  ComPtr<ID3D12Resource> m_dx12Texture;
  ComPtr<ID3D12Fence> m_dx12Fence;
  HANDLE m_dx12FenceEvent;

  // Shared handles and values
  uint64_t m_sharedFenceValue;
  HANDLE m_sharedMemoryHandle;
  HANDLE m_sharedSemaphoreHandle;

  // SYCL Objects
  sycl::queue m_syclQueue;
  sycl::device m_syclDevice;
  syclexp::image_descriptor m_syclImageDesc;
  syclexp::interop_mem_handle m_syclInteropMemHandle;
  syclexp::interop_semaphore_handle m_syclInteropSemaphoreHandle;
  syclexp::image_mem_handle m_syclImageMemHandle;
  syclexp::unsampled_image_handle m_syclImageHandle;
};
