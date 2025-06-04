#pragma once

// Reduce the size of Win32 header files
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

// Define NOMINMAX to enable compilation on Windows
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>

// Windows Runtime C++ Template Library (ComPtr and friends)
#include <wrl.h>

#include <d3d11.h>
#include <d3d12.h>

// As we are to share common DXGI interface functionality between our DX 11 and
// DX 12 tests, we include at least dxgi1_4 (introduced with Direct3D 12) as a
// minimum requirement for both. If OS version supports it, try to use dxgi1_6.
// DXGI 1.6 is included in Windows 10 as part of the Creators Update (0x0A00).
#define HAS_DXGI_1_6 (_WIN32_WINNT >= WINVER_WIN10)
#if HAS_DXGI_1_6
#include <dxgi1_6.h>
#else
#include <dxgi1_4.h>
#endif

#include <dxgidebug.h>

#include <sycl/detail/core.hpp>

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

namespace dx_helpers {

enum class dx_version { DX11, DX12 };

DXGI_FORMAT toDXGIFormat(int NChannels, sycl::image_channel_type channelType) {
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

/// @brief  This is a helper function to find an appropriate hardware adapter.
/// It does not create the final D3D device but rather tests creating it with
/// the chosen hardware adapter, so it can be used safely for SYCL interop.
/// This function has no thread-safety guarantees in its current state.
/// Ideally we will not be searching for the highest performing adapter but for
/// the one that specifically matches the SYCL device for interopability. For
/// that reason we will need to introduce an LUID device query extension first.
/// @tparam dxVer
/// @param pFactory
/// @param ppAdapter
template <dx_version dxVer>
void getDXGIHardwareAdapter(IDXGIFactory1 *pFactory, IDXGIAdapter1 **ppAdapter,
                        bool skipIntegrated = false) {
  *ppAdapter = nullptr;

  ComPtr<IDXGIAdapter1> adapter;

  constexpr D3D_FEATURE_LEVEL minFeatureLevel = D3D_FEATURE_LEVEL_11_0;

  bool foundAdapter{false};
  // Find a suitable hardware adapter.
#if HAS_DXGI_1_6
  ComPtr<IDXGIFactory6> factory6;
  if (SUCCEEDED(pFactory->QueryInterface(IID_PPV_ARGS(&factory6)))) {
    for (UINT adapterIndex = 0; SUCCEEDED(factory6->EnumAdapterByGpuPreference(
             adapterIndex,
             skipIntegrated ? DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE
                            : DXGI_GPU_PREFERENCE_UNSPECIFIED,
             IID_PPV_ARGS(&adapter)));
         ++adapterIndex) {
      if (!adapter) {
        continue;
      }

      DXGI_ADAPTER_DESC1 desc;
      if (FAILED(adapter->GetDesc1(&desc))) {
        continue;
      }

      // We don't want the Microsoft Basic Render Driver (software) adapter.
      if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
        continue;
      }

      if constexpr (dxVer == dx_version::DX12) {
        // Check to see whether the adapter supports Direct3D 12, but don't
        // create the actual device yet.
        if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                                        _uuidof(ID3D12Device), nullptr))) {
          foundAdapter = true;
          break;
        }
      } else {
        // Check to see if the adapter supports Direct3D 11, but don't create
        // the actual device yet.
        ComPtr<ID3D11Device> device = nullptr;
        ComPtr<ID3D11DeviceContext> deviceContext = nullptr;
        UINT deviceFlags = 0;
#if defined(D3D_DEVICE_DEBUG)
        deviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif
        D3D_FEATURE_LEVEL outFeatureLevel;
        // We don't want SW renderer. Ideally we specify
        // D3D_DRIVER_TYPE_HARDWARE on creation but when we specify an Adapter
        // we need to specify the D3D_DRIVER_TYPE_UNKNOWN enum, otherwise the
        // call fails. This is okay as we made sure to select the best possible
        // HW adapter. We can create Device without a DXGIAdapter but we want
        // one to exist, so we can use it for LUID matching with the SYCL device
        // for interop.
        HRESULT Result = D3D11CreateDevice(
            adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, deviceFlags,
            &minFeatureLevel, 1u, D3D11_SDK_VERSION, &device, &outFeatureLevel,
            &deviceContext);
        if (SUCCEEDED(Result)) {
          assert(outFeatureLevel == minFeatureLevel); // sanity check
          std::cout << "Created D3D11 Device successfully.\n";
          foundAdapter = true;
          break;
        }
      }
    }
    // If not found fallback to EnumAdapters1 to search all available HW
    // adapters.
  }
#endif

  // Find a suitable hardware adapter.
  if (!foundAdapter) {
    for (UINT adapterIndex = 0;
         pFactory->EnumAdapters1(adapterIndex, adapter.GetAddressOf()) !=
         DXGI_ERROR_NOT_FOUND;
         ++adapterIndex) {
      if (!adapter) {
        continue;
      }

      DXGI_ADAPTER_DESC1 desc;
      if (FAILED(adapter->GetDesc1(&desc))) {
        continue;
      }

      // We don't want the Microsoft Basic Render Driver (software) adapter.
      if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
        continue;
      }

      if (skipIntegrated) {
        // Simple heuristic, but it's hard to do better without external deps.
        ComPtr<IDXGIAdapter3> tmpAdapter3;
        DXGI_QUERY_VIDEO_MEMORY_INFO videoMemoryInfo;
        bool isNonLocalMemoryPresent{false};
        if (SUCCEEDED(adapter.As(&tmpAdapter3)) && tmpAdapter3 &&
            SUCCEEDED(tmpAdapter3->QueryVideoMemoryInfo(
                0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &videoMemoryInfo))) {
          isNonLocalMemoryPresent = videoMemoryInfo.Budget != 0;
        }
        // Integrated GPUs will dedicated the non-local (i.e. system RAM)
        // memory as local (i.e. VRAM-like), so from this we can potentially
        // derive that  if  if the non-local memory segment is 0 then it's an
        // intergrated graphics card that is using/allocating it as local.
        // In the case of dedicated GPUs both segments will be independent.
        if (bool isIntegrated = !isNonLocalMemoryPresent; isIntegrated) {
          continue;
        }
      }

      if constexpr (dxVer == dx_version::DX12) {
        // Check to see if the adapter supports Direct3D 12, but don't create
        // the actual device yet. All Direct3D 12 drivers support at least
        // feature level 11_0, making it a safe and reliable choice for device
        // creation. It is also perfectly sufficient for our interopability
        // feature testing purposes.
        if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), minFeatureLevel,
                                        _uuidof(ID3D12Device), nullptr))) {
          foundAdapter = true;
          break;
        }
      } else {
        // Check to see if the adapter supports Direct3D 11, but don't create
        // the actual device yet.
        ComPtr<ID3D11Device> device = nullptr;
        ComPtr<ID3D11DeviceContext> deviceContext = nullptr;
        UINT deviceFlags = 0;
#if defined(D3D_DEVICE_DEBUG)
        deviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif
        D3D_FEATURE_LEVEL outFeatureLevel;
        // We don't want SW renderer. Ideally we specify
        // D3D_DRIVER_TYPE_HARDWARE on creation but when we specify an Adapter
        // we need to specify the D3D_DRIVER_TYPE_UNKNOWN enum, otherwise the
        // call fails. This is okay as we made sure to select the best possible
        // HW adapter. We can create Device without a DXGIAdapter but we want
        // one to exist, so we can use it for LUID matching with the SYCL device
        // for interop.
        HRESULT Result = D3D11CreateDevice(
            adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, deviceFlags,
            &minFeatureLevel, 1u, D3D11_SDK_VERSION, &device, &outFeatureLevel,
            &deviceContext);
        if (SUCCEEDED(Result)) {
          assert(outFeatureLevel == minFeatureLevel); // sanity check
          std::cout << "Created D3D11 Device successfully.\n";
          foundAdapter = true;
          break;
        }
      }
    }
  }

  if (foundAdapter) {
    // Set the returned adapter.
    *ppAdapter = adapter.Detach();
  } else {
    std::cerr << "Could not find the requested DirectX hardware adapter.";
  }
}

} // namespace dx_heleprs
