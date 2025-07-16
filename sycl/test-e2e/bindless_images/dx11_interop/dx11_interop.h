#ifndef DX11_INTEROP_H
#define DX11_INTEROP_H

#pragma clang diagnostic ignored "-Waddress-of-temporary"

#include "../helpers/common.hpp"
#include "../helpers/dx_interop_common.hpp"

#include <sycl/detail/core.hpp>
#include <sycl/device.hpp>

#include <atomic>
#include <string_view>

using namespace dx_helpers;

namespace dx11_interop {

/// @brief
struct D3D11ProgramState {
  // device management
  ID3D11Device *device{nullptr};
  ID3D11DeviceContext *deviceContext{nullptr};

  // Temporary, this is to be replaced by LUID.
  // Can also store a DXGI_ADAPTER_DESC if more state is needed.
  std::string deviceName;

  // Keyed mutex ID for synchronizing access to the shared resource.
  std::atomic<UINT64> key;

  D3D11ProgramState(const sycl::device &syclDevice);
  ~D3D11ProgramState();
};

D3D11ProgramState::D3D11ProgramState(const sycl::device &syclDevice) {
  UINT dxgiFactoryFlags = 0;
#if WITH_DX_DEBUG
  dxgiFactoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
#endif
  ComPtr<IDXGIFactory3> factory;
  ThrowIfFailed(CreateDXGIFactory2(dxgiFactoryFlags, IID_PPV_ARGS(&factory)));

  auto hardwareAdapter = getDXGIHardwareAdapter<dx_version::DX11>(
      factory.Get(), syclDevice.get_info<sycl::info::device::name>());
  assert(hardwareAdapter && "Invalid DirectX hardware adapter.");

  // Creating the D3D11 device.
  UINT deviceFlags = 0;
#if defined(D3D_DEVICE_DEBUG)
  deviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif
  constexpr D3D_FEATURE_LEVEL requestedFeatureLevel = D3D_FEATURE_LEVEL_11_0;
  D3D_FEATURE_LEVEL featureLevel;
  ThrowIfFailed(D3D11CreateDevice(hardwareAdapter.Get(),
                                  D3D_DRIVER_TYPE_UNKNOWN, nullptr, deviceFlags,
                                  &requestedFeatureLevel, 1, D3D11_SDK_VERSION,
                                  &device, &featureLevel, &deviceContext));

  // Get the description of the adapter which contains the LUID, etc.
  DXGI_ADAPTER_DESC1 adapterDesc;
  ThrowIfFailed(hardwareAdapter->GetDesc1(&adapterDesc));

  deviceName = getD3DDeviceName(adapterDesc);
}

D3D11ProgramState::~D3D11ProgramState() {
  if (device)
    device->Release();
  if (deviceContext)
    deviceContext->Release();
}

} // namespace dx11_interop

#endif // DX11_INTEROP_H
