#ifndef DX11_INTEROP_H
#define DX11_INTEROP_H

#pragma clang diagnostic ignored "-Waddress-of-temporary"

#include "../helpers/common.hpp"
#include "../helpers/dx_interop_common.hpp"

#include <sycl/detail/core.hpp>

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

  // Keyed mutex ID for synchronizing access to the shared resource per device.
  std::atomic<UINT64> key;

  D3D11ProgramState() = default;
  ~D3D11ProgramState();
};

D3D11ProgramState::~D3D11ProgramState() {
  if (device)
    device->Release();
  if (deviceContext)
    deviceContext->Release();
}

void initializeD3D11(D3D11ProgramState *d3d11ProgramState) {
  assert(d3d11ProgramState);

  UINT dxgiFactoryFlags = 0;
#if WITH_DX_DEBUG
  dxgiFactoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
#endif
  ComPtr<IDXGIFactory3> factory;
  ThrowIfFailed(CreateDXGIFactory2(dxgiFactoryFlags, IID_PPV_ARGS(&factory)));

  ComPtr<IDXGIAdapter1> hardwareAdapter;
#if defined(PREFER_INTEGRATED_GPU)
  constexpr auto devicePreference = device_preference::Integrated;
#else
  constexpr auto devicePreference = device_preference::Dedicated;
#endif
  getDXGIHardwareAdapter<dx_version::DX11>(factory.Get(), &hardwareAdapter,
                                           devicePreference);
  if (!hardwareAdapter && devicePreference != device_preference::Unspecified) {
    // Request again but this time falling back to any available GPU.
    getDXGIHardwareAdapter<dx_version::DX11>(factory.Get(), &hardwareAdapter);
  }
  // At this point we must have a valid DirectX hardware adapter.
  // This will be resolved once LUID device queries are introduced in SYCL, so
  // we can first create a sycl device and match it exactly to the DirectX HW
  // adapter, so we won't need any of the generic heuristics that the above
  // GetHardwareAdapter function implements in order to find a "suitable" GPU.
  // That way we will also be able to control it via ONEAPI_DEVICE_SELECTOR
  // env.
  assert(hardwareAdapter && "Invalid DirectX hardware adapter.");

  // Creating the D3D11 device.
  ComPtr<ID3D11Device> device = nullptr;
  ComPtr<ID3D11DeviceContext> deviceContext = nullptr;
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

  d3d11ProgramState->device = device.Detach();
  d3d11ProgramState->deviceContext = deviceContext.Detach();

  // Convert the description string to a narrow string.
  // This conversion is a little imperfect due to consideration for locale etc.
  std::string description{};
  std::wstring wstrDescription = adapterDesc.Description;
  std::transform(std::begin(wstrDescription), std::end(wstrDescription),
                 std::back_inserter(description),
                 [](wchar_t c) { return static_cast<char>(c); });
  d3d11ProgramState->deviceName = std::move(description);
}

inline sycl::device
getSyclDeviceFromDX11(const D3D11ProgramState *d3d11ProgramState) {
  assert(d3d11ProgramState);
  return sycl::device(
      [deviceName = d3d11ProgramState->deviceName](const sycl::device &dev) {
        int score{-1};
        // We want a GPU device.
        if (dev.is_gpu()) {
          score = 1000;
        }

        // This heuristic, comparing names, is a little silly, ideally we want
        // device LUID matching. If we create the SYCL device first we will not
        // need the selector and instead compare LUIDs on D3D11 device creation.
        if (const std::string name = dev.get_info<sycl::info::device::name>();
            deviceName.find(name) != std::string::npos ||
            name.find(deviceName) != std::string::npos) {
          score += 500;
        }
        return score;
      });
}

} // namespace dx11_interop

#endif // DX11_INTEROP_H
