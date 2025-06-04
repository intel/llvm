// ...

#include <d3d11.h>

#include "../helpers/common.hpp"
#include "../helpers/dx_interop_common.hpp"

#include <sycl/ext/oneapi/bindless_images.hpp>
#include <sycl/properties/queue_properties.hpp>

#include <functional>
#include <limits>
#include <string_view>

#pragma clang diagnostic ignored "-Waddress-of-temporary"

using dx_helpers::dx_version;

struct g_d3d11DeviceState {
  ID3D11Device *device;
  ID3D11DeviceContext *deviceContext;
};

template <int NDims, typename DType, int NChannels>
void runTest(const g_d3d11DeviceState &d3d11DeviceState, sycl::queue syclQueue,
             sycl::image_channel_type channelType,
             const sycl::range<NDims> &globalSize,
             const sycl::range<NDims> &localSize) {
  // ... (Assume device, context, etc. are initialized) ...
  auto *pDevice = d3d11DeviceState.device;
  auto *pDeviceContext = d3d11DeviceState.deviceContext;
  assert(pDevice && pDeviceContext);

  syclexp::image_descriptor syclImageDesc{globalSize, NChannels, channelType};

  // Verify ability to allocate the above image descriptor.
  // E.g. LevelZero does not support `unorm` channel types.
  if (!bindless_helpers::memoryAllocationSupported(
          syclImageDesc, syclexp::image_memory_handle_type::opaque_handle,
          syclQueue)) {
    // We cannot allocate the image memory, skip the test.
    std::cout << "Memory allocation unsupported. Skipping test.\n";
    return;
  }

  // setup the texture dimensions and resource size.
  const UINT texWidth = globalSize[0];
  UINT texHeight = 1;
  UINT texDepth = 1;
  if constexpr (NDims > 1) {
    texHeight = globalSize[1];
    if constexpr (NDims > 2) {
      texDepth = globalSize[2];
    }
  }
  UINT texMipLevels = 0;
  const unsigned int numElems = texWidth * texHeight * texDepth * NChannels;
  // Unfortunately, DX11 does not expose the texture allocatoin information
  // like DX12, so we have to calculate it manually the best we can (no mips).
  const size_t allocationSize = numElems * sizeof(DType);

  // Create a shared texture
  ComPtr<ID3D11Texture2D> pTexture;
  // Initialize the texture description.
  // Default use means we'll use ID3D11DeviceContext::UpdateSubresource to fill
  // the texture data.
  D3D11_TEXTURE2D_DESC texDesc = {
      .Width = texWidth,
      .Height = texHeight,
      .MipLevels = texMipLevels,
      .ArraySize = texDepth,
      .Format = dx_helpers::toDXGIFormat(NChannels, channelType),
      .SampleDesc = {.Count = 1, .Quality = 0},
      .Usage = D3D11_USAGE_DEFAULT,
      .BindFlags = D3D11_BIND_SHADER_RESOURCE,
      .CPUAccessFlags = 0,
      // The below flags are required for DX11 resource handle sharing.
      .MiscFlags = D3D11_RESOURCE_MISC_SHARED_NTHANDLE |
                   D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX};
  HRESULT hr = pDevice->CreateTexture2D(&texDesc, nullptr, &pTexture);
  assert(SUCCEEDED(hr));

  // Create a shared resource.
  ComPtr<ID3D11Resource> pResource;
  hr = pTexture.As(&pResource);
  assert(SUCCEEDED(hr));

  ComPtr<IDXGIResource1> pDXGIResource;
  pResource.As(&pDXGIResource);
  assert(SUCCEEDED(hr));

  HANDLE sharedHandle = nullptr;
  hr = pDXGIResource->CreateSharedHandle(nullptr, 0, 0, &sharedHandle);
  assert(SUCCEEDED(hr));

  // Import the memory from the shared handle into SYCL
#if 1
  syclexp::external_mem_descriptor<syclexp::resource_win32_handle> extMemDesc{
      sharedHandle, syclexp::external_mem_handle_type::win32_nt_dx11_resource,
      allocationSize};

  auto syclExternalMemHandle =
      syclexp::import_external_memory(extMemDesc, syclQueue);

  auto syclImageMemHandle = syclexp::map_external_image_memory(
      syclExternalMemHandle, syclImageDesc, syclQueue);

  auto syclImageHandle =
      syclexp::create_image(syclImageMemHandle, syclImageDesc, syclQueue);
#endif

  // Populate the texture on the CPU
  std::vector<DType> inputData;
  // Initialise the texture data to upload.
  {
    inputData.resize(numElems);
    auto getInputValue = [&](int i) -> DType {
      if constexpr (std::is_integral_v<DType> ||
                    std::is_same_v<DType, sycl::half>)
        i = i % (static_cast<uint64_t>(std::numeric_limits<DType>::max()) / 2);
      return i;
    };
    for (int i = 0; i < inputData.size(); ++i) {
      inputData[i] = getInputValue(i);
    }
  }

  const auto rowPitch = texWidth * sizeof(DType) * NChannels;
  D3D11_BOX dstRegion;
  dstRegion.left = 0;
  dstRegion.right = texWidth;
  dstRegion.top = 0;
  dstRegion.bottom = texHeight;
  dstRegion.front = 0;
  dstRegion.back = 1;
  pDeviceContext->UpdateSubresource(pResource.Get(), 0, &dstRegion,
                                    inputData.data(), rowPitch, 0);

  // Submit the SYCL kernel.
#if 0
  try {
    auto imgHandle = syclImageHandle;
    using VecType = sycl::vec<DType, NChannels>;

    // All we are doing is doubling the value of each pixel in the texture.
    syclQueue.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(
          sycl::nd_range<NDims>{globalSize, localSize},
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
#endif

  // Read-back and Verify
}

int main() {
  UINT dxgiFactoryFlags = 0;
  ComPtr<IDXGIFactory4> factory;
  ThrowIfFailed(CreateDXGIFactory2(dxgiFactoryFlags, IID_PPV_ARGS(&factory)));

  ComPtr<IDXGIAdapter1> hardwareAdapter;
  bool skipIntegrated = true;
  dx_helpers::getDXGIHardwareAdapter<dx_version::DX11>(
      factory.Get(), &hardwareAdapter, skipIntegrated);
  if (!hardwareAdapter) {
    // Request again but this time falling back to any available GPU.
    skipIntegrated = false;
    dx_helpers::getDXGIHardwareAdapter<dx_version::DX11>(
        factory.Get(), &hardwareAdapter, skipIntegrated);
  }
  // At this point we must have a valid DirectX hardware adapter.
  // This will be resolved once LUID device queries are introduced in SYCL, so
  // we can first create a sycl device and match it exactly to the DirectX HW
  // adapter, so we won't need any of the generic heuristics that the above
  // GetHardwareAdapter function implements in order to find a "suitable" GPU.
  // That way we will also be able to control it via ONEAPI_DEVICE_SELECTOR env.
  assert(hardwareAdapter && "Could not find a valid DirectX hardware adapter.");

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
  DXGI_ADAPTER_DESC1 adapterDesc;
  assert(SUCCEEDED(hardwareAdapter->GetDesc1(&adapterDesc)));

  // Creating the SYCL device.
  // Convert the description string to a narrow string
  std::wstring wstrDescription = adapterDesc.Description;
  std::string description{}; // could preallocate with wstrDescription.size()
  std::transform(std::begin(wstrDescription), std::end(wstrDescription),
                 std::back_inserter(description),
                 [](wchar_t c) { return static_cast<char>(c); });
  std::cout << "D3D11 Adapter Name: " << description << std::endl;
  sycl::queue syclQueue([description](const sycl::device &dev) -> int {
    int score{-1};
    // We want a GPU device.
    if (dev.is_gpu()) {
      score = 1000;
    }
    // This heuristic is also very silly, we want LUID as already noted.
    // Also we won't need a custom device selector at that point since we
    // will be creating the sycl device first before the DX one anyways.
    if (std::string name = dev.get_info<sycl::info::device::name>();
        description.find(name) != std::string::npos ||
        name.find(description) != std::string::npos) {
      score += 500;
    }
    return score;
  });
  std::cout << "SYCL Device Name: "
            << syclQueue.get_device().get_info<sycl::info::device::name>()
            << std::endl;

  g_d3d11DeviceState d3d11DeviceState{.device = device.Get(),
                                      .deviceContext = deviceContext.Get()};

  // test 2D textures
  runTest<2, uint32_t, 1>(
      d3d11DeviceState, syclQueue, sycl::image_channel_type::unsigned_int32,
      sycl::range<2>{256, 256}, sycl::range<2>{16, 16});

  return 0;
}
