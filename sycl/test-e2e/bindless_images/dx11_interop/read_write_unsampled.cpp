// REQUIRES: aspect-ext_oneapi_external_memory_import
// REQUIRES: windows

// UNSUPPORTED: gpu-intel-gen12
// UNSUPPORTED-INTENDED: Unknown issue with integrated GPU failing
//                       when importing memory

// UNSUPPORTED: gpu-intel-dg2
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/21159

// XFAIL: windows && arch-intel_gpu_bmg_g21
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/20384

// RUN: %{build} %link-directx -o %t.out
// RUN: %{run-unfiltered-devices} env NEOReadDebugKeys=1 UseBindlessMode=1 UseExternalAllocatorForSshAndDsh=1 %t.out

#include "dx11_interop.h"

#include <sycl/ext/oneapi/bindless_images.hpp>

#include <d3d11_3.h>

#include <limits>

using namespace dx11_interop;
namespace syclexp = sycl::ext::oneapi::experimental;

// This is a global counter to keep track of the number of verified tests.
static int TotalNumVerifiedTests = 0;

template <typename DType, int NChannels>
void populateD3D11Texture(D3D11ProgramState &d3d11ProgramState,
                          ID3D11Resource *pResource, uint32_t width,
                          uint32_t height, uint32_t depth, DXGI_FORMAT format,
                          const DType *inputData, IDXGIKeyedMutex *keyedMutex) {
  assert(d3d11ProgramState.deviceContext);
  assert(keyedMutex);
  // There are more efficient ways than using UpdateSubresource (ie
  // Map/Unmap). However, this test application is not a realtime
  // performance-critical one, so this is good enough for our needs since we
  // aren't calling it in a loop.
  D3D11_BOX dstRegion;
  dstRegion.left = 0;
  dstRegion.right = width;
  dstRegion.top = 0;
  dstRegion.bottom = height;
  dstRegion.front = 0;
  dstRegion.back = 1;
  ThrowIfFailed(keyedMutex->AcquireSync(d3d11ProgramState.key++, INFINITE));
  const UINT rowPitch = width * NChannels * sizeof(DType);
  const UINT depthPitch = height * rowPitch;
  ID3D11DeviceContext *deviceContext = d3d11ProgramState.deviceContext;
  deviceContext->UpdateSubresource(pResource, 0, &dstRegion,
                                   static_cast<const void *>(inputData),
                                   rowPitch, depthPitch);
  ThrowIfFailed(keyedMutex->ReleaseSync(d3d11ProgramState.key));
}

syclexp::unsampled_image_handle
syclImportTextureMem(HANDLE sharedHandle, size_t allocationSize,
                     const syclexp::image_descriptor &syclImageDesc,
                     sycl::queue syclQueue) {
  // Import the memory from the shared handle into SYCL
  syclexp::external_mem_descriptor<syclexp::resource_win32_handle> extMemDesc{
      sharedHandle, syclexp::external_mem_handle_type::win32_nt_dx11_resource,
      allocationSize};

  auto syclExternalMemHandle =
      syclexp::import_external_memory(extMemDesc, syclQueue);

  auto syclImageMemHandle = syclexp::map_external_image_memory(
      syclExternalMemHandle, syclImageDesc, syclQueue);

  syclexp::unsampled_image_handle syclImageHandle =
      syclexp::create_image(syclImageMemHandle, syclImageDesc, syclQueue);
  return syclImageHandle;
}

template <int NDims, typename DType, int NChannels>
void callSyclKernel(sycl::queue syclQueue,
                    syclexp::unsampled_image_handle syclImageHandle,
                    const sycl::range<NDims> &globalSize,
                    const sycl::range<NDims> &localSize) {
  try {
    syclexp::unsampled_image_handle imgHandle = syclImageHandle;
    using VecType = sycl::vec<DType, NChannels>;

    // All we are doing is doubling the value of each pixel in the texture.
    syclQueue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for(
              sycl::nd_range<NDims>{globalSize, localSize},
              [=](sycl::nd_item<NDims> it) {
                if constexpr (NDims == 3) {
                  size_t dim0 = it.get_global_id(0);
                  size_t dim1 = it.get_global_id(1);
                  size_t dim2 = it.get_global_id(2);
                  // We simulate 3d textures through very tall 2D textures where
                  // the depth dimension has been collapsed onto the height
                  // dimension.
                  // So, logically speaking, the texture has
                  // dimensions Width x Height x Depth but practically speaking,
                  // it is a 2D texture with dimensions Width x (Height *
                  // Depth). So the calculation below globalSize[1] * dim2 +
                  // dim1 simply does this conversion from a 3D index to a 2D
                  // index.
                  auto px = syclexp::fetch_image<
                      std::conditional_t<NChannels == 1, DType, VecType>>(
                      imgHandle, sycl::int2(dim0, globalSize[1] * dim2 + dim1));
                  px *= static_cast<DType>(2);
                  syclexp::write_image(
                      imgHandle, sycl::int2(dim0, globalSize[1] * dim2 + dim1),
                      px);
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
        })
        .wait_and_throw();
    // Instead of wait_and_throw here, we may want to import and use the
    // ID3D11Fence interface to synchronize the SYCL queue with the D3D11
    // device by signaling the completion of the work and waiting for it on
    // the D3D11 side. I haven't implemented it here, but it is a good idea
    // to do this when testing a future ID3D11Fence interop implementation.
  } catch (sycl::exception e) {
    std::cerr << "\tSYCL kernel submission error: " << e.what() << std::endl;
  } catch (...) {
    std::cerr << "\tSYCL kernel submission error." << std::endl;
  }
}

template <typename DType, int NChannels>
bool verifyResult(D3D11ProgramState &d3d11ProgramState,
                  ID3D11Resource *pResource,
                  const D3D11_TEXTURE2D_DESC1 &texDesc, const DType *inputData,
                  IDXGIKeyedMutex *keyedMutex) {
  assert(d3d11ProgramState.device && d3d11ProgramState.deviceContext);
  auto *pDevice = d3d11ProgramState.device;
  auto *pDeviceContext = d3d11ProgramState.deviceContext;

  ComPtr<ID3D11Device3> device3;
  ThrowIfFailed(pDevice->QueryInterface(IID_PPV_ARGS(&device3)));

  static constexpr UINT bindFlags = 0;
  static constexpr UINT miscFlags = 0;

  // Create the staging texture
  D3D11_TEXTURE2D_DESC1 stagingDesc = texDesc;
  stagingDesc.Usage = D3D11_USAGE_STAGING;
  stagingDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
  stagingDesc.BindFlags = bindFlags;
  stagingDesc.MiscFlags = miscFlags;
  ComPtr<ID3D11Texture2D1> stagingTexture;
  ThrowIfFailed(
      device3->CreateTexture2D1(&stagingDesc, nullptr, &stagingTexture));

  // Copy the texture subresource
  ThrowIfFailed(keyedMutex->AcquireSync(d3d11ProgramState.key++, INFINITE));
  pDeviceContext->CopyResource(stagingTexture.Get(), pResource);
  ThrowIfFailed(keyedMutex->ReleaseSync(d3d11ProgramState.key));

  // Map the staging texture to CPU memory
  D3D11_MAPPED_SUBRESOURCE mappedResource;
  ZeroMemory(&mappedResource, sizeof(mappedResource));
  ThrowIfFailed(pDeviceContext->Map(stagingTexture.Get(), 0, D3D11_MAP_READ, 0,
                                    &mappedResource));
  auto bufferData = reinterpret_cast<DType *>(mappedResource.pData);
  const uint32_t bufferLength = texDesc.Width * texDesc.Height * NChannels;
  const uint32_t rowPitch = mappedResource.RowPitch / sizeof(DType);
  const uint32_t elemsPerRow = texDesc.Width * NChannels;
  bool mismatch = false;
  uint32_t bufferOffset = 0;
  for (uint32_t i = 0, bufferIndex = 0; i < bufferLength; ++i, ++bufferIndex) {
    if (i != 0 && (i % elemsPerRow == 0)) {
      bufferOffset += rowPitch;
      // reset the buffer index to start from the offsetted location
      bufferIndex = bufferOffset;
    }
    auto value = bufferData[bufferIndex];
    auto expected = inputData[i] * 2;
    if (value != expected) {
      mismatch = true;
#ifdef VERBOSE_PRINT
      std::cerr << value << " not matching " << expected << "\n";
      std::cerr << "Pixel value[" << bufferIndex << "] = "
                << static_cast<std::conditional_t<
                       std::is_integral_v<decltype(value)>, int, float>>(value)
                << "\n";
      std::cerr << "Expected value[" << i << "] = "
                << static_cast<std::conditional_t<
                       std::is_integral_v<decltype(expected)>, int, float>>(
                       expected)
                << "\n";

      break;
#endif
    }
  }
  // Unmap the staging texture
  pDeviceContext->Unmap(stagingTexture.Get(), 0);

  return !mismatch;
}

/// @brief Runner for the DX11-SYCL memory interopability functionality.
/// @return 0 on success and 1 on failure
template <int NDims, typename DType, int NChannels>
int runTest(D3D11ProgramState &d3d11ProgramState, sycl::queue syclQueue,
            sycl::image_channel_type channelType,
            const sycl::range<NDims> &globalSize,
            const sycl::range<NDims> &localSize) {
  assert(d3d11ProgramState.device && d3d11ProgramState.deviceContext);
  auto *pDevice = d3d11ProgramState.device;
  auto *pDeviceContext = d3d11ProgramState.deviceContext;

  syclexp::image_descriptor syclImageDesc{globalSize, NChannels, channelType};
  // Verify ability to allocate the above image descriptor.
  // E.g. LevelZero does not support `unorm` channel types.
  if (!bindless_helpers::memoryAllocationSupported(
          syclImageDesc, syclexp::image_memory_handle_type::opaque_handle,
          syclQueue) ||
      (channelType == sycl::image_channel_type::unorm_int8 &&
       syclQueue.get_device().get_backend() ==
           sycl::backend::ext_oneapi_level_zero)) {
    // We cannot allocate the image memory, skip the test.
#ifdef VERBOSE_PRINT
    std::cout << "Memory allocation unsupported. Skipping test.\n";
#endif
    // Early-exit successfully since this is not an error.
    return 0;
  }

  // setup the texture dimensions and resource size.
  const uint32_t texWidth = globalSize[0];
  const uint32_t texHeight = (NDims > 1) ? globalSize[1] : 1;
  const uint32_t texDepth = (NDims > 2) ? globalSize[2] : 1;

  DXGI_FORMAT texFormat = toDXGIFormat(NChannels, channelType);

  // DirectX 11 does not allow us to specify a row major layout for 2D textures
  // that have ArraySize > 1 and we would like to specify it in order to
  // accurately calculate the allocation size for the texture so that we can
  // import it from SYCL side. Hence, in light of this restriction, instea of
  // using ArraySize > 1 to simulate 3D textures, we simulate them by simply
  // collapsing the depth dimension onto the height dimension and set ArraySize
  // to 1. Create a shared texture
  ComPtr<ID3D11Texture2D1> texture;
  // Initialize the texture description.
  D3D11_TEXTURE2D_DESC1 texDesc{};
  texDesc.Width = texWidth;
  texDesc.Height =
      texHeight * texDepth; // if height is 1, we can mimic sharing 1D mem
  texDesc.MipLevels = 1;    // one mip level, so no sub-textures
  texDesc.ArraySize = 1;
  texDesc.Format = texFormat;
  texDesc.SampleDesc = {.Count = 1, .Quality = 0};
  texDesc.Usage = D3D11_USAGE_DEFAULT;
  texDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
  texDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE | D3D11_CPU_ACCESS_READ;
  // Note: Direct3D 11 does not support the
  // D3D11_RESOURCE_MISC_SHARED_NTHANDLE flag for 3D or 1D textures. This flag
  // is mainly used for sharing resources between different D3D11 devices, but
  // it is only applicable to 2D textures.
  texDesc.MiscFlags = D3D11_RESOURCE_MISC_SHARED_NTHANDLE |
                      D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX;
  texDesc.TextureLayout = D3D11_TEXTURE_LAYOUT_ROW_MAJOR;

  ComPtr<ID3D11Device3> device3;
  pDevice->QueryInterface(IID_PPV_ARGS(&device3));
  device3->CreateTexture2D1(&texDesc, NULL, &texture);

  // Create the keyed mutex for synchronising the shared resource.
  ComPtr<IDXGIKeyedMutex> keyedMutex;
  ThrowIfFailed(texture.As(&keyedMutex));
  d3d11ProgramState.key = 0;

  // Create an NT handle to a shared resource referring to our texture.
  // Opening the shared resource gives access to it for use on the SYCL device.
  ComPtr<IDXGIResource1> sharedResource;
  ThrowIfFailed(texture.As(&sharedResource));
  HANDLE sharedHandle = nullptr;
  ThrowIfFailed(sharedResource->CreateSharedHandle(
      nullptr, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE, nullptr,
      &sharedHandle));

  // Obtain a pointer to the shared resource for use in subsequent operations.
  ThrowIfFailed(device3->OpenSharedResource1(
      sharedHandle, IID_PPV_ARGS(sharedResource.GetAddressOf())));

  // Populate the texture on the CPU
  std::vector<DType> inputData(texWidth * texHeight * texDepth * NChannels, 0);
  if (ComPtr<ID3D11Resource> resource; SUCCEEDED(texture.As(&resource))) {
    // Initialize the texture data to upload.
    auto getInputValue = [&](int i) -> DType {
      if constexpr (std::is_integral_v<DType> ||
                    std::is_same_v<DType, sycl::half>) {
        i = i % (static_cast<uint64_t>(std::numeric_limits<DType>::max()) / 2);
      }
      return i;
    };
    for (int i = 0; i < inputData.size(); ++i) {
      inputData[i] = getInputValue(i);
    }
    populateD3D11Texture<DType, NChannels>(
        d3d11ProgramState, resource.Get(), texWidth, texHeight * texDepth, 1,
        texFormat, inputData.data(), keyedMutex.Get());
  }

  // Unfortunately, DX11 does not expose the texture allocation information
  // like DX12, so we have to calculate it manually the best we can (no mips).
  // The fact that the texture has been requested to have a row major layout
  // should support this speculative calculation.
  const size_t allocationSize =
      texWidth * texHeight * texDepth * NChannels * sizeof(DType);
  syclexp::unsampled_image_handle syclImageHandle = syclImportTextureMem(
      sharedHandle, allocationSize, syclImageDesc, syclQueue);

  // Submit the SYCL kernel.
  // When IDXGIKeyedMutex importing into SYCL is implemented, we'll be able to
  // call it from the SYCL API. All it does is ensuring only one device has
  // exclusive access.
  ThrowIfFailed(keyedMutex->AcquireSync(d3d11ProgramState.key++, INFINITE));
  callSyclKernel<NDims, DType, NChannels>(syclQueue, syclImageHandle,
                                          globalSize, localSize);
  // Back to the D3D11 process
  ThrowIfFailed(keyedMutex->ReleaseSync(d3d11ProgramState.key));

  // Read-back and verify
  int errc = 1;
  if (ComPtr<ID3D11Resource> resource; SUCCEEDED(texture.As(&resource))) {
    if (verifyResult<DType, NChannels>(d3d11ProgramState, resource.Get(),
                                       texDesc, inputData.data(),
                                       keyedMutex.Get())) {
      errc = 0;
    }
  }

  // cleanup of the shared handle.
  CloseNTHandle(sharedHandle);

#ifdef VERBOSE_PRINT
  if (errc == 1) {
    std::cerr << "\tTest failed: NDims " << NDims << " NChannels " << NChannels
              << " image_channel_type "
              << bindless_helpers::channelTypeToString(channelType) << "\n";
  } else {
    std::cout << "\tTest passed: NDims " << NDims << " NChannels " << NChannels
              << " image_channel_type "
              << bindless_helpers::channelTypeToString(channelType) << "\n";
  }
#endif
  TotalNumVerifiedTests++;
  return errc;
}

int main() {
  // Create SYCL queue, relying on SYCL device selection
  sycl::queue syclQueue;
  sycl::device syclDevice = syclQueue.get_device();

  // Initialize D3D11 and create DX11 programs state from the SYCL device
  D3D11ProgramState d3d11ProgramState{syclDevice};

  int errors = 0;

  // Test 1D texture interop
#ifdef TEST_SMALL_IMAGE_SIZE
  const sycl::range<1> globalSize1D{1024};
#else
  const sycl::range<1> globalSize1D{4096};
#endif
  errors += runTest<1, uint32_t, 1>(d3d11ProgramState, syclQueue,
                                    sycl::image_channel_type::unsigned_int32,
                                    globalSize1D, sycl::range{256});
  errors += runTest<1, uint8_t, 4>(d3d11ProgramState, syclQueue,
                                   sycl::image_channel_type::unorm_int8,
                                   globalSize1D, sycl::range{256});
  errors += runTest<1, float, 1>(d3d11ProgramState, syclQueue,
                                 sycl::image_channel_type::fp32, globalSize1D,
                                 sycl::range{256});
  errors += runTest<1, sycl::half, 2>(d3d11ProgramState, syclQueue,
                                      sycl::image_channel_type::fp16,
                                      globalSize1D, sycl::range{256});
  errors += runTest<1, sycl::half, 4>(d3d11ProgramState, syclQueue,
                                      sycl::image_channel_type::fp16,
                                      globalSize1D, sycl::range{256});

  // Test 2D texture interop
#ifdef TEST_SMALL_IMAGE_SIZE
  const sycl::range<2> globalSize2D[] = {
      sycl::range{64, 64}, sycl::range{64, 64}, sycl::range{64, 64},
      sycl::range{64, 64}, sycl::range{64, 64}};
#else
  const sycl::range<2> globalSize2D[] = {
      sycl::range{1024, 1024}, sycl::range{1920, 1080}, sycl::range{1920, 1080},
      sycl::range{1280, 720}, sycl::range{1280, 720}};
#endif
  errors += runTest<2, uint32_t, 1>(d3d11ProgramState, syclQueue,
                                    sycl::image_channel_type::unsigned_int32,
                                    globalSize2D[0], sycl::range{16, 16});
  errors += runTest<2, uint8_t, 4>(d3d11ProgramState, syclQueue,
                                   sycl::image_channel_type::unorm_int8,
                                   globalSize2D[1], sycl::range{16, 8});
  errors += runTest<2, float, 1>(d3d11ProgramState, syclQueue,
                                 sycl::image_channel_type::fp32,
                                 globalSize2D[2], sycl::range{16, 8});
  errors += runTest<2, sycl::half, 2>(d3d11ProgramState, syclQueue,
                                      sycl::image_channel_type::fp16,
                                      globalSize2D[3], sycl::range{16, 16});
  errors += runTest<2, sycl::half, 4>(d3d11ProgramState, syclQueue,
                                      sycl::image_channel_type::fp16,
                                      globalSize2D[4], sycl::range{16, 16});

// Test 3D texture interop
#ifdef TEST_SMALL_IMAGE_SIZE
  const sycl::range<3> globalSize3D[] = {
      sycl::range{64, 16, 4}, sycl::range{64, 16, 4}, sycl::range{64, 64, 4},
      sycl::range{64, 64, 4}, sycl::range{64, 64, 4}};
#else
  const sycl::range<3> globalSize3D[] = {
      sycl::range{256, 256, 32}, sycl::range{1920, 1080, 8},
      sycl::range{512, 256, 8}, sycl::range{1280, 720, 2},
      sycl::range{1280, 720, 2}};
#endif
  errors += runTest<3, uint32_t, 1>(d3d11ProgramState, syclQueue,
                                    sycl::image_channel_type::unsigned_int32,
                                    globalSize3D[0], sycl::range{16, 16, 1});
  errors += runTest<3, uint8_t, 4>(d3d11ProgramState, syclQueue,
                                   sycl::image_channel_type::unorm_int8,
                                   globalSize3D[1], sycl::range{16, 8, 2});
  errors += runTest<3, float, 1>(d3d11ProgramState, syclQueue,
                                 sycl::image_channel_type::fp32,
                                 globalSize3D[2], sycl::range{16, 8, 1});
  errors += runTest<3, sycl::half, 2>(d3d11ProgramState, syclQueue,
                                      sycl::image_channel_type::fp16,
                                      globalSize3D[3], sycl::range{16, 16, 1});
  errors += runTest<3, sycl::half, 4>(d3d11ProgramState, syclQueue,
                                      sycl::image_channel_type::fp16,
                                      globalSize3D[4], sycl::range{16, 16, 1});
  
#ifdef VERBOSE_PRINT
  std::string deviceName = syclDevice.get_info<sycl::info::device::name>();
  std::cout << "Tests pass rate for SYCL device: " << deviceName << "\n";
  const auto numPassedTests = (TotalNumVerifiedTests - errors);
  std::cerr << ((errors > 0) ? errors : numPassedTests) << " out of "
            << TotalNumVerifiedTests << " tested configurations were "
            << ((errors > 0) ? "unsuccessful" : "successful") << ".\n";
#endif

  return errors;
}
