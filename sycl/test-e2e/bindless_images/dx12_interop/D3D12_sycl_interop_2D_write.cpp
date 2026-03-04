// REQUIRES: aspect-ext_oneapi_bindless_images
// REQUIRES: aspect-ext_oneapi_external_memory_import
// REQUIRES: windows

// RUN: %{build} -o %t.exe -ld3d12 -ldxgi -ld3dcompiler
// RUN: %{run} %t.exe --type float --channels 4 32x33

// clang-format off
/*
    clang++.exe -fsycl -o ds2w.exe D3D12_sycl_interop_2D_write.cpp -ld3d12 -ldxgi -ld3dcompiler

    FLAGS:
    --sampled      ERROR: Sampled image writes are not supported
    --semaphores   Use DX12 Fences for SYCL Interop Sync
    --channels X   Set number of channels (1, 2, or 4). Default is 4 (RGBA)
    --type XXX     Set data type (float, half, uint32, int32, uint16, int16, uint8, int8, unorm8). 
                   Default is float 
   WxH             Set custom Width x Height (e.g. 8x4)
*/
// clang-format on

// RUN: %{run} %t.out --type float --channels 1 32x33
// RUN: %{run} %t.out --type float --channels 2 32x33
// RUN: %{run} %t.out --type float --channels 4 32x33
// RUN: %{run} %t.out --type half --channels 1 32x33
// RUN: %{run} %t.out --type half --channels 2 32x33
// RUN: %{run} %t.out --type half --channels 4 32x33
// RUN: %{run} %t.out --type int32 --channels 1 32x33
// RUN: %{run} %t.out --type int32 --channels 2 32x33
// RUN: %{run} %t.out --type int32 --channels 4 32x33
// RUN: %{run} %t.out --type uint32 --channels 1 32x33
// RUN: %{run} %t.out --type uint32 --channels 2 32x33
// RUN: %{run} %t.out --type uint32 --channels 4 32x33
// RUN: %{run} %t.out --type int16 --channels 1 32x33
// RUN: %{run} %t.out --type int16 --channels 2 32x33
// RUN: %{run} %t.out --type int16 --channels 4 32x33
// RUN: %{run} %t.out --type uint16 --channels 1 32x33
// RUN: %{run} %t.out --type uint16 --channels 2 32x33
// RUN: %{run} %t.out --type uint16 --channels 4 32x33
// RUN: %{run} %t.out --type uint8 --channels 1 32x33
// RUN: %{run} %t.out --type uint8 --channels 2 32x33
// RUN: %{run} %t.out --type uint8 --channels 4 32x33
// RUN: %{run} %t.out --type int8 --channels 1 32x33
// RUN: %{run} %t.out --type int8 --channels 2 32x33
// RUN: %{run} %t.out --type int8 --channels 4 32x33
// RUN: %{run} %t.out --type unorm8 --channels 1 32x33
// RUN: %{run} %t.out --type unorm8 --channels 2 32x33
// RUN: %{run} %t.out --type unorm8 --channels 4 32x33

#include "d3d12_setup.hpp"

#include <optional>
#include <string>
#include <sycl/builtins.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>
#include <sycl/ext/oneapi/bindless_images_interop.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

// ---------------------------------------------------------
// SYCL TYPE MAPPING HELPERS
// ---------------------------------------------------------
template <typename T> sycl::image_channel_type getSyclChannelType();

template <> inline sycl::image_channel_type getSyclChannelType<float>() {
  return sycl::image_channel_type::fp32;
}
template <> inline sycl::image_channel_type getSyclChannelType<sycl::half>() {
  return sycl::image_channel_type::fp16;
}
template <> inline sycl::image_channel_type getSyclChannelType<int32_t>() {
  return sycl::image_channel_type::signed_int32;
}
template <> inline sycl::image_channel_type getSyclChannelType<uint32_t>() {
  return sycl::image_channel_type::unsigned_int32;
}
template <> inline sycl::image_channel_type getSyclChannelType<int16_t>() {
  return sycl::image_channel_type::signed_int16;
}
template <> inline sycl::image_channel_type getSyclChannelType<uint16_t>() {
  return sycl::image_channel_type::unsigned_int16;
}
template <> inline sycl::image_channel_type getSyclChannelType<uint8_t>() {
  return sycl::image_channel_type::unsigned_int8;
}
template <> inline sycl::image_channel_type getSyclChannelType<int8_t>() {
  return sycl::image_channel_type::signed_int8;
}

template <> inline DXGI_FORMAT getDXGIFormat<sycl::half>(int channels) {
  if (channels == 1)
    return DXGI_FORMAT_R16_FLOAT;
  if (channels == 2)
    return DXGI_FORMAT_R16G16_FLOAT;
  if (channels == 4)
    return DXGI_FORMAT_R16G16B16A16_FLOAT;
  throw std::runtime_error("Unsupported channels for half");
}

// ---------------------------------------------------------
// UAV IMAGE CREATION
// ---------------------------------------------------------
inline D3D12ImageResources createExportableImageWrite(D3D12Context &ctx,
                                                      uint32_t width,
                                                      uint32_t height,
                                                      DXGI_FORMAT format) {
  D3D12ImageResources imgRes;
  D3D12_RESOURCE_DESC texDesc = {};
  texDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
  texDesc.Width = width;
  texDesc.Height = height;
  texDesc.DepthOrArraySize = 1;
  texDesc.MipLevels = 1;
  texDesc.Format = format;
  texDesc.SampleDesc.Count = 1;
  texDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;

  // KEY DIFFERENCE: Unordered Access is required for SYCL to write to the image
  texDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS |
                  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

  D3D12_HEAP_PROPERTIES defaultHeap = {D3D12_HEAP_TYPE_DEFAULT};

  ThrowIfFailed(ctx.device->CreateCommittedResource(
                    &defaultHeap, D3D12_HEAP_FLAG_SHARED, &texDesc,
                    D3D12_RESOURCE_STATE_COMMON, nullptr,
                    IID_PPV_ARGS(&imgRes.resource)),
                "Failed to create Shared UAV Texture");

  auto allocInfo = ctx.device->GetResourceAllocationInfo(0, 1, &texDesc);
  imgRes.allocationSize = allocInfo.SizeInBytes;

  ThrowIfFailed(ctx.device->CreateSharedHandle(imgRes.resource.Get(), nullptr,
                                               GENERIC_ALL, nullptr,
                                               &imgRes.sharedHandle),
                "Failed to export NT Handle");

  return imgRes;
}

// ---------------------------------------------------------
// TEMPLATED RUNNER
// ---------------------------------------------------------
template <typename T>
int runTest(
    int width, int height, int channels, bool useSemaphores,
    DXGI_FORMAT fmtOverride = DXGI_FORMAT_UNKNOWN,
    std::optional<sycl::image_channel_type> syclOverride = std::nullopt) {

  DXGI_FORMAT dxgiFormat = (fmtOverride != DXGI_FORMAT_UNKNOWN)
                               ? fmtOverride
                               : getDXGIFormat<T>(channels);

  // Setup D3D12 with UAV capability
  D3D12Context ctx = createD3D12Context();
  D3D12ImageResources imgRes =
      createExportableImageWrite(ctx, width, height, dxgiFormat);

  D3D12ExportableFence extFence;
  if (useSemaphores) {
    extFence = createExportableFence(ctx);
    // Signal fence to 1 to tell SYCL it can start writing
    signalExportableFence(ctx, extFence);
    std::cout << "D3D12 Fence Signaled (Value: " << extFence.fenceValue << ")"
              << std::endl;
  }

  try {
    sycl::queue q;

    syclexp::external_mem_descriptor<syclexp::resource_win32_handle> extMemDesc{
        imgRes.sharedHandle, syclexp::external_mem_handle_type::win32_nt_handle,
        imgRes.allocationSize};

    syclexp::external_mem extMem = syclexp::import_external_memory(
        extMemDesc, q.get_device(), q.get_context());

    sycl::image_channel_type syclType = syclOverride.has_value()
                                            ? syclOverride.value()
                                            : getSyclChannelType<T>();
    syclexp::image_descriptor imgDesc(sycl::range<2>(width, height), channels,
                                      syclType);

    syclexp::image_mem_handle devHandle = syclexp::map_external_image_memory(
        extMem, imgDesc, q.get_device(), q.get_context());
    syclexp::unsampled_image_handle unsampledHandle = syclexp::create_image(
        devHandle, imgDesc, q.get_device(), q.get_context());

    syclexp::external_semaphore extSem;
    if (useSemaphores) {
      syclexp::external_semaphore_descriptor<syclexp::resource_win32_handle>
          extSemDesc{extFence.sharedHandle,
                     syclexp::external_semaphore_handle_type::win32_nt_handle};
      extSem = syclexp::import_external_semaphore(extSemDesc, q.get_device(),
                                                  q.get_context());
    }

    sycl::event kernelEvent;

    // SYCL KERNEL: Write Image
    kernelEvent = q.submit([&](sycl::handler &h) {
      if (useSemaphores) {
        h.ext_oneapi_wait_external_semaphore(extSem, extFence.fenceValue);
      }

      h.parallel_for(sycl::range<2>(width, height), [=](sycl::item<2> item) {
        int x = item.get_id(0);
        int y = item.get_id(1);

        size_t index = y * width + x;
        size_t totalPixels = width * height;

        bool isUnorm = (syclType == sycl::image_channel_type::unorm_int8);
        if (isUnorm) {
          if (channels == 1) {
            float v =
                (float)generateTestValue<T>(index, 0, totalPixels) / 255.0f;
            syclexp::write_image(unsampledHandle, sycl::int2(x, y), v);
          } else if (channels == 2) {
            float v1 =
                (float)generateTestValue<T>(index, 0, totalPixels) / 255.0f;
            float v2 =
                (float)generateTestValue<T>(index, 1, totalPixels) / 255.0f;
            syclexp::write_image(unsampledHandle, sycl::int2(x, y),
                                 sycl::float2(v1, v2));
          } else {
            float v1 =
                (float)generateTestValue<T>(index, 0, totalPixels) / 255.0f;
            float v2 =
                (float)generateTestValue<T>(index, 1, totalPixels) / 255.0f;
            float v3 =
                (float)generateTestValue<T>(index, 2, totalPixels) / 255.0f;
            float v4 =
                (float)generateTestValue<T>(index, 3, totalPixels) / 255.0f;
            syclexp::write_image(unsampledHandle, sycl::int2(x, y),
                                 sycl::float4(v1, v2, v3, v4));
          }
          return; // Unorm scales handled
        }

        // Standard Data Paths
        if (channels == 1) {
          T val = generateTestValue<T>(index, 0, totalPixels);
          syclexp::write_image(unsampledHandle, sycl::int2(x, y), val);
        } else if (channels == 2) {
          using Vec2 = sycl::vec<T, 2>;
          Vec2 px(generateTestValue<T>(index, 0, totalPixels),
                  generateTestValue<T>(index, 1, totalPixels));
          syclexp::write_image(unsampledHandle, sycl::int2(x, y), px);
        } else {
          using Vec4 = sycl::vec<T, 4>;
          Vec4 px(generateTestValue<T>(index, 0, totalPixels),
                  generateTestValue<T>(index, 1, totalPixels),
                  generateTestValue<T>(index, 2, totalPixels),
                  generateTestValue<T>(index, 3, totalPixels));
          syclexp::write_image(unsampledHandle, sycl::int2(x, y), px);
        }
      });
    });

    if (useSemaphores) {
      q.submit([&](sycl::handler &h) {
        h.depends_on(kernelEvent);
        // Signal fence to 2 to tell D3D12 it can read
        h.ext_oneapi_signal_external_semaphore(extSem, extFence.fenceValue + 1);
      });
    }

    q.wait();

    syclexp::destroy_image_handle(unsampledHandle, q.get_device(),
                                  q.get_context());
    syclexp::release_external_memory(extMem, q.get_device(), q.get_context());
    if (useSemaphores) {
      syclexp::release_external_semaphore(extSem, q.get_device(),
                                          q.get_context());
    }

  } catch (std::exception &e) {
    std::cerr << "SYCL Exception: " << e.what() << std::endl;
    cleanupD3D12(ctx, imgRes);
    if (useSemaphores)
      cleanupExportableFence(extFence);
    return 1;
  }

  // Tell D3D12 queue to wait for SYCL to finish writing before doing the
  // readback verify
  if (useSemaphores) {
    ctx.cmdQueue->Wait(extFence.fence.Get(), extFence.fenceValue + 1);
    extFence.fenceValue++;
  }

  // Native D3D12 Verification (Checks the image memory against
  // generateTestValue)
  bool passed = verifyD3D12Data<T>(ctx, imgRes, width, height, channels);

  if (useSemaphores)
    cleanupExportableFence(extFence);
  cleanupD3D12(ctx, imgRes);

  return passed ? 0 : 1;
}

// ---------------------------------------------------------
// MAIN
// ---------------------------------------------------------
int main(int argc, char **argv) {
  int width = 32;
  int height = 33;
  int channels = 4;
  bool useSemaphores = false;
  std::string type = "float";

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--sampled") {
      std::cerr
          << "ERROR: --sampled flag is not supported for write operations."
          << std::endl;
      std::cerr << "Sampled images cannot be written to. Only unsampled "
                   "(storage) images support write_image()."
                << std::endl;
      return 1;
    } else if (arg == "--semaphores") {
      useSemaphores = true;
    } else if (arg == "--channels" && i + 1 < argc) {
      channels = std::stoi(argv[++i]);
    } else if (arg == "--type" && i + 1 < argc) {
      type = argv[++i];
    } else if (arg.find('x') != std::string::npos) {
      size_t pos = arg.find('x');
      width = std::stoi(arg.substr(0, pos));
      height = std::stoi(arg.substr(pos + 1));
    }
  }

  if (channels != 1 && channels != 2 && channels != 4) {
    std::cerr << "Error: Only 1, 2, or 4 channels supported." << std::endl;
    return 1;
  }

  std::cout << "Running UNSAMPLED D3D12 2D Write Test | Type: " << type
            << " | Size: " << width << "x" << height
            << " | Channels: " << channels
            << " | Semaphores: " << (useSemaphores ? "ON" : "OFF") << std::endl;

  if (type == "float")
    return runTest<float>(width, height, channels, useSemaphores);
  if (type == "half")
    return runTest<sycl::half>(width, height, channels, useSemaphores);
  if (type == "int32")
    return runTest<int32_t>(width, height, channels, useSemaphores);
  if (type == "uint32")
    return runTest<uint32_t>(width, height, channels, useSemaphores);
  if (type == "int16")
    return runTest<int16_t>(width, height, channels, useSemaphores);
  if (type == "uint16")
    return runTest<uint16_t>(width, height, channels, useSemaphores);
  if (type == "uint8")
    return runTest<uint8_t>(width, height, channels, useSemaphores);
  if (type == "int8")
    return runTest<int8_t>(width, height, channels, useSemaphores);

  if (type == "unorm8") {
    return runTest<uint8_t>(width, height, channels, useSemaphores,
                            getUnorm8Format(channels),
                            sycl::image_channel_type::unorm_int8);
  }

  std::cerr << "Unknown type: " << type << std::endl;
  return 1;
}