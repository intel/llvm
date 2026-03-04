// REQUIRES: aspect-ext_oneapi_bindless_images
// REQUIRES: aspect-ext_oneapi_external_memory_import
// REQUIRES: windows

// RUN: %{build} -o %t.exe -ld3d12 -ldxgi -ld3dcompiler
// RUN: %{run} %t.exe --type float --channels 4 8x8x8

// RUN: %{run} %t.exe --type float --channels 1 33x32x31
// RUN: %{run} %t.exe --type float --channels 2 32x33x31
// RUN: %{run} %t.exe --type float --channels 4 31x32x33
// RUN: %{run} %t.exe --type half --channels 1 16x17x15
// RUN: %{run} %t.exe --type half --channels 2 17x16x15
// RUN: %{run} %t.exe --type half --channels 4 15x16x17
// RUN: %{run} %t.exe --type int32 --channels 1 9x8x7
// RUN: %{run} %t.exe --type int32 --channels 2 8x9x7
// RUN: %{run} %t.exe --type int32 --channels 4 7x8x9
// RUN: %{run} %t.exe --type uint32 --channels 1 33x31x32
// RUN: %{run} %t.exe --type uint32 --channels 2 31x33x32
// RUN: %{run} %t.exe --type uint32 --channels 4 32x31x33
// RUN: %{run} %t.exe --type int16 --channels 1 17x15x16
// RUN: %{run} %t.exe --type int16 --channels 2 15x17x16
// RUN: %{run} %t.exe --type int16 --channels 4 16x15x17
// RUN: %{run} %t.exe --type uint16 --channels 1 9x7x8
// RUN: %{run} %t.exe --type uint16 --channels 2 7x9x8
// RUN: %{run} %t.exe --type uint16 --channels 4 8x7x9
// RUN: %{run} %t.exe --type uint8 --channels 1 33x32x31
// RUN: %{run} %t.exe --type uint8 --channels 2 32x31x33
// RUN: %{run} %t.exe --type uint8 --channels 4 31x33x32
// RUN: %{run} %t.exe --type int8 --channels 1 17x16x15
// RUN: %{run} %t.exe --type int8 --channels 2 16x15x17
// RUN: %{run} %t.exe --type int8 --channels 4 15x17x16
// RUN: %{run} %t.exe --type unorm8 --channels 1 9x8x7
// RUN: %{run} %t.exe --type unorm8 --channels 2 8x7x9
// RUN: %{run} %t.exe --type unorm8 --channels 4 7x9x8

// clang-format off
/*
    clang++.exe -fsycl -o ds3w.exe D3D12_sycl_interop_3D_write.cpp -ld3d12 -ldxgi -ld3dcompiler

    FLAGS:
    --sampled      ERROR: Sampled image writes are not supported
    --semaphores   Use DX12 Fences for SYCL Interop Sync
    --channels X   Set number of channels (1, 2, or 4). Default is 4 (RGBA)
    --type XXX     Set data type (float, half, uint32, int32, uint16, int16, uint8, int8, unorm8). 
                   Default is float 
   WxHxD           Set custom Width x Height x Depth (e.g. 8x4x41)
*/
// clang-format on

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
// TEMPLATED RUNNER
// ---------------------------------------------------------
template <typename T>
int runTest(
    int width, int height, int depth, int channels, bool useSemaphores,
    DXGI_FORMAT fmtOverride = DXGI_FORMAT_UNKNOWN,
    std::optional<sycl::image_channel_type> syclOverride = std::nullopt) {

  DXGI_FORMAT dxgiFormat = (fmtOverride != DXGI_FORMAT_UNKNOWN)
                               ? fmtOverride
                               : getDXGIFormat<T>(channels);

  // Setup D3D12 with UAV capability for 3D
  D3D12Context ctx = createD3D12Context();
  D3D12ImageResources imgRes =
      createExportableImageWrite3D(ctx, width, height, depth, dxgiFormat);

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
    syclexp::image_descriptor imgDesc(sycl::range<3>(width, height, depth),
                                      channels, syclType);

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

    // SYCL KERNEL: Write Image 3D
    kernelEvent = q.submit([&](sycl::handler &h) {
      if (useSemaphores) {
        h.ext_oneapi_wait_external_semaphore(extSem, extFence.fenceValue);
      }

      h.parallel_for(
          sycl::range<3>(width, height, depth), [=](sycl::item<3> item) {
            int x = item.get_id(0);
            int y = item.get_id(1);
            int z = item.get_id(2);

            size_t index = (z * width * height) + (y * width) + x;
            size_t totalPixels = width * height * depth;
            sycl::int3 coords(x, y, z);

            bool isUnorm = (syclType == sycl::image_channel_type::unorm_int8);
            if (isUnorm) {
              if (channels == 1) {
                float v =
                    (float)generateTestValue<T>(index, 0, totalPixels) / 255.0f;
                syclexp::write_image(unsampledHandle, coords, v);
              } else if (channels == 2) {
                float v1 =
                    (float)generateTestValue<T>(index, 0, totalPixels) / 255.0f;
                float v2 =
                    (float)generateTestValue<T>(index, 1, totalPixels) / 255.0f;
                syclexp::write_image(unsampledHandle, coords,
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
                syclexp::write_image(unsampledHandle, coords,
                                     sycl::float4(v1, v2, v3, v4));
              }
              return;
            }

            // Standard Data Paths
            if (channels == 1) {
              T val = generateTestValue<T>(index, 0, totalPixels);
              syclexp::write_image(unsampledHandle, coords, val);
            } else if (channels == 2) {
              using Vec2 = sycl::vec<T, 2>;
              Vec2 px(generateTestValue<T>(index, 0, totalPixels),
                      generateTestValue<T>(index, 1, totalPixels));
              syclexp::write_image(unsampledHandle, coords, px);
            } else {
              using Vec4 = sycl::vec<T, 4>;
              Vec4 px(generateTestValue<T>(index, 0, totalPixels),
                      generateTestValue<T>(index, 1, totalPixels),
                      generateTestValue<T>(index, 2, totalPixels),
                      generateTestValue<T>(index, 3, totalPixels));
              syclexp::write_image(unsampledHandle, coords, px);
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

  if (useSemaphores) {
    ctx.cmdQueue->Wait(extFence.fence.Get(), extFence.fenceValue + 1);
    extFence.fenceValue++;
  }

  // Pass depth for the readback validation mapping
  bool passed =
      verifyD3D12Data3D<T>(ctx, imgRes, width, height, depth, channels);

  if (useSemaphores)
    cleanupExportableFence(extFence);
  cleanupD3D12(ctx, imgRes);

  return passed ? 0 : 1;
}

// ---------------------------------------------------------
// MAIN
// ---------------------------------------------------------
int main(int argc, char **argv) {
  int width = 8;
  int height = 8;
  int depth = 8;
  int channels = 4;
  bool useSemaphores = false;
  std::string type = "float";

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--sampled") {
      std::cerr
          << "ERROR: --sampled flag is not supported for write operations."
          << std::endl;
      return 1;
    } else if (arg == "--semaphores") {
      useSemaphores = true;
    } else if (arg == "--channels" && i + 1 < argc) {
      channels = std::stoi(argv[++i]);
    } else if (arg == "--type" && i + 1 < argc) {
      type = argv[++i];
    } else if (arg.find('x') != std::string::npos) {
      size_t pos1 = arg.find('x');
      size_t pos2 = arg.find('x', pos1 + 1);
      if (pos2 != std::string::npos) {
        width = std::stoi(arg.substr(0, pos1));
        height = std::stoi(arg.substr(pos1 + 1, pos2 - pos1 - 1));
        depth = std::stoi(arg.substr(pos2 + 1));
      }
    }
  }

  if (channels != 1 && channels != 2 && channels != 4) {
    std::cerr << "Error: Only 1, 2, or 4 channels supported." << std::endl;
    return 1;
  }

  std::cout << "Running UNSAMPLED D3D12 3D Write Test | Type: " << type
            << " | Size: " << width << "x" << height << "x" << depth
            << " | Channels: " << channels
            << " | Semaphores: " << (useSemaphores ? "ON" : "OFF") << std::endl;

  if (type == "float")
    return runTest<float>(width, height, depth, channels, useSemaphores);
  if (type == "half")
    return runTest<sycl::half>(width, height, depth, channels, useSemaphores);
  if (type == "int32")
    return runTest<int32_t>(width, height, depth, channels, useSemaphores);
  if (type == "uint32")
    return runTest<uint32_t>(width, height, depth, channels, useSemaphores);
  if (type == "int16")
    return runTest<int16_t>(width, height, depth, channels, useSemaphores);
  if (type == "uint16")
    return runTest<uint16_t>(width, height, depth, channels, useSemaphores);
  if (type == "uint8")
    return runTest<uint8_t>(width, height, depth, channels, useSemaphores);
  if (type == "int8")
    return runTest<int8_t>(width, height, depth, channels, useSemaphores);

  if (type == "unorm8") {
    return runTest<uint8_t>(width, height, depth, channels, useSemaphores,
                            getUnorm8Format(channels),
                            sycl::image_channel_type::unorm_int8);
  }

  std::cerr << "Unknown type: " << type << std::endl;
  return 1;
}