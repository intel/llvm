// REQUIRES: aspect-ext_oneapi_bindless_images
// REQUIRES: aspect-ext_oneapi_external_memory_import
// REQUIRES: windows

// RUN: %{build} -o %t.exe -ld3d12 -ldxgi -ld3dcompiler
// RUN: %{run} %t.exe --type float --channels 4 32x33

// clang-format off
/*
    clang++.exe -fsycl -o ds2r.exe D3D12_sycl_interop_2D_read.cpp -ld3d12 -ldxgi -ld3dcompiler


    FLAGS:
    --sampled      Use sampled image path (default: unsampled/storage)
    --semaphores   Use DX12 Fences for SYCL Interop Sync
    --channels X   Set number of channels (1, 2, or 4). Default is 4 (RGBA)
    --type XXX     Set data type (float, half, uint32, int32, uint16, int16, uint8, int8, unorm8).
                   Default is float 
    WxH            Set custom Width x Height (e.g. 8x4)


    BMG:
    - 1x1 works, nothing else. Suggesting offset/pitch issue.
    - semaphores segfault. suggesting segfaulting semaphores.

    BMG $ sycl-ls
    [level_zero:gpu][level_zero:0] Intel(R) oneAPI Unified Runtime over
   Level-Zero V2, Intel(R) Arc(TM) B580 Graphics 20.1.0 [1.13.35716]
    [opencl:gpu][opencl:0] Intel(R) OpenCL Graphics, Intel(R) Arc(TM) B580
   Graphics OpenCL 3.0 NEO  [32.0.101.8331]

    DG2:
    - WORKS, including --sampled
    - semaphores segfault

    DG2 $ sycl-ls
    [level_zero:gpu][level_zero:0] Intel(R) oneAPI Unified Runtime over
   Level-Zero, Intel(R) Arc(TM) A770 Graphics 12.55.8 [1.6.34938]
    [opencl:gpu][opencl:0] Intel(R) OpenCL Graphics, Intel(R) Arc(TM) A770
   Graphics OpenCL 3.0 NEO  [32.0.101.8132]


   unorm8 seems to not be working. 

*/
// clang-format on

// RUN: %{run} %t.exe --type float --channels 1 32x33
// RUN: %{run} %t.exe --type float --channels 2 32x33
// RUN: %{run} %t.exe --type float --channels 4 32x33
// RUN: %{run} %t.exe --type half --channels 1 32x33
// RUN: %{run} %t.exe --type half --channels 2 32x33
// RUN: %{run} %t.exe --type half --channels 4 32x33
// RUN: %{run} %t.exe --type int32 --channels 1 32x33
// RUN: %{run} %t.exe --type int32 --channels 2 32x33
// RUN: %{run} %t.exe --type int32 --channels 4 32x33
// RUN: %{run} %t.exe --type uint32 --channels 1 32x33
// RUN: %{run} %t.exe --type uint32 --channels 2 32x33
// RUN: %{run} %t.exe --type uint32 --channels 4 32x33
// RUN: %{run} %t.exe --type int16 --channels 1 32x33
// RUN: %{run} %t.exe --type int16 --channels 2 32x33
// RUN: %{run} %t.exe --type int16 --channels 4 32x33
// RUN: %{run} %t.exe --type uint16 --channels 1 32x33
// RUN: %{run} %t.exe --type uint16 --channels 2 32x33
// RUN: %{run} %t.exe --type uint16 --channels 4 32x33
// RUN: %{run} %t.exe --type uint8 --channels 1 32x33
// RUN: %{run} %t.exe --type uint8 --channels 2 32x33
// RUN: %{run} %t.exe --type uint8 --channels 4 32x33
// RUN: %{run} %t.exe --type int8 --channels 1 32x33
// RUN: %{run} %t.exe --type int8 --channels 2 32x33
// RUN: %{run} %t.exe --type int8 --channels 4 32x33
// RUN: %{run} %t.exe --type float --channels 1 --sampled 32x33
// RUN: %{run} %t.exe --type float --channels 2 --sampled 32x33
// RUN: %{run} %t.exe --type float --channels 4 --sampled 32x33
// RUN: %{run} %t.exe --type half --channels 1 --sampled 32x33
// RUN: %{run} %t.exe --type half --channels 2 --sampled 32x33
// RUN: %{run} %t.exe --type half --channels 4 --sampled 32x33
// RUN: %{run} %t.exe --type int32 --channels 1 --sampled 32x33
// RUN: %{run} %t.exe --type int32 --channels 2 --sampled 32x33
// RUN: %{run} %t.exe --type int32 --channels 4 --sampled 32x33
// RUN: %{run} %t.exe --type uint32 --channels 1 --sampled 32x33
// RUN: %{run} %t.exe --type uint32 --channels 2 --sampled 32x33
// RUN: %{run} %t.exe --type uint32 --channels 4 --sampled 32x33
// RUN: %{run} %t.exe --type int16 --channels 1 --sampled 32x33
// RUN: %{run} %t.exe --type int16 --channels 2 --sampled 32x33
// RUN: %{run} %t.exe --type int16 --channels 4 --sampled 32x33
// RUN: %{run} %t.exe --type uint16 --channels 1 --sampled 32x33
// RUN: %{run} %t.exe --type uint16 --channels 2 --sampled 32x33
// RUN: %{run} %t.exe --type uint16 --channels 4 --sampled 32x33
// RUN: %{run} %t.exe --type uint8 --channels 1 --sampled 32x33
// RUN: %{run} %t.exe --type uint8 --channels 2 --sampled 32x33
// RUN: %{run} %t.exe --type uint8 --channels 4 --sampled 32x33
// RUN: %{run} %t.exe --type int8 --channels 1 --sampled 32x33
// RUN: %{run} %t.exe --type int8 --channels 2 --sampled 32x33
// RUN: %{run} %t.exe --type int8 --channels 4 --sampled 32x33

// Semaphore coverage tests
// RUN: %{run} %t.exe --type float --channels 4 --semaphores 32x33
// RUN: %{run} %t.exe --type half --channels 2 --semaphores 32x33
// RUN: %{run} %t.exe --type int32 --channels 1 --semaphores 32x33
// RUN: %{run} %t.exe --type uint32 --channels 4 --semaphores 32x33
// RUN: %{run} %t.exe --type int16 --channels 2 --semaphores 32x33
// RUN: %{run} %t.exe --type uint16 --channels 1 --semaphores 32x33
// RUN: %{run} %t.exe --type uint8 --channels 4 --semaphores 32x33
// RUN: %{run} %t.exe --type int8 --channels 2 --semaphores 32x33
// RUN: %{run} %t.exe --type float --channels 4 --sampled --semaphores 32x33
// RUN: %{run} %t.exe --type half --channels 2 --sampled --semaphores 32x33
// RUN: %{run} %t.exe --type int32 --channels 1 --sampled --semaphores 32x33
// RUN: %{run} %t.exe --type uint32 --channels 4 --sampled --semaphores 32x33


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
    int width, int height, int channels, bool useSampled, bool useSemaphores,
    DXGI_FORMAT fmtOverride = DXGI_FORMAT_UNKNOWN,
    std::optional<sycl::image_channel_type> syclOverride = std::nullopt) {

  DXGI_FORMAT dxgiFormat = (fmtOverride != DXGI_FORMAT_UNKNOWN)
                               ? fmtOverride
                               : getDXGIFormat<T>(channels);

  // Setup D3D12
  D3D12Context ctx = createD3D12Context();
  D3D12ImageResources imgRes =
      createExportableImage(ctx, width, height, dxgiFormat);

  // Upload test data
  if (!uploadTestData<T>(ctx, imgRes, width, height, channels)) {
    std::cerr << "D3D12 Upload Failed!" << std::endl;
    return 1;
  }

  if (!verifyD3D12Data<T>(ctx, imgRes, width, height, channels)) {
    std::cerr << "Aborting SYCL test. Native D3D12 readback failed."
              << std::endl;
    cleanupD3D12(ctx, imgRes);
    return 1;
  }

  // Setup Exportable Fence if requested
  D3D12ExportableFence extFence;
  if (useSemaphores) {
    extFence = createExportableFence(ctx);
    // Signal the fence from the D3D12 queue indicating upload is complete
    signalExportableFence(ctx, extFence);
    std::cout << "D3D12 Fence Signaled (Value: " << extFence.fenceValue << ")"
              << std::endl;
  }

  // SYCL Import and Verification
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

    // Note: For Phase 2, we stick to unsampled logic as a baseline. We will
    // branch for sampled in Phase 4.
    // syclexp::unsampled_image_handle unsampledHandle =
    // syclexp::create_image(devHandle, imgDesc, q.get_device(),
    // q.get_context());
    syclexp::sampled_image_handle sampledHandle;
    syclexp::unsampled_image_handle unsampledHandle;

    if (useSampled) {
      syclexp::bindless_image_sampler sampler(
          sycl::addressing_mode::clamp_to_edge,
          sycl::coordinate_normalization_mode::unnormalized,
          sycl::filtering_mode::nearest);
      sampledHandle = syclexp::create_image(devHandle, sampler, imgDesc,
                                            q.get_device(), q.get_context());
    } else {
      unsampledHandle = syclexp::create_image(devHandle, imgDesc,
                                              q.get_device(), q.get_context());
    }

    // Import the D3D12 Fence into SYCL
    syclexp::external_semaphore extSem;
    if (useSemaphores) {
      syclexp::external_semaphore_descriptor<syclexp::resource_win32_handle>
          extSemDesc{
              extFence.sharedHandle,
              syclexp::external_semaphore_handle_type::win32_nt_dx12_fence};
      extSem = syclexp::import_external_semaphore(extSemDesc, q.get_device(),
                                                  q.get_context());
    }

    size_t totalValues = width * height * channels;
    sycl::buffer<T, 1> checkBuf(totalValues);

    // Wait on the Fence
    sycl::event dependencyEvent;
    if (useSemaphores) {
      dependencyEvent = q.submit([&](sycl::handler &h) {
        // Note: D3D12 timeline fences require passing the specific fenceValue
        // to wait on
        h.ext_oneapi_wait_external_semaphore(extSem, extFence.fenceValue);
      });
    }

    q.submit([&](sycl::handler &h) {
       if (useSemaphores)
         h.depends_on(dependencyEvent);

       sycl::accessor outAcc(checkBuf, h, sycl::write_only);
       h.parallel_for(sycl::range<2>(width, height), [=](sycl::item<2> item) {
         int x = item.get_id(0);
         int y = item.get_id(1);
         size_t base = (y * width + x) * channels;

         if (useSampled) {
           // Sampled path: float coordinates offset to pixel center
           float coordX = (float)x + 0.5f;
           float coordY = (float)y + 0.5f;

           if constexpr (std::is_floating_point_v<T> ||
                         std::is_same_v<T, sycl::half>) {
             sycl::float4 pixel = syclexp::sample_image<sycl::float4>(
                 sampledHandle, sycl::float2(coordX, coordY));
             if (channels >= 1)
               outAcc[base + 0] = static_cast<T>(pixel.x());
             if (channels >= 2)
               outAcc[base + 1] = static_cast<T>(pixel.y());
             if (channels == 4) {
               outAcc[base + 2] = static_cast<T>(pixel.z());
               outAcc[base + 3] = static_cast<T>(pixel.w());
             }
           } else if constexpr (std::is_signed_v<T>) {
             sycl::int4 pixel = syclexp::sample_image<sycl::int4>(
                 sampledHandle, sycl::float2(coordX, coordY));
             if (channels >= 1)
               outAcc[base + 0] = static_cast<T>(pixel.x());
             if (channels >= 2)
               outAcc[base + 1] = static_cast<T>(pixel.y());
             if (channels == 4) {
               outAcc[base + 2] = static_cast<T>(pixel.z());
               outAcc[base + 3] = static_cast<T>(pixel.w());
             }
           } else {
             sycl::uint4 pixel = syclexp::sample_image<sycl::uint4>(
                 sampledHandle, sycl::float2(coordX, coordY));
             if (channels >= 1)
               outAcc[base + 0] = static_cast<T>(pixel.x());
             if (channels >= 2)
               outAcc[base + 1] = static_cast<T>(pixel.y());
             if (channels == 4) {
               outAcc[base + 2] = static_cast<T>(pixel.z());
               outAcc[base + 3] = static_cast<T>(pixel.w());
             }
           }
         } else {
           // Unsampled path
           sycl::int2 coords(x, y);
           if (channels == 1) {
             outAcc[base] = syclexp::fetch_image<T>(unsampledHandle, coords);
           } else if (channels == 2) {
             sycl::vec<T, 2> pixel =
                 syclexp::fetch_image<sycl::vec<T, 2>>(unsampledHandle, coords);
             outAcc[base + 0] = pixel.x();
             outAcc[base + 1] = pixel.y();
           } else if (channels == 4) {
             sycl::vec<T, 4> pixel =
                 syclexp::fetch_image<sycl::vec<T, 4>>(unsampledHandle, coords);
             outAcc[base + 0] = pixel.x();
             outAcc[base + 1] = pixel.y();
             outAcc[base + 2] = pixel.z();
             outAcc[base + 3] = pixel.w();
           }
         }
       });
     }).wait();

    // Verify results
    sycl::host_accessor hostAcc(checkBuf, sycl::read_only);
    bool passed = true;
    int errorCount = 0;
    size_t totalPixels = width * height;

    for (size_t i = 0; i < totalValues; ++i) {
      size_t pixelIdx = i / channels;
      int channelIdx = i % channels;
      T expected = generateTestValue<T>(pixelIdx, channelIdx, totalPixels);

      if (!checkValue(hostAcc[i], expected)) {
        if (errorCount < 10) {
          std::cerr << "Mismatch at index " << i << ": Expected " << expected
                    << ", Got " << hostAcc[i] << std::endl;
        }
        errorCount++;
        passed = false;
      }
    }

    if (passed) {
      std::cout << "SUCCESS! All " << totalValues << " values match."
                << std::endl;
    } else {
      std::cout << "FAILURE! " << errorCount << " errors out of " << totalValues
                << " values." << std::endl;
    }

    if (useSemaphores) {
      syclexp::release_external_semaphore(extSem, q.get_device(),
                                          q.get_context());
      cleanupExportableFence(extFence);
    }

    // Cleanup SYCL resources
    if (useSampled) {
      syclexp::destroy_image_handle(sampledHandle, q.get_device(),
                                    q.get_context());
    } else {
      syclexp::destroy_image_handle(unsampledHandle, q.get_device(),
                                    q.get_context());
    }

    syclexp::release_external_memory(extMem, q.get_device(), q.get_context());
    cleanupD3D12(ctx, imgRes);

    return passed ? 0 : 1;

  } catch (std::exception &e) {
    std::cerr << "SYCL Exception: " << e.what() << std::endl;
    cleanupD3D12(ctx, imgRes);
    return 1;
  }
}

// ---------------------------------------------------------
// MAIN
// ---------------------------------------------------------
int main(int argc, char **argv) {
  int width = 32;
  int height = 33;
  int channels = 4;
  bool useSemaphores = false;
  bool useSampled = false;
  std::string type = "float";

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--sampled") {
      useSampled = true;
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

  std::cout << "Running " << (useSampled ? "SAMPLED" : "UNSAMPLED")
            << " D3D12 2D Read Test | Type: " << type << " | Size: " << width
            << "x" << height << " | Channels: " << channels
            << " | Semaphores: " << (useSemaphores ? "ON" : "OFF") << std::endl;

  if (type == "float")
    return runTest<float>(width, height, channels, useSampled, useSemaphores);
  if (type == "half")
    return runTest<sycl::half>(width, height, channels, useSampled,
                               useSemaphores);
  if (type == "int32")
    return runTest<int32_t>(width, height, channels, useSampled, useSemaphores);
  if (type == "uint32")
    return runTest<uint32_t>(width, height, channels, useSampled,
                             useSemaphores);
  if (type == "int16")
    return runTest<int16_t>(width, height, channels, useSampled, useSemaphores);
  if (type == "uint16")
    return runTest<uint16_t>(width, height, channels, useSampled,
                             useSemaphores);
  if (type == "uint8")
    return runTest<uint8_t>(width, height, channels, useSampled, useSemaphores);
  if (type == "int8")
    return runTest<int8_t>(width, height, channels, useSampled, useSemaphores);

  if (type == "unorm8") {
    return runTest<uint8_t>(width, height, channels, useSampled, useSemaphores,
                            getUnorm8Format(channels),
                            sycl::image_channel_type::unorm_int8);
  }

  std::cerr << "Unknown type: " << type << std::endl;
  return 1;
}