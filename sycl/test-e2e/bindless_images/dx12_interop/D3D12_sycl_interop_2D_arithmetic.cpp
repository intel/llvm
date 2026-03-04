// REQUIRES: aspect-ext_oneapi_bindless_images
// REQUIRES: aspect-ext_oneapi_external_memory_import
// REQUIRES: windows

// RUN: %{build} -o %t.exe -ld3d12 -ldxgi -ld3dcompiler
// RUN: %{run} %t.exe --type float --channels 4 8x8

// clang-format off
/*
    clang++.exe -fsycl -o ds2A.exe D3D12_sycl_interop_2D_arithmetic.cpp -ld3d12 -ldxgi -ld3dcompiler

    FLAGS:
    --sampled      Use sampled image path (default: unsampled/storage)
    --semaphores   Use DX12 Fences for SYCL Interop Sync
    --channels X   Set number of channels (1, 2, or 4). Default is 4 (RGBA)
    --type XXX     Set data type (float, half, uint32, int32, uint16, int16, uint8, int8, unorm8). 
                   Default is float 
    WxH            Set custom Width x Height (e.g. 8x4)


    BMG:

    DG2:
    - WORKS, including --sampled
    - semaphores segfault

    DG2 $ sycl-ls
    [level_zero:gpu][level_zero:0] Intel(R) oneAPI Unified Runtime over
   Level-Zero, Intel(R) Arc(TM) A770 Graphics 12.55.8 [1.6.34938]
    [opencl:gpu][opencl:0] Intel(R) OpenCL Graphics, Intel(R) Arc(TM) A770
   Graphics OpenCL 3.0 NEO  [32.0.101.8132]

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
// RUN: %{run} %t.out --type float --channels 1 --sampled 32x33
// RUN: %{run} %t.out --type float --channels 2 --sampled 32x33
// RUN: %{run} %t.out --type float --channels 4 --sampled 32x33
// RUN: %{run} %t.out --type half --channels 1 --sampled 32x33
// RUN: %{run} %t.out --type half --channels 2 --sampled 32x33
// RUN: %{run} %t.out --type half --channels 4 --sampled 32x33
// RUN: %{run} %t.out --type int32 --channels 1 --sampled 32x33
// RUN: %{run} %t.out --type int32 --channels 2 --sampled 32x33
// RUN: %{run} %t.out --type int32 --channels 4 --sampled 32x33
// RUN: %{run} %t.out --type uint32 --channels 1 --sampled 32x33
// RUN: %{run} %t.out --type uint32 --channels 2 --sampled 32x33
// RUN: %{run} %t.out --type uint32 --channels 4 --sampled 32x33
// RUN: %{run} %t.out --type int16 --channels 1 --sampled 32x33
// RUN: %{run} %t.out --type int16 --channels 2 --sampled 32x33
// RUN: %{run} %t.out --type int16 --channels 4 --sampled 32x33
// RUN: %{run} %t.out --type uint16 --channels 1 --sampled 32x33
// RUN: %{run} %t.out --type uint16 --channels 2 --sampled 32x33
// RUN: %{run} %t.out --type uint16 --channels 4 --sampled 32x33
// RUN: %{run} %t.out --type uint8 --channels 1 --sampled 32x33
// RUN: %{run} %t.out --type uint8 --channels 2 --sampled 32x33
// RUN: %{run} %t.out --type uint8 --channels 4 --sampled 32x33
// RUN: %{run} %t.out --type int8 --channels 1 --sampled 32x33
// RUN: %{run} %t.out --type int8 --channels 2 --sampled 32x33
// RUN: %{run} %t.out --type int8 --channels 4 --sampled 32x33
// RUN: %{run} %t.out --type unorm8 --channels 1 --sampled 32x33
// RUN: %{run} %t.out --type unorm8 --channels 2 --sampled 32x33
// RUN: %{run} %t.out --type unorm8 --channels 4 --sampled 32x33

#include "d3d12_setup.hpp"

#include <algorithm>
#include <optional>
#include <string>
#include <sycl/builtins.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>
#include <sycl/ext/oneapi/bindless_images_interop.hpp>
#include <vector>

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
// GENERATORS & CUSTOM DATA HELPERS
// ---------------------------------------------------------
template <typename T> T generateValueA(size_t x, size_t y, int channel) {
  float val = (float)(x + y) / 100.0f;
  if constexpr (std::is_floating_point_v<T>)
    return static_cast<T>(val + channel * 0.1f);
  else
    return static_cast<T>((x + y + channel * 10) % 64);
}

template <typename T> T generateValueB(size_t x, size_t y, int channel) {
  float val = (float)(x * 2 + y) / 100.0f;
  if constexpr (std::is_floating_point_v<T>)
    return static_cast<T>(val + channel * 0.2f);
  else
    return static_cast<T>((x * 2 + y + channel * 5) % 64);
}

template <typename T, typename Func>
bool uploadCustomData(D3D12Context &ctx, D3D12ImageResources &imgRes,
                      uint32_t width, uint32_t height, int channels,
                      Func generator) {
  size_t totalPixels = width * height;
  std::vector<T> hostData(totalPixels * channels);
  for (size_t i = 0; i < hostData.size(); ++i) {
    size_t pixelIdx = i / channels;
    int channelIdx = i % channels;
    hostData[i] = generator(pixelIdx % width, pixelIdx / width, channelIdx);
  }

  D3D12_RESOURCE_DESC desc = imgRes.resource->GetDesc();
  D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint;
  UINT numRows;
  UINT64 rowSizeInBytes, totalBytes;
  ctx.device->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, &numRows,
                                    &rowSizeInBytes, &totalBytes);

  ComPtr<ID3D12Resource> uploadBuffer;
  D3D12_HEAP_PROPERTIES uploadHeap = {D3D12_HEAP_TYPE_UPLOAD};
  D3D12_RESOURCE_DESC bufDesc = {};
  bufDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  bufDesc.Width = totalBytes;
  bufDesc.Height = 1;
  bufDesc.DepthOrArraySize = 1;
  bufDesc.MipLevels = 1;
  bufDesc.Format = DXGI_FORMAT_UNKNOWN;
  bufDesc.SampleDesc.Count = 1;
  bufDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  bufDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

  ThrowIfFailed(ctx.device->CreateCommittedResource(
      &uploadHeap, D3D12_HEAP_FLAG_NONE, &bufDesc,
      D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&uploadBuffer)));

  uint8_t *pData;
  ThrowIfFailed(
      uploadBuffer->Map(0, nullptr, reinterpret_cast<void **>(&pData)));
  const uint8_t *pSrc = reinterpret_cast<const uint8_t *>(hostData.data());
  for (UINT y = 0; y < height; ++y) {
    memcpy(pData + y * footprint.Footprint.RowPitch,
           pSrc + y * (width * sizeof(T) * channels),
           width * sizeof(T) * channels);
  }
  uploadBuffer->Unmap(0, nullptr);

  ThrowIfFailed(ctx.cmdAlloc->Reset());
  ThrowIfFailed(ctx.cmdList->Reset(ctx.cmdAlloc.Get(), nullptr));

  D3D12_TEXTURE_COPY_LOCATION dst = {};
  dst.pResource = imgRes.resource.Get();
  dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  dst.SubresourceIndex = 0;

  D3D12_TEXTURE_COPY_LOCATION src = {};
  src.pResource = uploadBuffer.Get();
  src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
  src.PlacedFootprint = footprint;

  ctx.cmdList->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
  ThrowIfFailed(ctx.cmdList->Close());

  executeAndWait(ctx);
  return true;
}

template <typename T, typename Func>
bool verifyCustomData(D3D12Context &ctx, D3D12ImageResources &imgRes,
                      uint32_t width, uint32_t height, int channels,
                      Func expectedGenerator) {
  D3D12_RESOURCE_DESC desc = imgRes.resource->GetDesc();
  D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint;
  UINT numRows;
  UINT64 rowSizeInBytes, totalBytes;
  ctx.device->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, &numRows,
                                    &rowSizeInBytes, &totalBytes);

  ComPtr<ID3D12Resource> readbackBuffer;
  D3D12_HEAP_PROPERTIES readbackHeap = {D3D12_HEAP_TYPE_READBACK};
  D3D12_RESOURCE_DESC bufDesc = {};
  bufDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  bufDesc.Width = totalBytes;
  bufDesc.Height = 1;
  bufDesc.DepthOrArraySize = 1;
  bufDesc.MipLevels = 1;
  bufDesc.Format = DXGI_FORMAT_UNKNOWN;
  bufDesc.SampleDesc.Count = 1;
  bufDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

  ThrowIfFailed(ctx.device->CreateCommittedResource(
      &readbackHeap, D3D12_HEAP_FLAG_NONE, &bufDesc,
      D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readbackBuffer)));

  D3D12_RESOURCE_BARRIER barrier = {};
  barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  barrier.Transition.pResource = imgRes.resource.Get();
  barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
  barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
  barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

  ThrowIfFailed(ctx.cmdAlloc->Reset());
  ThrowIfFailed(ctx.cmdList->Reset(ctx.cmdAlloc.Get(), nullptr));
  ctx.cmdList->ResourceBarrier(1, &barrier);

  D3D12_TEXTURE_COPY_LOCATION dst = {};
  dst.pResource = readbackBuffer.Get();
  dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
  dst.PlacedFootprint = footprint;

  D3D12_TEXTURE_COPY_LOCATION src = {};
  src.pResource = imgRes.resource.Get();
  src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  src.SubresourceIndex = 0;

  ctx.cmdList->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);

  barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
  barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
  ctx.cmdList->ResourceBarrier(1, &barrier);

  ThrowIfFailed(ctx.cmdList->Close());
  executeAndWait(ctx);

  uint8_t *pData;
  ThrowIfFailed(
      readbackBuffer->Map(0, nullptr, reinterpret_cast<void **>(&pData)));

  bool passed = true;
  int errorCount = 0;

  for (UINT y = 0; y < height; ++y) {
    T *row = reinterpret_cast<T *>(pData + y * footprint.Footprint.RowPitch);
    for (UINT x = 0; x < width; ++x) {
      for (int c = 0; c < channels; ++c) {
        T expected = expectedGenerator(x, y, c);
        T actual = row[x * channels + c];

        if (!checkValue(actual, expected)) {
          if (errorCount < 10) {
            std::cerr << "[D3D12 Only] Mismatch at x:" << x << " y:" << y
                      << " c:" << c << " | Expected: " << expected
                      << ", Got: " << actual << std::endl;
          }
          errorCount++;
          passed = false;
        }
      }
    }
  }
  readbackBuffer->Unmap(0, nullptr);

  if (passed)
    std::cout << "[D3D12 Only] Readback Verified Successfully." << std::endl;
  else
    std::cerr << "[D3D12 Only] Readback FAILED with " << errorCount
              << " errors." << std::endl;

  return passed;
}

// UAV IMAGE CREATION HELPERS
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
  texDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS |
                  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

  D3D12_HEAP_PROPERTIES defaultHeap = {D3D12_HEAP_TYPE_DEFAULT};
  ThrowIfFailed(ctx.device->CreateCommittedResource(
      &defaultHeap, D3D12_HEAP_FLAG_SHARED, &texDesc,
      D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&imgRes.resource)));

  auto allocInfo = ctx.device->GetResourceAllocationInfo(0, 1, &texDesc);
  imgRes.allocationSize = allocInfo.SizeInBytes;
  ThrowIfFailed(ctx.device->CreateSharedHandle(imgRes.resource.Get(), nullptr,
                                               GENERIC_ALL, nullptr,
                                               &imgRes.sharedHandle));

  return imgRes;
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

  D3D12Context ctx = createD3D12Context();
  D3D12ImageResources imgA =
      createExportableImage(ctx, width, height, dxgiFormat);
  D3D12ImageResources imgB =
      createExportableImage(ctx, width, height, dxgiFormat);
  D3D12ImageResources imgOut =
      createExportableImageWrite(ctx, width, height, dxgiFormat);

  D3D12ExportableFence extFenceA, extFenceB, extFenceOut;
  if (useSemaphores) {
    extFenceA = createExportableFence(ctx);
    extFenceB = createExportableFence(ctx);
    extFenceOut = createExportableFence(ctx);
  }

  uploadCustomData<T>(
      ctx, imgA, width, height, channels,
      [](size_t x, size_t y, int c) { return generateValueA<T>(x, y, c); });
  if (useSemaphores)
    signalExportableFence(ctx, extFenceA);

  uploadCustomData<T>(
      ctx, imgB, width, height, channels,
      [](size_t x, size_t y, int c) { return generateValueB<T>(x, y, c); });
  if (useSemaphores)
    signalExportableFence(ctx, extFenceB);

  try {
    sycl::queue q;

    auto extMemA = syclexp::import_external_memory(
        syclexp::external_mem_descriptor<syclexp::resource_win32_handle>{
            imgA.sharedHandle,
            syclexp::external_mem_handle_type::win32_nt_handle,
            imgA.allocationSize},
        q.get_device(), q.get_context());
    auto extMemB = syclexp::import_external_memory(
        syclexp::external_mem_descriptor<syclexp::resource_win32_handle>{
            imgB.sharedHandle,
            syclexp::external_mem_handle_type::win32_nt_handle,
            imgB.allocationSize},
        q.get_device(), q.get_context());
    auto extMemOut = syclexp::import_external_memory(
        syclexp::external_mem_descriptor<syclexp::resource_win32_handle>{
            imgOut.sharedHandle,
            syclexp::external_mem_handle_type::win32_nt_handle,
            imgOut.allocationSize},
        q.get_device(), q.get_context());

    syclexp::external_semaphore extSemA, extSemB, extSemOut;
    if (useSemaphores) {
      extSemA = syclexp::import_external_semaphore(
          syclexp::external_semaphore_descriptor<
              syclexp::resource_win32_handle>{
              extFenceA.sharedHandle,
              syclexp::external_semaphore_handle_type::win32_nt_handle},
          q.get_device(), q.get_context());
      extSemB = syclexp::import_external_semaphore(
          syclexp::external_semaphore_descriptor<
              syclexp::resource_win32_handle>{
              extFenceB.sharedHandle,
              syclexp::external_semaphore_handle_type::win32_nt_handle},
          q.get_device(), q.get_context());
      extSemOut = syclexp::import_external_semaphore(
          syclexp::external_semaphore_descriptor<
              syclexp::resource_win32_handle>{
              extFenceOut.sharedHandle,
              syclexp::external_semaphore_handle_type::win32_nt_handle},
          q.get_device(), q.get_context());
    }

    sycl::image_channel_type syclType = syclOverride.has_value()
                                            ? syclOverride.value()
                                            : getSyclChannelType<T>();
    syclexp::image_descriptor imgDesc(sycl::range<2>(width, height), channels,
                                      syclType);

    auto imgMemA = syclexp::map_external_image_memory(
        extMemA, imgDesc, q.get_device(), q.get_context());
    auto imgMemB = syclexp::map_external_image_memory(
        extMemB, imgDesc, q.get_device(), q.get_context());
    auto imgMemOut = syclexp::map_external_image_memory(
        extMemOut, imgDesc, q.get_device(), q.get_context());

    auto handleOut = syclexp::create_image(imgMemOut, imgDesc, q.get_device(),
                                           q.get_context());

    std::vector<sycl::event> waitEvents;
    if (useSemaphores) {
      waitEvents.push_back(q.submit([&](sycl::handler &h) {
        h.ext_oneapi_wait_external_semaphore(extSemA, extFenceA.fenceValue);
      }));
      waitEvents.push_back(q.submit([&](sycl::handler &h) {
        h.ext_oneapi_wait_external_semaphore(extSemB, extFenceB.fenceValue);
      }));
    }

    sycl::event kernelEvent;

    if (useSampled) {
      syclexp::bindless_image_sampler sampler(
          sycl::addressing_mode::clamp_to_edge,
          sycl::coordinate_normalization_mode::unnormalized,
          sycl::filtering_mode::nearest);
      auto handleA = syclexp::create_image(imgMemA, sampler, imgDesc,
                                           q.get_device(), q.get_context());
      auto handleB = syclexp::create_image(imgMemB, sampler, imgDesc,
                                           q.get_device(), q.get_context());

      kernelEvent = q.submit([&](sycl::handler &h) {
        h.depends_on(waitEvents);
        h.parallel_for(sycl::range<2>(width, height), [=](sycl::item<2> item) {
          int x = item.get_id(0);
          int y = item.get_id(1);

          bool isUnorm = (syclType == sycl::image_channel_type::unorm_int8);
          using Vec4 = sycl::vec<float, 4>;
          Vec4 valA(0, 0, 0, 0);
          Vec4 valB(0, 0, 0, 0);

          if (isUnorm) {
            valA = syclexp::sample_image<Vec4>(
                handleA, sycl::float2(x + 0.5f, y + 0.5f));
            valB = syclexp::sample_image<Vec4>(
                handleB, sycl::float2(x + 0.5f, y + 0.5f));
          } else {
            using SampleT = sycl::vec<T, 4>;
            auto rawA = syclexp::sample_image<SampleT>(
                handleA, sycl::float2(x + 0.5f, y + 0.5f));
            auto rawB = syclexp::sample_image<SampleT>(
                handleB, sycl::float2(x + 0.5f, y + 0.5f));
            valA = {(float)rawA.x(), (float)rawA.y(), (float)rawA.z(),
                    (float)rawA.w()};
            valB = {(float)rawB.x(), (float)rawB.y(), (float)rawB.z(),
                    (float)rawB.w()};
          }

          Vec4 sum = valA + valB;
          if (isUnorm) {
            sum = sycl::clamp(sum, 0.0f, 1.0f);
            if (channels == 1)
              syclexp::write_image(handleOut, sycl::int2(x, y), sum.x());
            else if (channels == 2)
              syclexp::write_image(handleOut, sycl::int2(x, y),
                                   sycl::float2(sum.x(), sum.y()));
            else
              syclexp::write_image(handleOut, sycl::int2(x, y), sum);
          } else {
            if (channels == 1)
              syclexp::write_image(handleOut, sycl::int2(x, y),
                                   static_cast<T>(sum.x()));
            else if (channels == 2)
              syclexp::write_image(handleOut, sycl::int2(x, y),
                                   sycl::vec<T, 2>(static_cast<T>(sum.x()),
                                                   static_cast<T>(sum.y())));
            else
              syclexp::write_image(handleOut, sycl::int2(x, y),
                                   sycl::vec<T, 4>(static_cast<T>(sum.x()),
                                                   static_cast<T>(sum.y()),
                                                   static_cast<T>(sum.z()),
                                                   static_cast<T>(sum.w())));
          }
        });
      });
      syclexp::destroy_image_handle(handleA, q.get_device(), q.get_context());
      syclexp::destroy_image_handle(handleB, q.get_device(), q.get_context());
    } else {
      auto handleA = syclexp::create_image(imgMemA, imgDesc, q.get_device(),
                                           q.get_context());
      auto handleB = syclexp::create_image(imgMemB, imgDesc, q.get_device(),
                                           q.get_context());

      kernelEvent = q.submit([&](sycl::handler &h) {
        h.depends_on(waitEvents);
        h.parallel_for(sycl::range<2>(width, height), [=](sycl::item<2> item) {
          int x = item.get_id(0);
          int y = item.get_id(1);

          bool isUnorm = (syclType == sycl::image_channel_type::unorm_int8);
          using Vec4 = sycl::vec<float, 4>;
          Vec4 valA(0, 0, 0, 0);
          Vec4 valB(0, 0, 0, 0);

          if (isUnorm) {
            valA = syclexp::fetch_image<Vec4>(handleA, sycl::int2(x, y));
            valB = syclexp::fetch_image<Vec4>(handleB, sycl::int2(x, y));
          } else {
            auto fetchT = [&](auto &hdl) {
              Vec4 v(0, 0, 0, 0);
              if (channels == 1)
                v.x() = (float)syclexp::fetch_image<T>(hdl, sycl::int2(x, y));
              else {
                auto raw = syclexp::fetch_image<sycl::vec<T, 4>>(
                    hdl, sycl::int2(x, y));
                v.x() = (float)raw.x();
                v.y() = (float)raw.y();
                v.z() = (float)raw.z();
                v.w() = (float)raw.w();
              }
              return v;
            };
            valA = fetchT(handleA);
            valB = fetchT(handleB);
          }

          Vec4 sum = valA + valB;
          if (isUnorm) {
            sum = sycl::clamp(sum, 0.0f, 1.0f);
            if (channels == 1)
              syclexp::write_image(handleOut, sycl::int2(x, y), sum.x());
            else if (channels == 2)
              syclexp::write_image(handleOut, sycl::int2(x, y),
                                   sycl::float2(sum.x(), sum.y()));
            else
              syclexp::write_image(handleOut, sycl::int2(x, y), sum);
          } else {
            if (channels == 1)
              syclexp::write_image(handleOut, sycl::int2(x, y),
                                   static_cast<T>(sum.x()));
            else if (channels == 2)
              syclexp::write_image(handleOut, sycl::int2(x, y),
                                   sycl::vec<T, 2>(static_cast<T>(sum.x()),
                                                   static_cast<T>(sum.y())));
            else
              syclexp::write_image(handleOut, sycl::int2(x, y),
                                   sycl::vec<T, 4>(static_cast<T>(sum.x()),
                                                   static_cast<T>(sum.y()),
                                                   static_cast<T>(sum.z()),
                                                   static_cast<T>(sum.w())));
          }
        });
      });
      syclexp::destroy_image_handle(handleA, q.get_device(), q.get_context());
      syclexp::destroy_image_handle(handleB, q.get_device(), q.get_context());
    }

    if (useSemaphores) {
      q.submit([&](sycl::handler &h) {
        h.depends_on(kernelEvent);
        h.ext_oneapi_signal_external_semaphore(extSemOut,
                                               extFenceOut.fenceValue + 1);
      });
    }
    q.wait();

    syclexp::destroy_image_handle(handleOut, q.get_device(), q.get_context());
    syclexp::release_external_memory(extMemA, q.get_device(), q.get_context());
    syclexp::release_external_memory(extMemB, q.get_device(), q.get_context());
    syclexp::release_external_memory(extMemOut, q.get_device(),
                                     q.get_context());

    if (useSemaphores) {
      syclexp::release_external_semaphore(extSemA, q.get_device(),
                                          q.get_context());
      syclexp::release_external_semaphore(extSemB, q.get_device(),
                                          q.get_context());
      syclexp::release_external_semaphore(extSemOut, q.get_device(),
                                          q.get_context());
    }

  } catch (std::exception &e) {
    std::cerr << "SYCL Exception: " << e.what() << std::endl;
    cleanupD3D12(ctx, imgA);
    cleanupD3D12(ctx, imgB);
    cleanupD3D12(ctx, imgOut);
    if (useSemaphores) {
      cleanupExportableFence(extFenceA);
      cleanupExportableFence(extFenceB);
      cleanupExportableFence(extFenceOut);
    }
    return 1;
  }

  if (useSemaphores) {
    ctx.cmdQueue->Wait(extFenceOut.fence.Get(), extFenceOut.fenceValue + 1);
    extFenceOut.fenceValue++;
  }

  bool passed = verifyCustomData<T>(
      ctx, imgOut, width, height, channels, [&](size_t x, size_t y, int c) {
        T a = generateValueA<T>(x, y, c);
        T b = generateValueB<T>(x, y, c);
        if (syclOverride.has_value() &&
            syclOverride.value() == sycl::image_channel_type::unorm_int8) {
          float fa = (float)a / 255.0f;
          float fb = (float)b / 255.0f;
          float sum = (std::min)(fa + fb, 1.0f);
          return static_cast<T>(sum * 255.0f + 0.5f);
        } else {
          return static_cast<T>(a + b);
        }
      });

  if (useSemaphores) {
    cleanupExportableFence(extFenceA);
    cleanupExportableFence(extFenceB);
    cleanupExportableFence(extFenceOut);
  }
  cleanupD3D12(ctx, imgA);
  cleanupD3D12(ctx, imgB);
  cleanupD3D12(ctx, imgOut);

  return passed ? 0 : 1;
}

// ---------------------------------------------------------
// MAIN
// ---------------------------------------------------------
int main(int argc, char **argv) {
  int width = 4;
  int height = 4;
  int channels = 4;
  bool useSemaphores = false;
  bool useSampled = false;
  std::string type = "float";

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--sampled")
      useSampled = true;
    else if (arg == "--semaphores")
      useSemaphores = true;
    else if (arg == "--channels" && i + 1 < argc)
      channels = std::stoi(argv[++i]);
    else if (arg == "--type" && i + 1 < argc)
      type = argv[++i];
    else if (arg.find('x') != std::string::npos) {
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
            << " D3D12 2D ARITHMETIC Test (C = A + B) | Type: " << type
            << " | Size: " << width << "x" << height
            << " | Channels: " << channels
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