// REQUIRES: aspect-ext_oneapi_bindless_images
// REQUIRES: aspect-ext_oneapi_external_memory_import
// REQUIRES: windows

// DG2 accesses imported textures as if they were uncompressed.
// XFAIL: windows && run-mode && gpu-intel-dg2
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/21985

// RUN: %{build} -o %t.exe %link-directx

// Fills a texture with D3D12 (copy or clear), reads it and writes it in place
// with a SYCL kernel, and checks the texture with D3D12. 32x33 is too small to
// be compressed, and 1366 isn't a multiple of the tile width.

// clang-format off
/*
  Standalone build:
    clang++ -fsycl -o ds2rw.exe D3D12_sycl_interop_2D_read_write.cpp -ld3d12 -ldxgi

  Usage: ds2rw.exe [options] [WxH]
    --type T         Channel type: float, half, unorm8, snorm8, unorm16 or
                     snorm16 (default: float)
    --channels N     Number of channels: 1, 2 or 4 (default: 4)
    --clear          Fill the texture with ClearRenderTargetView (a constant
                     color) instead of copying a pattern from a buffer
    --semaphores     Synchronize D3D12 and SYCL on the GPU with a shared fence
                     instead of waiting on the host
    --dx12-resource  Import the texture with the win32_nt_dx12_resource handle
                     type instead of win32_nt_handle
    WxH              Texture size (default: 32x33)
*/
// RUN: %{run} %t.exe --type float --channels 4 32x33
// RUN: %{run} %t.exe --type unorm8 --channels 4 --clear 32x33
// RUN: %{run} %t.exe --type float --channels 1 1920x1080
// RUN: %{run} %t.exe --type float --channels 2 1366x768
// RUN: %{run} %t.exe --type float --channels 4 1920x1080
// RUN: %{run} %t.exe --type float --channels 1 --clear 1366x768
// RUN: %{run} %t.exe --type half --channels 1 1366x768
// RUN: %{run} %t.exe --type half --channels 2 --dx12-resource 1920x1080
// RUN: %{run} %t.exe --type half --channels 4 --clear 3840x2160
// RUN: %{run} %t.exe --type unorm8 --channels 1 1920x1080
// RUN: %{run} %t.exe --type unorm8 --channels 2 --clear 1366x768
// RUN: %{run} %t.exe --type unorm8 --channels 4 1920x1080
// RUN: %{run} %t.exe --type snorm8 --channels 1 1366x768
// RUN: %{run} %t.exe --type snorm8 --channels 2 1920x1080
// RUN: %{run} %t.exe --type snorm8 --channels 4 --clear --dx12-resource 1920x1080
// RUN: %{run} %t.exe --type unorm16 --channels 1 1920x1080
// RUN: %{run} %t.exe --type unorm16 --channels 2 --clear 1920x1080
// RUN: %{run} %t.exe --type unorm16 --channels 4 1366x768
// RUN: %{run} %t.exe --type snorm16 --channels 1 --clear 1920x1080
// RUN: %{run} %t.exe --type snorm16 --channels 2 1366x768
// RUN: %{run} %t.exe --type snorm16 --channels 4 1920x1080
// RUN: %{run} %t.exe --type float --channels 4 --semaphores 1920x1080
// RUN: %{run} %t.exe --type unorm8 --channels 4 --clear --semaphores 1920x1080
// clang-format on

#include "../helpers/interop_read_write.hpp"
#include "d3d12_setup.hpp"

#include <sycl/properties/queue_properties.hpp>
#include <sycl/usm.hpp>

#include <array>

using namespace interop_read_write;

DXGI_FORMAT getFormat(const std::string &type, int channels) {
  const int i = channels == 1 ? 0 : channels == 2 ? 1 : 2;
  if (type == "float")
    return std::array{DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R32G32_FLOAT,
                      DXGI_FORMAT_R32G32B32A32_FLOAT}[i];
  if (type == "half")
    return std::array{DXGI_FORMAT_R16_FLOAT, DXGI_FORMAT_R16G16_FLOAT,
                      DXGI_FORMAT_R16G16B16A16_FLOAT}[i];
  if (type == "unorm8")
    return std::array{DXGI_FORMAT_R8_UNORM, DXGI_FORMAT_R8G8_UNORM,
                      DXGI_FORMAT_R8G8B8A8_UNORM}[i];
  if (type == "snorm8")
    return std::array{DXGI_FORMAT_R8_SNORM, DXGI_FORMAT_R8G8_SNORM,
                      DXGI_FORMAT_R8G8B8A8_SNORM}[i];
  if (type == "unorm16")
    return std::array{DXGI_FORMAT_R16_UNORM, DXGI_FORMAT_R16G16_UNORM,
                      DXGI_FORMAT_R16G16B16A16_UNORM}[i];
  if (type == "snorm16")
    return std::array{DXGI_FORMAT_R16_SNORM, DXGI_FORMAT_R16G16_SNORM,
                      DXGI_FORMAT_R16G16B16A16_SNORM}[i];
  return DXGI_FORMAT_UNKNOWN;
}

D3D12ImageResources createTexture(D3D12Context &ctx, int width, int height,
                                  DXGI_FORMAT format, bool renderTarget) {
  D3D12_RESOURCE_DESC texDesc = {};
  texDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
  texDesc.Width = width;
  texDesc.Height = height;
  texDesc.DepthOrArraySize = 1;
  texDesc.MipLevels = 1;
  texDesc.Format = format;
  texDesc.SampleDesc.Count = 1;
  texDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
  // Not ALLOW_SIMULTANEOUS_ACCESS: it disables compression of the texture
  texDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
  if (renderTarget)
    texDesc.Flags |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;

  D3D12_HEAP_PROPERTIES defaultHeap = {D3D12_HEAP_TYPE_DEFAULT};
  D3D12ImageResources imgRes;
  ThrowIfFailed(ctx.device->CreateCommittedResource(
                    &defaultHeap, D3D12_HEAP_FLAG_SHARED, &texDesc,
                    D3D12_RESOURCE_STATE_COMMON, nullptr,
                    IID_PPV_ARGS(&imgRes.resource)),
                "Failed to create Shared Texture");
  imgRes.allocationSize =
      ctx.device->GetResourceAllocationInfo(0, 1, &texDesc).SizeInBytes;
  ThrowIfFailed(ctx.device->CreateSharedHandle(imgRes.resource.Get(), nullptr,
                                               GENERIC_ALL, nullptr,
                                               &imgRes.sharedHandle),
                "Failed to export NT Handle");
  return imgRes;
}

ComPtr<ID3D12Resource> createBuffer(D3D12Context &ctx, D3D12_HEAP_TYPE type,
                                    UINT64 size) {
  D3D12_RESOURCE_DESC bufDesc = {};
  bufDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  bufDesc.Width = size;
  bufDesc.Height = 1;
  bufDesc.DepthOrArraySize = 1;
  bufDesc.MipLevels = 1;
  bufDesc.Format = DXGI_FORMAT_UNKNOWN;
  bufDesc.SampleDesc.Count = 1;
  bufDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  D3D12_HEAP_PROPERTIES heap = {type};
  ComPtr<ID3D12Resource> buffer;
  ThrowIfFailed(ctx.device->CreateCommittedResource(
                    &heap, D3D12_HEAP_FLAG_NONE, &bufDesc,
                    type == D3D12_HEAP_TYPE_UPLOAD
                        ? D3D12_RESOURCE_STATE_GENERIC_READ
                        : D3D12_RESOURCE_STATE_COPY_DEST,
                    nullptr, IID_PPV_ARGS(&buffer)),
                "Failed to create Buffer");
  return buffer;
}

D3D12_PLACED_SUBRESOURCE_FOOTPRINT
getFootprint(D3D12Context &ctx, ID3D12Resource *texture, UINT64 &totalBytes) {
  const D3D12_RESOURCE_DESC desc = texture->GetDesc();
  D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint;
  ctx.device->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, nullptr,
                                    nullptr, &totalBytes);
  return footprint;
}

// Records commands that fill the texture and leave it in the COMMON state.
// The upload buffer and the RTV heap must be kept alive until they complete.
void recordFill(D3D12Context &ctx, ID3D12Resource *texture,
                const ChannelFormat &f, int width, int height, bool clear,
                ComPtr<ID3D12Resource> &uploadBuffer,
                ComPtr<ID3D12DescriptorHeap> &rtvHeap) {
  if (clear) {
    D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
    heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    heapDesc.NumDescriptors = 1;
    ThrowIfFailed(
        ctx.device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&rtvHeap)),
        "Failed to create RTV Heap");
    const D3D12_CPU_DESCRIPTOR_HANDLE rtv =
        rtvHeap->GetCPUDescriptorHandleForHeapStart();
    ctx.device->CreateRenderTargetView(texture, nullptr, rtv);
    transitionResource(ctx, texture, D3D12_RESOURCE_STATE_COMMON,
                       D3D12_RESOURCE_STATE_RENDER_TARGET);
    ctx.cmdList->ClearRenderTargetView(rtv, clearColor, 0, nullptr);
    transitionResource(ctx, texture, D3D12_RESOURCE_STATE_RENDER_TARGET,
                       D3D12_RESOURCE_STATE_COMMON);
    return;
  }

  UINT64 totalBytes;
  const D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint =
      getFootprint(ctx, texture, totalBytes);
  uploadBuffer = createBuffer(ctx, D3D12_HEAP_TYPE_UPLOAD, totalBytes);
  uint8_t *data;
  ThrowIfFailed(
      uploadBuffer->Map(0, nullptr, reinterpret_cast<void **>(&data)));
  packPattern(f, width, height, data + footprint.Offset,
              footprint.Footprint.RowPitch);
  uploadBuffer->Unmap(0, nullptr);

  D3D12_TEXTURE_COPY_LOCATION dst = {};
  dst.pResource = texture;
  dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  dst.SubresourceIndex = 0;
  D3D12_TEXTURE_COPY_LOCATION src = {};
  src.pResource = uploadBuffer.Get();
  src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
  src.PlacedFootprint = footprint;
  transitionResource(ctx, texture, D3D12_RESOURCE_STATE_COMMON,
                     D3D12_RESOURCE_STATE_COPY_DEST);
  ctx.cmdList->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
  transitionResource(ctx, texture, D3D12_RESOURCE_STATE_COPY_DEST,
                     D3D12_RESOURCE_STATE_COMMON);
}

std::vector<uint32_t> readback(D3D12Context &ctx, ID3D12Resource *texture,
                               const ChannelFormat &f, int width, int height) {
  UINT64 totalBytes;
  const D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint =
      getFootprint(ctx, texture, totalBytes);
  ComPtr<ID3D12Resource> readbackBuffer =
      createBuffer(ctx, D3D12_HEAP_TYPE_READBACK, totalBytes);

  ThrowIfFailed(ctx.cmdAlloc->Reset());
  ThrowIfFailed(ctx.cmdList->Reset(ctx.cmdAlloc.Get(), nullptr));
  D3D12_TEXTURE_COPY_LOCATION src = {};
  src.pResource = texture;
  src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  src.SubresourceIndex = 0;
  D3D12_TEXTURE_COPY_LOCATION dst = {};
  dst.pResource = readbackBuffer.Get();
  dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
  dst.PlacedFootprint = footprint;
  transitionResource(ctx, texture, D3D12_RESOURCE_STATE_COMMON,
                     D3D12_RESOURCE_STATE_COPY_SOURCE);
  ctx.cmdList->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
  transitionResource(ctx, texture, D3D12_RESOURCE_STATE_COPY_SOURCE,
                     D3D12_RESOURCE_STATE_COMMON);
  ThrowIfFailed(ctx.cmdList->Close());
  executeAndWait(ctx);

  uint8_t *data;
  ThrowIfFailed(
      readbackBuffer->Map(0, nullptr, reinterpret_cast<void **>(&data)));
  std::vector<uint32_t> raw = unpack(f, width, height, data + footprint.Offset,
                                     footprint.Footprint.RowPitch);
  readbackBuffer->Unmap(0, nullptr);
  return raw;
}

int runTest(const ChannelFormat &f, DXGI_FORMAT format, int width, int height,
            bool clear, bool useSemaphores,
            syclexp::external_mem_handle_type handleType) {
  D3D12Context ctx = createD3D12Context();
  D3D12ImageResources imgRes = createTexture(ctx, width, height, format, clear);

  ThrowIfFailed(ctx.cmdAlloc->Reset());
  ThrowIfFailed(ctx.cmdList->Reset(ctx.cmdAlloc.Get(), nullptr));
  ComPtr<ID3D12Resource> uploadBuffer;
  ComPtr<ID3D12DescriptorHeap> rtvHeap;
  recordFill(ctx, imgRes.resource.Get(), f, width, height, clear, uploadBuffer,
             rtvHeap);
  ThrowIfFailed(ctx.cmdList->Close());

  D3D12ExportableFence extFence;
  if (useSemaphores) {
    // SYCL waits for the fill on the device
    extFence = createExportableFence(ctx);
    ID3D12CommandList *lists[] = {ctx.cmdList.Get()};
    ctx.cmdQueue->ExecuteCommandLists(1, lists);
    signalExportableFence(ctx, extFence);
    ThrowIfFailed(ctx.cmdQueue->Signal(ctx.fence.Get(), ++ctx.fenceValue),
                  "Queue Signal Failed");
  } else {
    executeAndWait(ctx);
  }

  size_t errors = 0;
  try {
    // External semaphore operations require immediate command lists
    sycl::property_list queueProps =
        useSemaphores ? sycl::property_list{sycl::property::queue::in_order{},
                                            sycl::ext::intel::property::queue::
                                                immediate_command_list{}}
                      : sycl::property_list{sycl::property::queue::in_order{}};
    sycl::queue q{queueProps};

    syclexp::external_mem_descriptor<syclexp::resource_win32_handle> memDesc{
        imgRes.sharedHandle, handleType, imgRes.allocationSize};
    syclexp::external_mem extMem = syclexp::import_external_memory(memDesc, q);
    syclexp::image_descriptor imgDesc(sycl::range<2>(width, height), f.channels,
                                      f.channelType);
    syclexp::image_mem_handle memHandle =
        syclexp::map_external_image_memory(extMem, imgDesc, q);
    syclexp::unsampled_image_handle image =
        syclexp::create_image(memHandle, imgDesc, q);

    syclexp::external_semaphore extSem;
    sycl::event waitEvent;
    if (useSemaphores) {
      syclexp::external_semaphore_descriptor<syclexp::resource_win32_handle>
          semDesc{extFence.sharedHandle,
                  syclexp::external_semaphore_handle_type::win32_nt_dx12_fence};
      extSem = syclexp::import_external_semaphore(semDesc, q);
      waitEvent =
          q.ext_oneapi_wait_external_semaphore(extSem, extFence.fenceValue);
    }

    const size_t numValues = size_t(width) * height * f.channels;
    float *fetched = sycl::malloc_shared<float>(numValues, q);
    std::fill(fetched, fetched + numValues, -12345.f);
    sycl::event kernelEvent =
        readModifyWrite(q, f, image, width, height, fetched, waitEvent);

    if (useSemaphores) {
      // D3D12 waits for the kernel on the device
      q.ext_oneapi_signal_external_semaphore(extSem, extFence.fenceValue + 1,
                                             kernelEvent);
      ThrowIfFailed(
          ctx.cmdQueue->Wait(extFence.fence.Get(), ++extFence.fenceValue),
          "Failed to wait for shared fence");
      // The fill must complete before its command allocator is reset
      ThrowIfFailed(
          ctx.fence->SetEventOnCompletion(ctx.fenceValue, ctx.fenceEvent),
          "SetEventOnCompletion Failed");
      WaitForSingleObject(ctx.fenceEvent, INFINITE);
    } else {
      q.wait_and_throw();
    }

    const std::vector<uint32_t> written =
        readback(ctx, imgRes.resource.Get(), f, width, height);
    q.wait_and_throw();
    errors = checkResults(f, width, height, clear, fetched, written);

    sycl::free(fetched, q);
    syclexp::destroy_image_handle(image, q);
    syclexp::unmap_external_image_memory(memHandle,
                                         syclexp::image_type::standard, q);
    syclexp::release_external_memory(extMem, q);
    if (useSemaphores)
      syclexp::release_external_semaphore(extSem, q);
  } catch (std::exception &e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    errors = 1;
  }

  if (useSemaphores)
    cleanupExportableFence(extFence);
  cleanupD3D12(ctx, imgRes);

  if (errors) {
    std::cerr << "FAILURE! " << errors << " errors" << std::endl;
    return 1;
  }
  std::cout << "Test passed" << std::endl;
  return 0;
}

int main(int argc, char **argv) {
  int width = 32;
  int height = 33;
  int channels = 4;
  bool clear = false;
  bool useSemaphores = false;
  auto handleType = syclexp::external_mem_handle_type::win32_nt_handle;
  std::string type = "float";

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--clear") {
      clear = true;
    } else if (arg == "--semaphores") {
      useSemaphores = true;
    } else if (arg == "--dx12-resource") {
      handleType = syclexp::external_mem_handle_type::win32_nt_dx12_resource;
    } else if (arg == "--channels" && i + 1 < argc) {
      channels = std::stoi(argv[++i]);
    } else if (arg == "--type" && i + 1 < argc) {
      type = argv[++i];
    } else if (arg.find('x') != std::string::npos) {
      size_t pos = arg.find('x');
      width = std::stoi(arg.substr(0, pos));
      height = std::stoi(arg.substr(pos + 1));
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      return 1;
    }
  }

  if (channels != 1 && channels != 2 && channels != 4) {
    std::cerr << "Error: Only 1, 2, or 4 channels supported." << std::endl;
    return 1;
  }
  std::optional<ChannelFormat> format = getChannelFormat(type, channels);
  if (!format) {
    std::cerr << "Unknown type: " << type << std::endl;
    return 1;
  }

  std::cout << "Running D3D12 2D Read Write Test | Type: " << type
            << " | Size: " << width << "x" << height
            << " | Channels: " << channels
            << " | Fill: " << (clear ? "clear" : "copy")
            << " | Semaphores: " << (useSemaphores ? "ON" : "OFF") << std::endl;

  return runTest(*format, getFormat(type, channels), width, height, clear,
                 useSemaphores, handleType);
}
