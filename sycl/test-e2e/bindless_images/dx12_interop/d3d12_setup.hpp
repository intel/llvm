#pragma once

#include <d3d12.h>
#include <dxgi1_6.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <windows.h>
#include <wrl/client.h>

using Microsoft::WRL::ComPtr;

// ---------------------------------------------------------
// ERROR HANDLING
// ---------------------------------------------------------
inline void ThrowIfFailed(HRESULT hr, const std::string &msg = "D3D12 Error") {
  if (FAILED(hr)) {
    throw std::runtime_error(msg + " (HRESULT: " + std::to_string(hr) + ")");
  }
}

// ---------------------------------------------------------
// DXGI FORMAT MAPPING
// ---------------------------------------------------------
template <typename T> DXGI_FORMAT getDXGIFormat(int channels);

template <> inline DXGI_FORMAT getDXGIFormat<float>(int channels) {
  if (channels == 1)
    return DXGI_FORMAT_R32_FLOAT;
  if (channels == 2)
    return DXGI_FORMAT_R32G32_FLOAT;
  if (channels == 4)
    return DXGI_FORMAT_R32G32B32A32_FLOAT;
  throw std::runtime_error("Unsupported channels for float");
}

template <> inline DXGI_FORMAT getDXGIFormat<int32_t>(int channels) {
  if (channels == 1)
    return DXGI_FORMAT_R32_SINT;
  if (channels == 2)
    return DXGI_FORMAT_R32G32_SINT;
  if (channels == 4)
    return DXGI_FORMAT_R32G32B32A32_SINT;
  throw std::runtime_error("Unsupported channels for int32");
}

template <> inline DXGI_FORMAT getDXGIFormat<uint32_t>(int channels) {
  if (channels == 1)
    return DXGI_FORMAT_R32_UINT;
  if (channels == 2)
    return DXGI_FORMAT_R32G32_UINT;
  if (channels == 4)
    return DXGI_FORMAT_R32G32B32A32_UINT;
  throw std::runtime_error("Unsupported channels for uint32");
}

template <> inline DXGI_FORMAT getDXGIFormat<int16_t>(int channels) {
  if (channels == 1)
    return DXGI_FORMAT_R16_SINT;
  if (channels == 2)
    return DXGI_FORMAT_R16G16_SINT;
  if (channels == 4)
    return DXGI_FORMAT_R16G16B16A16_SINT;
  throw std::runtime_error("Unsupported channels for int16");
}

template <> inline DXGI_FORMAT getDXGIFormat<uint16_t>(int channels) {
  if (channels == 1)
    return DXGI_FORMAT_R16_UINT;
  if (channels == 2)
    return DXGI_FORMAT_R16G16_UINT;
  if (channels == 4)
    return DXGI_FORMAT_R16G16B16A16_UINT;
  throw std::runtime_error("Unsupported channels for uint16");
}

template <> inline DXGI_FORMAT getDXGIFormat<int8_t>(int channels) {
  if (channels == 1)
    return DXGI_FORMAT_R8_SINT;
  if (channels == 2)
    return DXGI_FORMAT_R8G8_SINT;
  if (channels == 4)
    return DXGI_FORMAT_R8G8B8A8_SINT;
  throw std::runtime_error("Unsupported channels for int8");
}

template <> inline DXGI_FORMAT getDXGIFormat<uint8_t>(int channels) {
  if (channels == 1)
    return DXGI_FORMAT_R8_UINT;
  if (channels == 2)
    return DXGI_FORMAT_R8G8_UINT;
  if (channels == 4)
    return DXGI_FORMAT_R8G8B8A8_UINT;
  throw std::runtime_error("Unsupported channels for uint8");
}

inline DXGI_FORMAT getUnorm8Format(int channels) {
  if (channels == 1)
    return DXGI_FORMAT_R8_UNORM;
  if (channels == 2)
    return DXGI_FORMAT_R8G8_UNORM;
  if (channels == 4)
    return DXGI_FORMAT_R8G8B8A8_UNORM;
  throw std::runtime_error("Unsupported channels for unorm8");
}

// ---------------------------------------------------------
// DATA GENERATION & VERIFICATION
// ---------------------------------------------------------
template <typename T>
T generateTestValue(size_t pixelIdx, int channelIdx, size_t totalPixels) {
  // Simple gradient generation for testing
  float val = static_cast<float>(pixelIdx % 256) + (channelIdx * 10.0f);
  if constexpr (std::is_integral_v<T>) {
    return static_cast<T>(static_cast<int>(val) %
                          127); // Keep it safe for int8/unorm
  }
  return static_cast<T>(val);
}

template <typename T> bool checkValue(T actual, T expected) {
  if constexpr (std::is_floating_point_v<T>) {
    return std::abs(actual - expected) < 0.001f;
  }
  return actual == expected;
}

// ---------------------------------------------------------
// D3D12 STRUCTURES
// ---------------------------------------------------------
struct D3D12Context {
  ComPtr<ID3D12Device> device;
  ComPtr<ID3D12CommandQueue> cmdQueue;
  ComPtr<ID3D12CommandAllocator> cmdAlloc;
  ComPtr<ID3D12GraphicsCommandList> cmdList;
  ComPtr<ID3D12Fence> fence;
  uint64_t fenceValue = 0;
  HANDLE fenceEvent = nullptr;
};

struct D3D12ImageResources {
  ComPtr<ID3D12Resource> resource;
  HANDLE sharedHandle = nullptr;
  size_t allocationSize = 0;
};

// ---------------------------------------------------------
// D3D12 LIFECYCLE MANAGEMENT
// ---------------------------------------------------------
inline D3D12Context createD3D12Context() {
  D3D12Context ctx;

  ThrowIfFailed(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0,
                                  IID_PPV_ARGS(&ctx.device)),
                "Failed to create D3D12 Device");

  D3D12_COMMAND_QUEUE_DESC queueDesc = {};
  queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
  queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
  ThrowIfFailed(
      ctx.device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&ctx.cmdQueue)),
      "Failed to create Command Queue");

  ThrowIfFailed(
      ctx.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                         IID_PPV_ARGS(&ctx.cmdAlloc)),
      "Failed to create Command Allocator");
  ThrowIfFailed(ctx.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                              ctx.cmdAlloc.Get(), nullptr,
                                              IID_PPV_ARGS(&ctx.cmdList)),
                "Failed to create Command List");

  // Command list starts in recording state, close it for now
  ctx.cmdList->Close();

  ThrowIfFailed(ctx.device->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                        IID_PPV_ARGS(&ctx.fence)),
                "Failed to create Fence");
  ctx.fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
  if (!ctx.fenceEvent)
    throw std::runtime_error("Failed to create Fence Event");

  return ctx;
}

inline void executeAndWait(D3D12Context &ctx) {
  ID3D12CommandList *ppCommandLists[] = {ctx.cmdList.Get()};
  ctx.cmdQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);

  ctx.fenceValue++;
  ThrowIfFailed(ctx.cmdQueue->Signal(ctx.fence.Get(), ctx.fenceValue),
                "Queue Signal Failed");

  if (ctx.fence->GetCompletedValue() < ctx.fenceValue) {
    ThrowIfFailed(
        ctx.fence->SetEventOnCompletion(ctx.fenceValue, ctx.fenceEvent),
        "SetEventOnCompletion Failed");
    WaitForSingleObject(ctx.fenceEvent, INFINITE);
  }
}

inline void cleanupD3D12(D3D12Context &ctx, D3D12ImageResources &imgRes) {
  if (imgRes.sharedHandle)
    CloseHandle(imgRes.sharedHandle);
  if (ctx.fenceEvent)
    CloseHandle(ctx.fenceEvent);
  // ComPtrs will automatically release D3D12 objects when they go out of scope
}

// ---------------------------------------------------------
// RESOURCE CREATION & UPLOAD
// ---------------------------------------------------------
inline D3D12ImageResources createExportableImage(D3D12Context &ctx,
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
  texDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;

  D3D12_HEAP_PROPERTIES defaultHeap = {D3D12_HEAP_TYPE_DEFAULT};

  ThrowIfFailed(ctx.device->CreateCommittedResource(
                    &defaultHeap, D3D12_HEAP_FLAG_SHARED, &texDesc,
                    D3D12_RESOURCE_STATE_COMMON, nullptr,
                    IID_PPV_ARGS(&imgRes.resource)),
                "Failed to create Shared Texture");

  auto allocInfo = ctx.device->GetResourceAllocationInfo(0, 1, &texDesc);
  imgRes.allocationSize = allocInfo.SizeInBytes;

  ThrowIfFailed(ctx.device->CreateSharedHandle(imgRes.resource.Get(), nullptr,
                                               GENERIC_ALL, nullptr,
                                               &imgRes.sharedHandle),
                "Failed to export NT Handle");

  return imgRes;
}

inline D3D12ImageResources
createExportableImage1D(D3D12Context &ctx, uint32_t width, DXGI_FORMAT format) {
  D3D12ImageResources imgRes;

  D3D12_RESOURCE_DESC texDesc = {};
  texDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE1D;
  texDesc.Width = width;
  texDesc.Height = 1;
  texDesc.DepthOrArraySize = 1;
  texDesc.MipLevels = 1;
  texDesc.Format = format;
  texDesc.SampleDesc.Count = 1;
  texDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
  texDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;

  D3D12_HEAP_PROPERTIES defaultHeap = {D3D12_HEAP_TYPE_DEFAULT};

  ThrowIfFailed(ctx.device->CreateCommittedResource(
                    &defaultHeap, D3D12_HEAP_FLAG_SHARED, &texDesc,
                    D3D12_RESOURCE_STATE_COMMON, nullptr,
                    IID_PPV_ARGS(&imgRes.resource)),
                "Failed to create Shared 1D Texture");

  auto allocInfo = ctx.device->GetResourceAllocationInfo(0, 1, &texDesc);
  imgRes.allocationSize = allocInfo.SizeInBytes;

  ThrowIfFailed(ctx.device->CreateSharedHandle(imgRes.resource.Get(), nullptr,
                                               GENERIC_ALL, nullptr,
                                               &imgRes.sharedHandle),
                "Failed to export NT Handle");

  return imgRes;
}

// ---------------------------------------------------------
// 1D UAV IMAGE CREATION
// ---------------------------------------------------------
inline D3D12ImageResources createExportableImageWrite1D(D3D12Context &ctx,
                                                        uint32_t width,
                                                        DXGI_FORMAT format) {
  D3D12ImageResources imgRes;
  D3D12_RESOURCE_DESC texDesc = {};
  texDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE1D;
  texDesc.Width = width;
  texDesc.Height = 1;
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
                    D3D12_RESOURCE_STATE_COMMON, nullptr,
                    IID_PPV_ARGS(&imgRes.resource)),
                "Failed to create Shared UAV 1D Texture");

  auto allocInfo = ctx.device->GetResourceAllocationInfo(0, 1, &texDesc);
  imgRes.allocationSize = allocInfo.SizeInBytes;

  ThrowIfFailed(ctx.device->CreateSharedHandle(imgRes.resource.Get(), nullptr,
                                               GENERIC_ALL, nullptr,
                                               &imgRes.sharedHandle),
                "Failed to export NT Handle");

  return imgRes;
}

// ---------------------------------------------------------
// 3D IMAGE RESOURCES
// ---------------------------------------------------------
inline D3D12ImageResources
createExportableImage3D(D3D12Context &ctx, uint32_t width, uint32_t height,
                        uint32_t depth, DXGI_FORMAT format) {
  D3D12ImageResources imgRes;

  D3D12_RESOURCE_DESC texDesc = {};
  texDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
  texDesc.Width = width;
  texDesc.Height = height;
  texDesc.DepthOrArraySize = depth; // KEY: For 3D, this is the depth!
  texDesc.MipLevels = 1;
  texDesc.Format = format;
  texDesc.SampleDesc.Count = 1;
  texDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
  texDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;

  D3D12_HEAP_PROPERTIES defaultHeap = {D3D12_HEAP_TYPE_DEFAULT};

  ThrowIfFailed(ctx.device->CreateCommittedResource(
                    &defaultHeap, D3D12_HEAP_FLAG_SHARED, &texDesc,
                    D3D12_RESOURCE_STATE_COMMON, nullptr,
                    IID_PPV_ARGS(&imgRes.resource)),
                "Failed to create Shared 3D Texture");

  auto allocInfo = ctx.device->GetResourceAllocationInfo(0, 1, &texDesc);
  imgRes.allocationSize = allocInfo.SizeInBytes;

  ThrowIfFailed(ctx.device->CreateSharedHandle(imgRes.resource.Get(), nullptr,
                                               GENERIC_ALL, nullptr,
                                               &imgRes.sharedHandle),
                "Failed to export NT Handle");

  return imgRes;
}

template <typename T>
bool uploadTestData3D(D3D12Context &ctx, D3D12ImageResources &imgRes,
                      uint32_t width, uint32_t height, uint32_t depth,
                      int channels) {
  size_t totalPixels = width * height * depth;
  std::vector<T> hostData(totalPixels * channels);
  for (size_t i = 0; i < hostData.size(); ++i) {
    size_t pixelIdx = i / channels;
    int channelIdx = i % channels;
    hostData[i] = generateTestValue<T>(pixelIdx, channelIdx, totalPixels);
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
                    D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                    IID_PPV_ARGS(&uploadBuffer)),
                "Failed to create Upload Buffer");

  uint8_t *pData;
  ThrowIfFailed(
      uploadBuffer->Map(0, nullptr, reinterpret_cast<void **>(&pData)));
  const uint8_t *pSrc = reinterpret_cast<const uint8_t *>(hostData.data());

  // Slice pitch is the distance between 2D layers
  UINT slicePitch = footprint.Footprint.RowPitch * numRows;

  for (UINT z = 0; z < depth; ++z) {
    for (UINT y = 0; y < height; ++y) {
      memcpy(pData + (z * slicePitch) + (y * footprint.Footprint.RowPitch),
             pSrc + ((z * width * height) + (y * width)) * sizeof(T) * channels,
             width * sizeof(T) * channels);
    }
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

template <typename T>
bool verifyD3D12Data3D(D3D12Context &ctx, D3D12ImageResources &imgRes,
                       uint32_t width, uint32_t height, uint32_t depth,
                       int channels) {
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
  bufDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

  ThrowIfFailed(ctx.device->CreateCommittedResource(
                    &readbackHeap, D3D12_HEAP_FLAG_NONE, &bufDesc,
                    D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                    IID_PPV_ARGS(&readbackBuffer)),
                "Failed to create Readback Buffer");

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
  size_t totalPixels = width * height * depth;
  UINT slicePitch = footprint.Footprint.RowPitch * numRows;

  for (UINT z = 0; z < depth; ++z) {
    for (UINT y = 0; y < height; ++y) {
      T *row = reinterpret_cast<T *>(pData + (z * slicePitch) +
                                     (y * footprint.Footprint.RowPitch));
      for (UINT x = 0; x < width; ++x) {
        for (int c = 0; c < channels; ++c) {
          size_t pixelIdx = (z * width * height) + (y * width) + x;
          T expected = generateTestValue<T>(pixelIdx, c, totalPixels);
          T actual = row[x * channels + c];

          if (!checkValue(actual, expected)) {
            if (errorCount < 10) {
              std::cerr << "[D3D12 Only] Mismatch at x:" << x << " y:" << y
                        << " z:" << z << " c:" << c
                        << " | Expected: " << expected << ", Got: " << actual
                        << std::endl;
            }
            errorCount++;
            passed = false;
          }
        }
      }
    }
  }
  readbackBuffer->Unmap(0, nullptr);

  if (passed) {
    std::cout << "[D3D12 Only] Readback Verified Successfully." << std::endl;
  } else {
    std::cerr << "[D3D12 Only] Readback FAILED with " << errorCount
              << " errors." << std::endl;
  }

  return passed;
}

// ---------------------------------------------------------
// 3D UAV IMAGE CREATION
// ---------------------------------------------------------
inline D3D12ImageResources
createExportableImageWrite3D(D3D12Context &ctx, uint32_t width, uint32_t height,
                             uint32_t depth, DXGI_FORMAT format) {
  D3D12ImageResources imgRes;
  D3D12_RESOURCE_DESC texDesc = {};
  texDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
  texDesc.Width = width;
  texDesc.Height = height;
  texDesc.DepthOrArraySize = depth;
  texDesc.MipLevels = 1;
  texDesc.Format = format;
  texDesc.SampleDesc.Count = 1;
  texDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;

  texDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS |
                  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

  D3D12_HEAP_PROPERTIES defaultHeap = {D3D12_HEAP_TYPE_DEFAULT};

  ThrowIfFailed(ctx.device->CreateCommittedResource(
                    &defaultHeap, D3D12_HEAP_FLAG_SHARED, &texDesc,
                    D3D12_RESOURCE_STATE_COMMON, nullptr,
                    IID_PPV_ARGS(&imgRes.resource)),
                "Failed to create Shared UAV 3D Texture");

  auto allocInfo = ctx.device->GetResourceAllocationInfo(0, 1, &texDesc);
  imgRes.allocationSize = allocInfo.SizeInBytes;

  ThrowIfFailed(ctx.device->CreateSharedHandle(imgRes.resource.Get(), nullptr,
                                               GENERIC_ALL, nullptr,
                                               &imgRes.sharedHandle),
                "Failed to export NT Handle");

  return imgRes;
}

template <typename T>
bool uploadTestData(D3D12Context &ctx, D3D12ImageResources &imgRes,
                    uint32_t width, uint32_t height, int channels) {
  // Generate host data
  size_t totalPixels = width * height;
  std::vector<T> hostData(totalPixels * channels);
  for (size_t i = 0; i < hostData.size(); ++i) {
    size_t pixelIdx = i / channels;
    int channelIdx = i % channels;
    hostData[i] = generateTestValue<T>(pixelIdx, channelIdx, totalPixels);
  }

  // Get footprint to understand pitch/alignment requirements
  D3D12_RESOURCE_DESC desc = imgRes.resource->GetDesc();
  D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint;
  UINT numRows;
  UINT64 rowSizeInBytes, totalBytes;
  ctx.device->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, &numRows,
                                    &rowSizeInBytes, &totalBytes);

  // Create Upload Buffer
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
                    D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                    IID_PPV_ARGS(&uploadBuffer)),
                "Failed to create Upload Buffer");

  // Map and Copy Data (accounting for pitch)
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

  // Record copy commands
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

  // Execute and block until upload is finished
  executeAndWait(ctx);

  return true;
}

template <typename T>
bool verifyD3D12Data(D3D12Context &ctx, D3D12ImageResources &imgRes,
                     uint32_t width, uint32_t height, int channels) {
  // Get footprint to understand pitch
  D3D12_RESOURCE_DESC desc = imgRes.resource->GetDesc();
  D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint;
  UINT numRows;
  UINT64 rowSizeInBytes, totalBytes;
  ctx.device->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, &numRows,
                                    &rowSizeInBytes, &totalBytes);

  // Create Readback Buffer
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
  bufDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

  ThrowIfFailed(ctx.device->CreateCommittedResource(
                    &readbackHeap, D3D12_HEAP_FLAG_NONE, &bufDesc,
                    D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                    IID_PPV_ARGS(&readbackBuffer)),
                "Failed to create Readback Buffer");

  // Transition texture: COMMON -> COPY_SOURCE
  D3D12_RESOURCE_BARRIER barrier = {};
  barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  barrier.Transition.pResource = imgRes.resource.Get();
  barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
  barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
  barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

  ThrowIfFailed(ctx.cmdAlloc->Reset());
  ThrowIfFailed(ctx.cmdList->Reset(ctx.cmdAlloc.Get(), nullptr));
  ctx.cmdList->ResourceBarrier(1, &barrier);

  // Copy Texture to Readback Buffer
  D3D12_TEXTURE_COPY_LOCATION dst = {};
  dst.pResource = readbackBuffer.Get();
  dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
  dst.PlacedFootprint = footprint;

  D3D12_TEXTURE_COPY_LOCATION src = {};
  src.pResource = imgRes.resource.Get();
  src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  src.SubresourceIndex = 0;

  ctx.cmdList->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);

  // Transition texture: COPY_SOURCE -> COMMON (Required before SYCL import)
  barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
  barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
  ctx.cmdList->ResourceBarrier(1, &barrier);

  ThrowIfFailed(ctx.cmdList->Close());
  executeAndWait(ctx);

  // Map and verify the Readback Buffer
  uint8_t *pData;
  ThrowIfFailed(
      readbackBuffer->Map(0, nullptr, reinterpret_cast<void **>(&pData)));

  bool passed = true;
  int errorCount = 0;
  size_t totalPixels = width * height;

  for (UINT y = 0; y < height; ++y) {
    T *row = reinterpret_cast<T *>(pData + y * footprint.Footprint.RowPitch);
    for (UINT x = 0; x < width; ++x) {
      for (int c = 0; c < channels; ++c) {
        size_t pixelIdx = y * width + x;
        T expected = generateTestValue<T>(pixelIdx, c, totalPixels);
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

  if (passed) {
    std::cout << "[D3D12 Only] Readback Verified Successfully." << std::endl;
  } else {
    std::cerr << "[D3D12 Only] Readback FAILED with " << errorCount
              << " errors." << std::endl;
  }

  return passed;
}

// ---------------------------------------------------------
// EXPORTABLE FENCES
// ---------------------------------------------------------
struct D3D12ExportableFence {
  ComPtr<ID3D12Fence> fence;
  HANDLE sharedHandle = nullptr;
  uint64_t fenceValue = 0;
};

inline D3D12ExportableFence createExportableFence(D3D12Context &ctx) {
  D3D12ExportableFence extFence;
  // KEY: Must be created with the SHARED flag
  ThrowIfFailed(ctx.device->CreateFence(0, D3D12_FENCE_FLAG_SHARED,
                                        IID_PPV_ARGS(&extFence.fence)),
                "Failed to create Shared Fence");

  ThrowIfFailed(ctx.device->CreateSharedHandle(extFence.fence.Get(), nullptr,
                                               GENERIC_ALL, nullptr,
                                               &extFence.sharedHandle),
                "Failed to export Fence NT Handle");

  return extFence;
}

inline void signalExportableFence(D3D12Context &ctx,
                                  D3D12ExportableFence &extFence) {
  extFence.fenceValue++;
  ThrowIfFailed(ctx.cmdQueue->Signal(extFence.fence.Get(), extFence.fenceValue),
                "Failed to signal shared fence");
}

inline void cleanupExportableFence(D3D12ExportableFence &extFence) {
  if (extFence.sharedHandle)
    CloseHandle(extFence.sharedHandle);
}

// ---------------------------------------------------------
// BUFFER STUFFER
// ---------------------------------------------------------
struct D3D12BufferResources {
  ComPtr<ID3D12Resource> resource;
  HANDLE sharedHandle = nullptr;
  size_t allocationSize = 0;
};

inline D3D12BufferResources createExportableBuffer(D3D12Context &ctx,
                                                   size_t sizeBytes) {
  D3D12BufferResources bufRes;
  D3D12_RESOURCE_DESC bufDesc = {};
  bufDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  bufDesc.Width = sizeBytes;
  bufDesc.Height = 1;
  bufDesc.DepthOrArraySize = 1;
  bufDesc.MipLevels = 1;
  bufDesc.Format = DXGI_FORMAT_UNKNOWN;
  bufDesc.SampleDesc.Count = 1;
  bufDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  bufDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

  D3D12_HEAP_PROPERTIES defaultHeap = {D3D12_HEAP_TYPE_DEFAULT};

  ThrowIfFailed(ctx.device->CreateCommittedResource(
                    &defaultHeap, D3D12_HEAP_FLAG_SHARED, &bufDesc,
                    D3D12_RESOURCE_STATE_COMMON, nullptr,
                    IID_PPV_ARGS(&bufRes.resource)),
                "Failed to create Shared Buffer");

  bufRes.allocationSize = sizeBytes;

  ThrowIfFailed(ctx.device->CreateSharedHandle(bufRes.resource.Get(), nullptr,
                                               GENERIC_ALL, nullptr,
                                               &bufRes.sharedHandle),
                "Failed to export Buffer NT Handle");

  return bufRes;
}

inline D3D12BufferResources createUploadBuffer(D3D12Context &ctx,
                                               size_t sizeBytes) {
  D3D12BufferResources bufRes;
  D3D12_RESOURCE_DESC bufDesc = {};
  bufDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  bufDesc.Width = sizeBytes;
  bufDesc.Height = 1;
  bufDesc.DepthOrArraySize = 1;
  bufDesc.MipLevels = 1;
  bufDesc.Format = DXGI_FORMAT_UNKNOWN;
  bufDesc.SampleDesc.Count = 1;
  bufDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  bufDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

  D3D12_HEAP_PROPERTIES uploadHeap = {D3D12_HEAP_TYPE_UPLOAD};

  ThrowIfFailed(ctx.device->CreateCommittedResource(
                    &uploadHeap, D3D12_HEAP_FLAG_NONE, &bufDesc,
                    D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                    IID_PPV_ARGS(&bufRes.resource)),
                "Failed to create Upload Buffer");

  bufRes.allocationSize = sizeBytes;
  return bufRes;
}

inline D3D12BufferResources createReadbackBuffer(D3D12Context &ctx,
                                                 size_t sizeBytes) {
  D3D12BufferResources bufRes;
  D3D12_RESOURCE_DESC bufDesc = {};
  bufDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  bufDesc.Width = sizeBytes;
  bufDesc.Height = 1;
  bufDesc.DepthOrArraySize = 1;
  bufDesc.MipLevels = 1;
  bufDesc.Format = DXGI_FORMAT_UNKNOWN;
  bufDesc.SampleDesc.Count = 1;
  bufDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  bufDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

  D3D12_HEAP_PROPERTIES readbackHeap = {D3D12_HEAP_TYPE_READBACK};

  ThrowIfFailed(ctx.device->CreateCommittedResource(
                    &readbackHeap, D3D12_HEAP_FLAG_NONE, &bufDesc,
                    D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                    IID_PPV_ARGS(&bufRes.resource)),
                "Failed to create Readback Buffer");

  bufRes.allocationSize = sizeBytes;
  return bufRes;
}

inline void cleanupBuffer(D3D12BufferResources &bufRes) {
  if (bufRes.sharedHandle)
    CloseHandle(bufRes.sharedHandle);
}