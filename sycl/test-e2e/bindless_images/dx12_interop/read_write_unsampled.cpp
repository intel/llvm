// REQUIRES: aspect-ext_oneapi_external_memory_import
// REQUIRES: windows

// UNSUPPORTED: gpu-intel-gen12
// UNSUPPORTED-INTENDED: Unknown issue with integrated GPU failing
//                       when importing memory

//-DTEST_SEMAPHORE_IMPORT -DTEST_SMALL_IMAGE_SIZE -DVERBOSE_PRINT
// SYCL_DX12_ONLY_2D_UINT32_1=1 EXTRA_SMALL_SIZE=1 SYCL_DX12_IMPORT_PROBE_ONLY=1

// RUN: %{build} %link-directx  -DSYCL_BINDLESS_IMAGES_DEBUG_FETCH -o %t.out
// %{run-unfiltered-devices} env SYCL_BINDLESS_IMAGES_DEBUG=1
// SYCL_BINDLESS_IMAGES_DEBUG_L0_COPY=1 NEOReadDebugKeys=1 UseBindlessMode=1
// UseExternalAllocatorForSshAndDsh=1 %t.out

#pragma clang diagnostic ignored "-Waddress-of-temporary"

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include <sycl/stream.hpp>

#include "read_write_unsampled.h"

static bool envFlagIsSet(const char *Name) {
  char Value[8] = {};
  DWORD ValueSize = GetEnvironmentVariableA(Name, Value, sizeof(Value));
  return ValueSize != 0 && !(Value[0] == '0' && Value[1] == '\0');
}

static bool runOnly2DUint32Case() {
  return envFlagIsSet("SYCL_DX12_ONLY_2D_UINT32_1");
}

static bool importProbeOnly() {
  return envFlagIsSet("SYCL_DX12_IMPORT_PROBE_ONLY");
}

#ifdef VERBOSE_PRINT
static bool traceAllValues() {
  return envFlagIsSet("SYCL_DX12_TRACE_ALL_VALUES");
}

static size_t traceValueLimit(size_t Count) {
  return traceAllValues() ? Count : std::min<size_t>(Count, 32);
}

template <typename T> static auto printableValue(T Value) {
  if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>)
    return static_cast<int>(Value);
  else
    return Value;
}

template <typename T>
static void traceLinearValues(const char *Stage, const T *Data, size_t Count) {
  const size_t Limit = traceValueLimit(Count);
  std::cout << "[dx12-trace] " << Stage << " count=" << Count
            << " printed=" << Limit;
  if (!traceAllValues() && Limit < Count)
    std::cout << " (set SYCL_DX12_TRACE_ALL_VALUES=1 for full dump)";
  std::cout << "\n";
  for (size_t I = 0; I < Limit; ++I)
    std::cout << "[dx12-trace]   [" << I << "]=" << printableValue(Data[I])
              << "\n";
}

template <typename T>
static void tracePitchedComparison(const char *Stage, const T *Input,
                                   const T *Output, int Width, int Height,
                                   int Depth, int NChannels, UINT RowPitch,
                                   UINT NumRows) {
  const size_t Count = static_cast<size_t>(Width) * Height * Depth * NChannels;
  const size_t Limit = traceValueLimit(Count);
  const UINT SlicePitch = RowPitch * NumRows;
  const size_t ValuesPerRow = static_cast<size_t>(Width) * NChannels;
  const uint8_t *OutputBytes = reinterpret_cast<const uint8_t *>(Output);

  std::cout << "[dx12-trace] " << Stage << " count=" << Count
            << " printed=" << Limit << " rowPitch=" << RowPitch
            << " numRows=" << NumRows << " slicePitch=" << SlicePitch;
  if (!traceAllValues() && Limit < Count)
    std::cout << " (set SYCL_DX12_TRACE_ALL_VALUES=1 for full dump)";
  std::cout << "\n";

  for (size_t I = 0; I < Limit; ++I) {
    const size_t Plane = static_cast<size_t>(Width) * Height * NChannels;
    const size_t Z = I / Plane;
    const size_t InPlane = I % Plane;
    const size_t Y = InPlane / ValuesPerRow;
    const size_t XChannel = InPlane % ValuesPerRow;
    const T *OutputRow = reinterpret_cast<const T *>(
        OutputBytes + Z * SlicePitch + Y * RowPitch);
    std::cout << "[dx12-trace]   linear=" << I << " z=" << Z << " y=" << Y
              << " xChannel=" << XChannel
              << " input=" << printableValue(Input[I])
              << " output=" << printableValue(OutputRow[XChannel]) << "\n";
  }
}

template <typename T>
static void tracePitchedValues(const char *Stage, const T *Data, int Width,
                               int Height, int Depth, int NChannels,
                               UINT RowPitch, UINT NumRows) {
  const size_t Count = static_cast<size_t>(Width) * Height * Depth * NChannels;
  const size_t Limit = traceValueLimit(Count);
  const UINT SlicePitch = RowPitch * NumRows;
  const size_t ValuesPerRow = static_cast<size_t>(Width) * NChannels;
  const uint8_t *Bytes = reinterpret_cast<const uint8_t *>(Data);

  std::cout << "[dx12-trace] " << Stage << " count=" << Count
            << " printed=" << Limit << " rowPitch=" << RowPitch
            << " numRows=" << NumRows << " slicePitch=" << SlicePitch;
  if (!traceAllValues() && Limit < Count)
    std::cout << " (set SYCL_DX12_TRACE_ALL_VALUES=1 for full dump)";
  std::cout << "\n";

  for (size_t I = 0; I < Limit; ++I) {
    const size_t Plane = static_cast<size_t>(Width) * Height * NChannels;
    const size_t Z = I / Plane;
    const size_t InPlane = I % Plane;
    const size_t Y = InPlane / ValuesPerRow;
    const size_t XChannel = InPlane % ValuesPerRow;
    const T *Row =
        reinterpret_cast<const T *>(Bytes + Z * SlicePitch + Y * RowPitch);
    std::cout << "[dx12-trace]   linear=" << I << " z=" << Z << " y=" << Y
              << " xChannel=" << XChannel
              << " value=" << printableValue(Row[XChannel]) << "\n";
  }
}

static void traceFootprint(const char *Stage,
                           const D3D12_PLACED_SUBRESOURCE_FOOTPRINT &Footprint,
                           UINT NumRows, UINT64 RowSizeBytes,
                           UINT64 TotalBytes) {
  std::cout << "[dx12-trace] " << Stage
            << " footprint: offset=" << Footprint.Offset
            << " width=" << Footprint.Footprint.Width
            << " height=" << Footprint.Footprint.Height
            << " depth=" << Footprint.Footprint.Depth
            << " rowPitch=" << Footprint.Footprint.RowPitch
            << " format=" << Footprint.Footprint.Format
            << " numRows=" << NumRows << " rowSizeBytes=" << RowSizeBytes
            << " totalBytes=" << TotalBytes << "\n";
}

static size_t traceKernelStreamSize(size_t Count) {
  const size_t Limit = traceValueLimit(Count);
  return std::max<size_t>(8192, Limit * 512);
}
#endif

DX12SYCLDevice::DX12SYCLDevice()
    : m_syclQueue{{sycl::property::queue::in_order{}}},
      m_syclDevice{m_syclQueue.get_device()} {
  initDX12Device();
  initDX12CommandList();
}

void DX12SYCLDevice::initDX12Device() {
  // Create DXGI factory.
  ThrowIfFailed(CreateDXGIFactory2(0 /* dxgiFactoryFlags */,
                                   IID_PPV_ARGS(&m_dx12Factory)));

  // Get the hardware adapter for a suitable GPU.
  m_dx12HardwareAdapter = getDXGIHardwareAdapter<dx_version::DX12>(
      m_dx12Factory.Get(), m_syclDevice.get_info<sycl::info::device::name>());

  // Create a device from our hardware adapter.
  ThrowIfFailed(D3D12CreateDevice(m_dx12HardwareAdapter.Get(),
                                  D3D_FEATURE_LEVEL_12_0,
                                  IID_PPV_ARGS(&m_dx12Device)));
}

void DX12SYCLDevice::initDX12CommandList() {
  // Describe and create the command queue.
  D3D12_COMMAND_QUEUE_DESC queueDesc = {D3D12_COMMAND_LIST_TYPE_DIRECT, 0,
                                        D3D12_COMMAND_QUEUE_FLAG_NONE, 0};
  ThrowIfFailed(m_dx12Device->CreateCommandQueue(
      &queueDesc, IID_PPV_ARGS(&m_dx12CommandQueue)));

  // Create the command allocator.
  ThrowIfFailed(m_dx12Device->CreateCommandAllocator(
      D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_dx12CommandAllocator)));

  // Create the command list.
  ThrowIfFailed(m_dx12Device->CreateCommandList(
      0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_dx12CommandAllocator.Get(), NULL,
      IID_PPV_ARGS(&m_dx12CommandList)));
}

template <int NDims, typename DType, int NChannels>
DX12InteropTest<NDims, DType, NChannels>::DX12InteropTest(
    DX12SYCLDevice &device, sycl::image_channel_type channelType,
    sycl::range<NDims> globalSize, sycl::range<NDims> localSize)
    : m_device(device), m_channelType(channelType), m_globalSize(globalSize),
      m_localSize(localSize) {
  m_width = m_globalSize[0];
  m_height = 1;
  m_depth = 1;
  if constexpr (NDims > 1) {
    m_height = m_globalSize[1];
    if constexpr (NDims > 2)
      m_depth = m_globalSize[2];
  }
  m_numElems = m_width * m_height * m_depth * NChannels;
}

template <int NDims, typename DType, int NChannels>
void DX12InteropTest<NDims, DType, NChannels>::initDX12Resources() {

  // Define default heap properties.
  D3D12_HEAP_PROPERTIES defaultHeapProperties = {};
  defaultHeapProperties.Type = D3D12_HEAP_TYPE_DEFAULT;
  defaultHeapProperties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
  defaultHeapProperties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
  defaultHeapProperties.CreationNodeMask = 1;
  defaultHeapProperties.VisibleNodeMask = 1;

  // Define texture resource descriptor.
  D3D12_RESOURCE_DESC textureResourceDesc = {};
  if constexpr (NDims == 1)
    textureResourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE1D;
  else if constexpr (NDims == 2)
    textureResourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
  else
    textureResourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
  textureResourceDesc.Alignment = 0;
  textureResourceDesc.Width = m_width;
  textureResourceDesc.Height = m_height;
  textureResourceDesc.DepthOrArraySize = m_depth;
  textureResourceDesc.MipLevels = 1;
  textureResourceDesc.Format = toDXGIFormat(NChannels, m_channelType);
  textureResourceDesc.SampleDesc = DXGI_SAMPLE_DESC{1, 0};
  textureResourceDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
  textureResourceDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS |
                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

#ifdef VERBOSE_PRINT
  std::cout << "[dx12-trace] create texture: NDims=" << NDims
            << " width=" << m_width << " height=" << m_height
            << " depth=" << m_depth << " channels=" << NChannels
            << " channelType="
            << bindless_helpers::channelTypeToString(m_channelType)
            << " dxgiFormat=" << textureResourceDesc.Format
            << " flags=" << textureResourceDesc.Flags
            << " initialState=D3D12_RESOURCE_STATE_COMMON\n";
#endif

  // Create the DX12 texture.
  auto *dx12Device = m_device.getDx12Device();
  ThrowIfFailed(dx12Device->CreateCommittedResource(
      &defaultHeapProperties, D3D12_HEAP_FLAG_SHARED, &textureResourceDesc,
      D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&m_dx12Texture)));

#ifdef VERBOSE_PRINT
  std::cout << "[dx12-trace] created DX12 texture resource="
            << m_dx12Texture.Get() << "\n";
#endif

  // Create a shared handle for our texture.
  ThrowIfFailed(dx12Device->CreateSharedHandle(m_dx12Texture.Get(), nullptr,
                                               GENERIC_ALL, nullptr,
                                               &m_sharedMemoryHandle));

#ifdef VERBOSE_PRINT
  std::cout << "[dx12-trace] created shared memory handle="
            << m_sharedMemoryHandle << "\n";
#endif

  D3D12_RESOURCE_ALLOCATION_INFO textureAllocationInfo;
  textureAllocationInfo =
      dx12Device->GetResourceAllocationInfo(1, 1, &textureResourceDesc);
  size_t allocationSize = textureAllocationInfo.SizeInBytes;

#ifdef VERBOSE_PRINT
  std::cout << "[dx12-trace] texture allocation size=" << allocationSize
            << " alignment=" << textureAllocationInfo.Alignment << "\n";
#endif

  // Create the DX12 fence and map to a SYCL semaphore.
  ThrowIfFailed(dx12Device->CreateFence(
      m_sharedFenceValue, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&m_dx12Fence)));
  m_sharedFenceValue++;

#ifdef TEST_SEMAPHORE_IMPORT
  ThrowIfFailed(dx12Device->CreateSharedHandle(m_dx12Fence.Get(), nullptr,
                                               GENERIC_ALL, nullptr,
                                               &m_sharedSemaphoreHandle));

  // Import our shared DX12 fence resource to SYCL.
  importDX12SharedSemaphoreHandle();
#endif

  // Create an event handle to use for synchronization.
  m_dx12FenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
  if (m_dx12FenceEvent == nullptr) {
    ThrowIfFailed(HRESULT_FROM_WIN32(GetLastError()));
  }

  populateDX12Texture();

  // Match the DX11 interop test: initialize the shared texture before creating
  // the SYCL image view of that external resource.
  importDX12SharedMemoryHandle(allocationSize);
}

template <int NDims, typename DType, int NChannels>
void DX12InteropTest<NDims, DType, NChannels>::importDX12SharedMemoryHandle(
    size_t allocationSize) {
  syclexp::external_mem_descriptor<syclexp::resource_win32_handle> extMemDesc{
      m_sharedMemoryHandle, syclexp::external_mem_handle_type::win32_nt_handle,
      allocationSize};

  auto &syclQueue = m_device.getSyclQueue();

#ifdef VERBOSE_PRINT
  const D3D12_RESOURCE_DESC dx12TextureDesc = m_dx12Texture->GetDesc();
  std::cout << "[dx12-trace] import_external_memory: handle="
            << m_sharedMemoryHandle << " handleType=win32_nt_handle"
            << " allocationSize=" << allocationSize << "\n";
  std::cout << "[dx12-trace] import_external_memory DX12 desc: dimension="
            << dx12TextureDesc.Dimension << " width=" << dx12TextureDesc.Width
            << " height=" << dx12TextureDesc.Height
            << " depthOrArraySize=" << dx12TextureDesc.DepthOrArraySize
            << " mipLevels=" << dx12TextureDesc.MipLevels
            << " format=" << dx12TextureDesc.Format
            << " sampleCount=" << dx12TextureDesc.SampleDesc.Count
            << " sampleQuality=" << dx12TextureDesc.SampleDesc.Quality
            << " layout=" << dx12TextureDesc.Layout
            << " flags=" << dx12TextureDesc.Flags << "\n";
#endif

  m_syclExternalMemHandle =
      syclexp::import_external_memory(extMemDesc, syclQueue);

#ifdef VERBOSE_PRINT
  std::cerr << "[dx12-trace] import_external_memory returned to test\n";
  std::cerr.flush();
#endif

  syclexp::image_descriptor syclImageDesc{m_globalSize, NChannels,
                                          m_channelType};

#ifdef VERBOSE_PRINT
  std::cerr << "[dx12-trace] constructed SYCL image descriptor before map\n";
  std::cerr.flush();
#endif

#ifdef VERBOSE_PRINT
  std::cout << "[dx12-trace] map_external_image_memory: NDims=" << NDims
            << " width=" << m_width << " height=" << m_height
            << " depth=" << m_depth << " channels=" << NChannels
            << " channelType="
            << bindless_helpers::channelTypeToString(m_channelType) << "\n";
#endif

#ifdef TEST_SEMAPHORE_IMPORT
#ifdef VERBOSE_PRINT
  std::cout << "[dx12-trace] wait imported DX12 upload semaphore before "
            << "map_external_image_memory: value=" << m_sharedFenceValue
            << "\n";
#endif
  syclQueue.ext_oneapi_wait_external_semaphore(m_syclExternalSemaphoreHandle,
                                               m_sharedFenceValue);
  syclQueue.wait();
  m_sharedFenceValue++;
#ifdef VERBOSE_PRINT
  std::cout << "[dx12-trace] imported DX12 upload semaphore wait completed, "
            << "next fence value " << m_sharedFenceValue << "\n";

  {
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT snapshotFootprint = {};
    UINT snapshotNumRows = 0;
    UINT64 snapshotRowSizeBytes = 0;
    UINT64 snapshotBufferSize = 0;
    auto *dx12Device = m_device.getDx12Device();
    dx12Device->GetCopyableFootprints(
        &m_dx12Texture->GetDesc(), 0, 1, 0, &snapshotFootprint,
        &snapshotNumRows, &snapshotRowSizeBytes, &snapshotBufferSize);

    traceFootprint("DX12 post-semaphore pre-map snapshot", snapshotFootprint,
                   snapshotNumRows, snapshotRowSizeBytes, snapshotBufferSize);

    D3D12_RESOURCE_DESC snapshotBufferResourceDesc = {};
    snapshotBufferResourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    snapshotBufferResourceDesc.Alignment = 0;
    snapshotBufferResourceDesc.Width = snapshotBufferSize;
    snapshotBufferResourceDesc.Height = 1;
    snapshotBufferResourceDesc.DepthOrArraySize = 1;
    snapshotBufferResourceDesc.MipLevels = 1;
    snapshotBufferResourceDesc.Format = DXGI_FORMAT_UNKNOWN;
    snapshotBufferResourceDesc.SampleDesc = DXGI_SAMPLE_DESC{1, 0};
    snapshotBufferResourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    snapshotBufferResourceDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

    D3D12_HEAP_PROPERTIES snapshotHeapProperties = {};
    snapshotHeapProperties.Type = D3D12_HEAP_TYPE_READBACK;
    snapshotHeapProperties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    snapshotHeapProperties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    snapshotHeapProperties.CreationNodeMask = 1;
    snapshotHeapProperties.VisibleNodeMask = 1;

    ComPtr<ID3D12Resource> snapshotBuffer;
    ThrowIfFailed(dx12Device->CreateCommittedResource(
        &snapshotHeapProperties, D3D12_HEAP_FLAG_NONE,
        &snapshotBufferResourceDesc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
        IID_PPV_ARGS(&snapshotBuffer)));

    ThrowIfFailed(m_device.resetCommandList());

    D3D12_TEXTURE_COPY_LOCATION snapshotDest = {};
    snapshotDest.pResource = snapshotBuffer.Get();
    snapshotDest.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    snapshotDest.PlacedFootprint = snapshotFootprint;

    D3D12_TEXTURE_COPY_LOCATION snapshotSrc = {};
    snapshotSrc.pResource = m_dx12Texture.Get();
    snapshotSrc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    snapshotSrc.SubresourceIndex = 0;

    D3D12_RESOURCE_BARRIER snapshotBarrier = {};
    snapshotBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    snapshotBarrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    snapshotBarrier.Transition.pResource = m_dx12Texture.Get();
    snapshotBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
    snapshotBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    snapshotBarrier.Transition.Subresource =
        D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

    auto *snapshotCommandList = m_device.getDx12CommandList();
    snapshotCommandList->ResourceBarrier(1, &snapshotBarrier);
    std::cout << "[dx12-trace] CopyTextureRegion DX12 post-semaphore "
              << "pre-map snapshot: srcTexture=" << m_dx12Texture.Get()
              << " dstBuffer=" << snapshotBuffer.Get()
              << " rowPitch=" << snapshotFootprint.Footprint.RowPitch << "\n";
    snapshotCommandList->CopyTextureRegion(&snapshotDest, 0, 0, 0, &snapshotSrc,
                                           nullptr);
    snapshotBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
    snapshotBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
    snapshotCommandList->ResourceBarrier(1, &snapshotBarrier);

    ThrowIfFailed(snapshotCommandList->Close());
    ID3D12CommandList *snapshotCommandLists[] = {snapshotCommandList};
    auto *dx12CommandQueue = m_device.getDx12CommandQueue();
    dx12CommandQueue->ExecuteCommandLists(_countof(snapshotCommandLists),
                                          snapshotCommandLists);
    ThrowIfFailed(
        dx12CommandQueue->Signal(m_dx12Fence.Get(), m_sharedFenceValue));
    waitDX12Fence();
    m_sharedFenceValue++;

    D3D12_RANGE snapshotRange{0, snapshotBufferSize};
    DType *snapshotData{};
    ThrowIfFailed(snapshotBuffer->Map(
        0, &snapshotRange, reinterpret_cast<void **>(&snapshotData)));
    tracePitchedValues("DX12 texture snapshot after imported semaphore wait",
                       snapshotData, m_width, m_height, m_depth, NChannels,
                       snapshotFootprint.Footprint.RowPitch, snapshotNumRows);
    D3D12_RANGE snapshotEmptyRange{0, 0};
    snapshotBuffer->Unmap(0, &snapshotEmptyRange);
  }
#endif
#endif

  m_syclImageMemHandle = syclexp::map_external_image_memory(
      m_syclExternalMemHandle, syclImageDesc, syclQueue);

#ifdef VERBOSE_PRINT
  std::cout << "[dx12-trace] create_image from mapped external image memory\n";
#endif

  m_syclImageHandle =
      syclexp::create_image(m_syclImageMemHandle, syclImageDesc, syclQueue);
}

template <int NDims, typename DType, int NChannels>
void DX12InteropTest<NDims, DType,
                     NChannels>::importDX12SharedSemaphoreHandle() {
  syclexp::external_semaphore_descriptor<syclexp::resource_win32_handle>
      extSemDesc{m_sharedSemaphoreHandle,
                 syclexp::external_semaphore_handle_type::win32_nt_dx12_fence};

  m_syclExternalSemaphoreHandle =
      syclexp::import_external_semaphore(extSemDesc, m_device.getSyclQueue());

#ifdef VERBOSE_PRINT
  std::cout << "[dx12-trace] import_external_semaphore: handle="
            << m_sharedSemaphoreHandle << " handleType=win32_nt_dx12_fence\n";
#endif
}

template <int NDims, typename DType, int NChannels>
void DX12InteropTest<NDims, DType, NChannels>::callSYCLKernel() {
  auto &syclQueue = m_device.getSyclQueue();

  if (importProbeOnly()) {
#ifdef VERBOSE_PRINT
    std::cout << "[dx12-trace] import probe only mode: skipping SYCL kernel "
              << "submission and leaving DX12 contents unchanged\n";
#endif
    return;
  }

#ifdef VERBOSE_PRINT
  const size_t TraceKernelValueLimit = traceValueLimit(m_numElems);
  const size_t TraceKernelStreamSize = traceKernelStreamSize(m_numElems);
#endif

#ifdef VERBOSE_PRINT
  std::cout << "[dx12-trace] submit SYCL kernel: NDims=" << NDims << " global=";
  for (int I = 0; I < NDims; ++I)
    std::cout << (I ? "x" : "") << m_globalSize[I];
  std::cout << " local=";
  for (int I = 0; I < NDims; ++I)
    std::cout << (I ? "x" : "") << m_localSize[I];
  std::cout << " operation=fetch_image * 2 -> write_image\n";
#endif

#ifdef TEST_SEMAPHORE_IMPORT
  // The DX12 upload semaphore was waited before creating the SYCL image view,
  // so the map/create debug copies and the kernel see the same synchronized
  // external resource state.
#endif

  // We can't capture the image handle through `this` in the lambda.
  // If we do the kernel will crash.
  auto imgHandle = m_syclImageHandle;
  const size_t KernelWidth = m_width;
  const size_t KernelHeight = m_height;

  using VecType = sycl::vec<DType, NChannels>;

  // Submit our SYCL kernel. All we do is double the value of each pixel in the
  // texture.
  try {
    syclQueue.submit([&](sycl::handler &cgh) {
#ifdef VERBOSE_PRINT
      sycl::stream KernelTrace(TraceKernelStreamSize, 512, cgh);
#endif
      cgh.parallel_for(
          sycl::nd_range<NDims>{m_globalSize, m_localSize},
          [=](sycl::nd_item<NDims> it) {
            if constexpr (NDims == 3) {
              size_t dim0 = it.get_global_id(0);
              size_t dim1 = it.get_global_id(1);
              size_t dim2 = it.get_global_id(2);
              size_t linear = (dim2 * KernelWidth * KernelHeight +
                               dim1 * KernelWidth + dim0) *
                              NChannels;
              auto px = syclexp::fetch_image<
                  std::conditional_t<NChannels == 1, DType, VecType>>(
                  imgHandle, sycl::int3(dim0, dim1, dim2));
#ifdef VERBOSE_PRINT
              if (linear < TraceKernelValueLimit) {
                if constexpr (NChannels == 1) {
                  KernelTrace << "[dx12-trace] sycl fetch linear=" << linear
                              << " z=" << dim2 << " y=" << dim1 << " x=" << dim0
                              << " c=0 value=" << static_cast<float>(px)
                              << "\n";
                } else {
                  for (int channel = 0; channel < NChannels; ++channel)
                    KernelTrace
                        << "[dx12-trace] sycl fetch linear="
                        << (linear + channel) << " z=" << dim2 << " y=" << dim1
                        << " x=" << dim0 << " c=" << channel
                        << " value=" << static_cast<float>(px[channel]) << "\n";
                }
              }
#endif
              px *= static_cast<DType>(2);
#ifdef VERBOSE_PRINT
              if (linear < TraceKernelValueLimit) {
                if constexpr (NChannels == 1) {
                  KernelTrace << "[dx12-trace] sycl write linear=" << linear
                              << " z=" << dim2 << " y=" << dim1 << " x=" << dim0
                              << " c=0 value=" << static_cast<float>(px)
                              << "\n";
                } else {
                  for (int channel = 0; channel < NChannels; ++channel)
                    KernelTrace
                        << "[dx12-trace] sycl write linear="
                        << (linear + channel) << " z=" << dim2 << " y=" << dim1
                        << " x=" << dim0 << " c=" << channel
                        << " value=" << static_cast<float>(px[channel]) << "\n";
                }
              }
#endif
              syclexp::write_image(imgHandle, sycl::int3(dim0, dim1, dim2), px);
            } else if constexpr (NDims == 2) {
              size_t dim0 = it.get_global_id(0);
              size_t dim1 = it.get_global_id(1);
              size_t linear = (dim1 * KernelWidth + dim0) * NChannels;
              auto px = syclexp::fetch_image<
                  std::conditional_t<NChannels == 1, DType, VecType>>(
                  imgHandle, sycl::int2(dim0, dim1));
#ifdef VERBOSE_PRINT
              if (linear < TraceKernelValueLimit) {
                if constexpr (NChannels == 1) {
                  KernelTrace << "[dx12-trace] sycl fetch linear=" << linear
                              << " y=" << dim1 << " x=" << dim0
                              << " c=0 value=" << static_cast<float>(px)
                              << "\n";
                } else {
                  for (int channel = 0; channel < NChannels; ++channel)
                    KernelTrace << "[dx12-trace] sycl fetch linear="
                                << (linear + channel) << " y=" << dim1
                                << " x=" << dim0 << " c=" << channel
                                << " value=" << static_cast<float>(px[channel])
                                << "\n";
                }
              }
#endif
              px *= static_cast<DType>(2);
#ifdef VERBOSE_PRINT
              if (linear < TraceKernelValueLimit) {
                if constexpr (NChannels == 1) {
                  KernelTrace << "[dx12-trace] sycl write linear=" << linear
                              << " y=" << dim1 << " x=" << dim0
                              << " c=0 value=" << static_cast<float>(px)
                              << "\n";
                } else {
                  for (int channel = 0; channel < NChannels; ++channel)
                    KernelTrace << "[dx12-trace] sycl write linear="
                                << (linear + channel) << " y=" << dim1
                                << " x=" << dim0 << " c=" << channel
                                << " value=" << static_cast<float>(px[channel])
                                << "\n";
                }
              }
#endif
              syclexp::write_image(imgHandle, sycl::int2(dim0, dim1), px);
            } else {
              size_t dim0 = it.get_global_id(0);
              size_t linear = dim0 * NChannels;
              auto px = syclexp::fetch_image<
                  std::conditional_t<NChannels == 1, DType, VecType>>(
                  imgHandle, int(dim0));
#ifdef VERBOSE_PRINT
              if (linear < TraceKernelValueLimit) {
                if constexpr (NChannels == 1) {
                  KernelTrace << "[dx12-trace] sycl fetch linear=" << linear
                              << " x=" << dim0
                              << " c=0 value=" << static_cast<float>(px)
                              << "\n";
                } else {
                  for (int channel = 0; channel < NChannels; ++channel)
                    KernelTrace << "[dx12-trace] sycl fetch linear="
                                << (linear + channel) << " x=" << dim0
                                << " c=" << channel
                                << " value=" << static_cast<float>(px[channel])
                                << "\n";
                }
              }
#endif
              px *= static_cast<DType>(2);
#ifdef VERBOSE_PRINT
              if (linear < TraceKernelValueLimit) {
                if constexpr (NChannels == 1) {
                  KernelTrace << "[dx12-trace] sycl write linear=" << linear
                              << " x=" << dim0
                              << " c=0 value=" << static_cast<float>(px)
                              << "\n";
                } else {
                  for (int channel = 0; channel < NChannels; ++channel)
                    KernelTrace << "[dx12-trace] sycl write linear="
                                << (linear + channel) << " x=" << dim0
                                << " c=" << channel
                                << " value=" << static_cast<float>(px[channel])
                                << "\n";
                }
              }
#endif
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

#ifdef TEST_SEMAPHORE_IMPORT
  // Increment the fence value.
  m_sharedFenceValue++;

  // Signal imported semaphore.
  syclQueue.submit([&](sycl::handler &cgh) {
    cgh.ext_oneapi_signal_external_semaphore(m_syclExternalSemaphoreHandle,
                                             m_sharedFenceValue);
  });

  // Use DX12 to wait for the semaphore signalled by SYCL above.
  waitDX12Fence();
  m_sharedFenceValue++;
#else
  syclQueue.wait();

#ifdef VERBOSE_PRINT
  std::cout << "[dx12-trace] SYCL kernel completed via syclQueue.wait()\n";
#endif

  // Additional fence signal/wait after SYCL work completes to ensure
  // Level Zero writes are visible to DX12. This flushes caches.
  auto *dx12CommandQueue = m_device.getDx12CommandQueue();
  ThrowIfFailed(
      dx12CommandQueue->Signal(m_dx12Fence.Get(), m_sharedFenceValue));
  waitDX12Fence();
  m_sharedFenceValue++;
#endif
}

template <int NDims, typename DType, int NChannels>
void DX12InteropTest<NDims, DType, NChannels>::populateDX12Texture() {

  // Set our texture data to upload.
  m_inputData.resize(m_numElems);
  auto getInputValue = [&](int i) -> DType {
    if constexpr (std::is_integral_v<DType> ||
                  std::is_same_v<DType, sycl::half>)
      i = i % (static_cast<uint64_t>(std::numeric_limits<DType>::max()) / 2);
    return i;
  };
  for (int i = 0; i < m_numElems; ++i) {
    m_inputData[i] = getInputValue(i);
  }

#ifdef VERBOSE_PRINT
  traceLinearValues("generated input values", m_inputData.data(), m_numElems);
#endif

  // Get the texture footprint (actual row pitch and total buffer size).
  D3D12_PLACED_SUBRESOURCE_FOOTPRINT uploadFootprint = {};
  UINT uploadNumRows = 0;
  UINT64 uploadRowSizeBytes = 0;
  UINT64 stagingBufferSize = 0;
  auto *dx12Device = m_device.getDx12Device();
  dx12Device->GetCopyableFootprints(&m_dx12Texture->GetDesc(), 0, 1, 0,
                                    &uploadFootprint, &uploadNumRows,
                                    &uploadRowSizeBytes, &stagingBufferSize);

#ifdef VERBOSE_PRINT
  traceFootprint("upload", uploadFootprint, uploadNumRows, uploadRowSizeBytes,
                 stagingBufferSize);
#endif

  // Define upload heap properties.
  D3D12_HEAP_PROPERTIES uploadHeapProperties = {};
  uploadHeapProperties.Type = D3D12_HEAP_TYPE_UPLOAD;
  uploadHeapProperties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
  uploadHeapProperties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
  uploadHeapProperties.CreationNodeMask = 1;
  uploadHeapProperties.VisibleNodeMask = 1;

  // Define upload buffer resource descriptor.
  D3D12_RESOURCE_DESC uploadBufferResourceDesc = {};
  uploadBufferResourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  uploadBufferResourceDesc.Alignment = 0;
  uploadBufferResourceDesc.Width = stagingBufferSize;
  uploadBufferResourceDesc.Height = 1;
  uploadBufferResourceDesc.DepthOrArraySize = 1;
  uploadBufferResourceDesc.MipLevels = 1;
  uploadBufferResourceDesc.Format = DXGI_FORMAT_UNKNOWN;
  uploadBufferResourceDesc.SampleDesc = DXGI_SAMPLE_DESC{1, 0};
  uploadBufferResourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  uploadBufferResourceDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

  // Allocate the staging upload buffer.
  ComPtr<ID3D12Resource> stagingBuffer;
  ThrowIfFailed(dx12Device->CreateCommittedResource(
      &uploadHeapProperties, D3D12_HEAP_FLAG_NONE, &uploadBufferResourceDesc,
      D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
      IID_PPV_ARGS(&stagingBuffer)));

  // Map the upload staging buffer to host visible memory.
  D3D12_RANGE stagingBufferRange{0, stagingBufferSize};
  DType *pStagingBufferData{};
  ThrowIfFailed(stagingBuffer->Map(
      0, &stagingBufferRange, reinterpret_cast<void **>(&pStagingBufferData)));

  // Populate the staging buffer row-by-row using the actual row pitch from
  // GetCopyableFootprints. A manually-computed pitch may differ from the
  // driver's required pitch (e.g. for UAV/SIMULTANEOUS_ACCESS textures).
  const UINT uploadRowPitch = uploadFootprint.Footprint.RowPitch;
  const UINT uploadSlicePitch = uploadRowPitch * uploadNumRows;
  const UINT uploadRowDataBytes =
      static_cast<UINT>(m_width) * sizeof(DType) * NChannels;
  for (int z = 0; z < m_depth; ++z) {
    for (int y = 0; y < m_height; ++y) {
      memcpy(reinterpret_cast<uint8_t *>(pStagingBufferData) +
                 z * uploadSlicePitch + y * uploadRowPitch,
             m_inputData.data() +
                 (z * m_width * m_height + y * m_width) * NChannels,
             uploadRowDataBytes);
    }
  }

#ifdef VERBOSE_PRINT
  tracePitchedValues("staging upload buffer after host memcpy",
                     pStagingBufferData, m_width, m_height, m_depth, NChannels,
                     uploadRowPitch, uploadNumRows);
#endif

  // Unmap the staging buffer.
  D3D12_RANGE emptyRange{0, 0};
  stagingBuffer->Unmap(0, &emptyRange);

  // Reset command list to inital state if necessary.
  std::ignore = m_device.resetCommandList();

  // Set the copy source and destination footprint/locations using the
  // actual footprint returned by GetCopyableFootprints.
  D3D12_TEXTURE_COPY_LOCATION copyDest = {};
  copyDest.pResource = m_dx12Texture.Get();
  copyDest.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  copyDest.SubresourceIndex = 0;

  D3D12_TEXTURE_COPY_LOCATION copySrc = {};
  copySrc.pResource = stagingBuffer.Get();
  copySrc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
  copySrc.PlacedFootprint = uploadFootprint;

#ifdef VERBOSE_PRINT
  std::cout << "[dx12-trace] CopyTextureRegion upload: srcBuffer="
            << stagingBuffer.Get() << " dstTexture=" << m_dx12Texture.Get()
            << " rowPitch=" << uploadRowPitch
            << " rowDataBytes=" << uploadRowDataBytes << "\n";
#endif

  auto *dx12CommandList = m_device.getDx12CommandList();
  D3D12_RESOURCE_BARRIER uploadBarrier = {};
  uploadBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  uploadBarrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
  uploadBarrier.Transition.pResource = m_dx12Texture.Get();
  uploadBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
  uploadBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
  uploadBarrier.Transition.Subresource =
      D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
  dx12CommandList->ResourceBarrier(1, &uploadBarrier);

  dx12CommandList->CopyTextureRegion(&copyDest, 0, 0, 0, &copySrc, nullptr);

  uploadBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
  uploadBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
  dx12CommandList->ResourceBarrier(1, &uploadBarrier);

  // Add UAV barrier to ensure the texture is visible to other engines
  // (Level Zero) when ALLOW_UNORDERED_ACCESS is set
  D3D12_RESOURCE_BARRIER uavBarrier = {};
  uavBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
  uavBarrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
  uavBarrier.UAV.pResource = m_dx12Texture.Get();
  dx12CommandList->ResourceBarrier(1, &uavBarrier);

  // Execute the command list.
  ThrowIfFailed(dx12CommandList->Close());
  ID3D12CommandList *ppCommandLists[] = {dx12CommandList};
  auto *dx12CommandQueue = m_device.getDx12CommandQueue();
  dx12CommandQueue->ExecuteCommandLists(_countof(ppCommandLists),
                                        ppCommandLists);
  ThrowIfFailed(
      dx12CommandQueue->Signal(m_dx12Fence.Get(), m_sharedFenceValue));

#ifdef VERBOSE_PRINT
  std::cout << "[dx12-trace] upload command list signalled fence value "
            << m_sharedFenceValue << "\n";
#endif

#ifdef TEST_SEMAPHORE_IMPORT
  waitDX12Fence();
  m_sharedFenceValue++;

#ifdef VERBOSE_PRINT
  std::cout << "[dx12-trace] DX12 upload completed before imported "
            << "semaphore handoff, next fence value " << m_sharedFenceValue
            << "\n";
#endif

  ThrowIfFailed(
      dx12CommandQueue->Signal(m_dx12Fence.Get(), m_sharedFenceValue));

#ifdef VERBOSE_PRINT
  std::cout << "[dx12-trace] upload handoff signalled fence value "
            << m_sharedFenceValue << " for imported semaphore wait\n";
#endif
#else
  waitDX12Fence();
  m_sharedFenceValue++;

#ifdef VERBOSE_PRINT
  std::cout << "[dx12-trace] DX12 upload completed, next fence value "
            << m_sharedFenceValue << "\n";

  {
    D3D12_RESOURCE_DESC snapshotBufferResourceDesc = {};
    snapshotBufferResourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    snapshotBufferResourceDesc.Alignment = 0;
    snapshotBufferResourceDesc.Width = stagingBufferSize;
    snapshotBufferResourceDesc.Height = 1;
    snapshotBufferResourceDesc.DepthOrArraySize = 1;
    snapshotBufferResourceDesc.MipLevels = 1;
    snapshotBufferResourceDesc.Format = DXGI_FORMAT_UNKNOWN;
    snapshotBufferResourceDesc.SampleDesc = DXGI_SAMPLE_DESC{1, 0};
    snapshotBufferResourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    snapshotBufferResourceDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

    D3D12_HEAP_PROPERTIES snapshotHeapProperties = {};
    snapshotHeapProperties.Type = D3D12_HEAP_TYPE_READBACK;
    snapshotHeapProperties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    snapshotHeapProperties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    snapshotHeapProperties.CreationNodeMask = 1;
    snapshotHeapProperties.VisibleNodeMask = 1;

    ComPtr<ID3D12Resource> snapshotBuffer;
    ThrowIfFailed(dx12Device->CreateCommittedResource(
        &snapshotHeapProperties, D3D12_HEAP_FLAG_NONE,
        &snapshotBufferResourceDesc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
        IID_PPV_ARGS(&snapshotBuffer)));

    ThrowIfFailed(m_device.resetCommandList());

    D3D12_TEXTURE_COPY_LOCATION snapshotDest = {};
    snapshotDest.pResource = snapshotBuffer.Get();
    snapshotDest.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    snapshotDest.PlacedFootprint = uploadFootprint;

    D3D12_TEXTURE_COPY_LOCATION snapshotSrc = {};
    snapshotSrc.pResource = m_dx12Texture.Get();
    snapshotSrc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    snapshotSrc.SubresourceIndex = 0;

    D3D12_RESOURCE_BARRIER snapshotBarrier = {};
    snapshotBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    snapshotBarrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    snapshotBarrier.Transition.pResource = m_dx12Texture.Get();
    snapshotBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
    snapshotBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    snapshotBarrier.Transition.Subresource =
        D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

    auto *snapshotCommandList = m_device.getDx12CommandList();
    snapshotCommandList->ResourceBarrier(1, &snapshotBarrier);
    std::cout << "[dx12-trace] CopyTextureRegion DX12 upload snapshot: "
              << "srcTexture=" << m_dx12Texture.Get()
              << " dstBuffer=" << snapshotBuffer.Get()
              << " rowPitch=" << uploadRowPitch << "\n";
    snapshotCommandList->CopyTextureRegion(&snapshotDest, 0, 0, 0, &snapshotSrc,
                                           nullptr);
    snapshotBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
    snapshotBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
    snapshotCommandList->ResourceBarrier(1, &snapshotBarrier);

    ThrowIfFailed(snapshotCommandList->Close());
    ID3D12CommandList *snapshotCommandLists[] = {snapshotCommandList};
    dx12CommandQueue->ExecuteCommandLists(_countof(snapshotCommandLists),
                                          snapshotCommandLists);
    ThrowIfFailed(
        dx12CommandQueue->Signal(m_dx12Fence.Get(), m_sharedFenceValue));
    waitDX12Fence();
    m_sharedFenceValue++;

    D3D12_RANGE snapshotRange{0, stagingBufferSize};
    DType *snapshotData{};
    ThrowIfFailed(snapshotBuffer->Map(
        0, &snapshotRange, reinterpret_cast<void **>(&snapshotData)));
    tracePitchedValues("DX12 texture snapshot after upload copy", snapshotData,
                       m_width, m_height, m_depth, NChannels, uploadRowPitch,
                       uploadNumRows);
    D3D12_RANGE snapshotEmptyRange{0, 0};
    snapshotBuffer->Unmap(0, &snapshotEmptyRange);
  }
#endif

  // Additional fence signal/wait to ensure cache coherency.
  // This ensures DX12 upload is fully visible before SYCL/Level Zero access.
  ThrowIfFailed(
      dx12CommandQueue->Signal(m_dx12Fence.Get(), m_sharedFenceValue));
  waitDX12Fence();
  m_sharedFenceValue++;
#endif
}

template <int NDims, typename DType, int NChannels>
bool DX12InteropTest<NDims, DType, NChannels>::validateOutput() {

  // Ensure all previous GPU work is complete before validation
  auto *dx12CommandQueue = m_device.getDx12CommandQueue();
  ThrowIfFailed(
      dx12CommandQueue->Signal(m_dx12Fence.Get(), m_sharedFenceValue));
  waitDX12Fence();
  m_sharedFenceValue++;

  // Reset the command list.
  ThrowIfFailed(m_device.resetCommandList());

  // Get the texture footprint (actual row pitch and total buffer size).
  D3D12_PLACED_SUBRESOURCE_FOOTPRINT readbackFootprint = {};
  UINT readbackNumRows = 0;
  UINT64 readbackRowSizeBytes = 0;
  UINT64 readbackBufferSize = 0;
  auto *dx12Device = m_device.getDx12Device();
  dx12Device->GetCopyableFootprints(&m_dx12Texture->GetDesc(), 0, 1, 0,
                                    &readbackFootprint, &readbackNumRows,
                                    &readbackRowSizeBytes, &readbackBufferSize);

#ifdef VERBOSE_PRINT
  traceFootprint("readback", readbackFootprint, readbackNumRows,
                 readbackRowSizeBytes, readbackBufferSize);
#endif

  // Define readback heap properties.
  D3D12_HEAP_PROPERTIES readbackHeapProperties = {};
  readbackHeapProperties.Type = D3D12_HEAP_TYPE_READBACK;
  readbackHeapProperties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
  readbackHeapProperties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
  readbackHeapProperties.CreationNodeMask = 1;
  readbackHeapProperties.VisibleNodeMask = 1;

  // Define readback buffer resource descriptor.
  D3D12_RESOURCE_DESC readbackBufferResourceDesc = {};
  readbackBufferResourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  readbackBufferResourceDesc.Alignment = 0;
  readbackBufferResourceDesc.Width = readbackBufferSize;
  readbackBufferResourceDesc.Height = 1;
  readbackBufferResourceDesc.DepthOrArraySize = 1;
  readbackBufferResourceDesc.MipLevels = 1;
  readbackBufferResourceDesc.Format = DXGI_FORMAT_UNKNOWN;
  readbackBufferResourceDesc.SampleDesc = DXGI_SAMPLE_DESC{1, 0};
  readbackBufferResourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  readbackBufferResourceDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

  // Create the readback buffer.
  ComPtr<ID3D12Resource> readbackBuffer;
  ThrowIfFailed(dx12Device->CreateCommittedResource(
      &readbackHeapProperties, D3D12_HEAP_FLAG_NONE,
      &readbackBufferResourceDesc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
      IID_PPV_ARGS(&readbackBuffer)));

  // Set the copy source and destination footprint/locations using the
  // actual footprint returned by GetCopyableFootprints.
  D3D12_TEXTURE_COPY_LOCATION copyDest = {};
  copyDest.pResource = readbackBuffer.Get();
  copyDest.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
  copyDest.PlacedFootprint = readbackFootprint;

  D3D12_TEXTURE_COPY_LOCATION copySrc = {};
  copySrc.pResource = m_dx12Texture.Get();
  copySrc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  copySrc.SubresourceIndex = 0;

#ifdef VERBOSE_PRINT
  std::cout << "[dx12-trace] CopyTextureRegion readback: srcTexture="
            << m_dx12Texture.Get() << " dstBuffer=" << readbackBuffer.Get()
            << " rowPitch=" << readbackFootprint.Footprint.RowPitch << "\n";
#endif

  // Transition texture from COMMON state (after Level Zero writes and implicit
  // decay) to COPY_SOURCE. With D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS,
  // this transition flushes and invalidates caches to make Level Zero's writes
  // visible to the D3D12 copy engine.
  auto *dx12CommandList = m_device.getDx12CommandList();

  // First, add UAV barrier to ensure Level Zero writes are visible
  D3D12_RESOURCE_BARRIER uavBarrier = {};
  uavBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
  uavBarrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
  uavBarrier.UAV.pResource = m_dx12Texture.Get();
  dx12CommandList->ResourceBarrier(1, &uavBarrier);

  D3D12_RESOURCE_BARRIER readbackBarrier = {};
  readbackBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  readbackBarrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
  readbackBarrier.Transition.pResource = m_dx12Texture.Get();
  readbackBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
  readbackBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
  readbackBarrier.Transition.Subresource =
      D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
  dx12CommandList->ResourceBarrier(1, &readbackBarrier);

  // Copy the texture to our readback buffer.
  dx12CommandList->CopyTextureRegion(&copyDest, 0, 0, 0, &copySrc, nullptr);

  readbackBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
  readbackBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
  dx12CommandList->ResourceBarrier(1, &readbackBarrier);

  // Execute the command list.
  ThrowIfFailed(dx12CommandList->Close());
  ID3D12CommandList *ppCommandLists[] = {dx12CommandList};
  // Reuse dx12CommandQueue variable declared at the beginning of the function
  dx12CommandQueue->ExecuteCommandLists(_countof(ppCommandLists),
                                        ppCommandLists);
  ThrowIfFailed(
      dx12CommandQueue->Signal(m_dx12Fence.Get(), m_sharedFenceValue));

  // Wait for the command list to finish execution and increment the fence
  // value.
  waitDX12Fence();
  m_sharedFenceValue++;

  // Additional fence wait before mapping to ensure all GPU work is complete
  // and caches are flushed. This is critical for cache coherency.
  ThrowIfFailed(
      dx12CommandQueue->Signal(m_dx12Fence.Get(), m_sharedFenceValue));
  waitDX12Fence();
  m_sharedFenceValue++;

  // Map the readback buffer to host visible memory.
  D3D12_RANGE readbackBufferRange{0, readbackBufferSize};
  DType *pReadbackBufferData{};
  ThrowIfFailed(
      readbackBuffer->Map(0, &readbackBufferRange,
                          reinterpret_cast<void **>(&pReadbackBufferData)));

#ifdef VERBOSE_PRINT
  tracePitchedValues("readback buffer after DX12 copy", pReadbackBufferData,
                     m_width, m_height, m_depth, NChannels,
                     readbackFootprint.Footprint.RowPitch, readbackNumRows);
  tracePitchedComparison("host input -> host readback after SYCL processing",
                         m_inputData.data(), pReadbackBufferData, m_width,
                         m_height, m_depth, NChannels,
                         readbackFootprint.Footprint.RowPitch, readbackNumRows);
#endif

  // Additional wait for the GPU. Sometimes the Mapped memory isn't immediately
  // visible to the host. This ensures cache coherency between GPU and CPU.
  ThrowIfFailed(
      dx12CommandQueue->Signal(m_dx12Fence.Get(), m_sharedFenceValue));
  waitDX12Fence();
  m_sharedFenceValue++;

  // Read back the updated texture data and validate it.
  // Use row-by-row access with the actual row pitch to correctly handle any
  // padding between rows in the readback buffer.
  const UINT readbackRowPitch = readbackFootprint.Footprint.RowPitch;
  const UINT readbackSlicePitch = readbackRowPitch * readbackNumRows;
  bool validated = true;
#ifdef VERBOSE_PRINT
  size_t printedValidationValues = 0;
  const size_t validationPrintLimit = traceValueLimit(m_numElems);
#endif
  for (int z = 0; z < m_depth; ++z) {
    for (int y = 0; y < m_height; ++y) {
      const DType *rowData = reinterpret_cast<const DType *>(
          reinterpret_cast<const uint8_t *>(pReadbackBufferData) +
          z * readbackSlicePitch + y * readbackRowPitch);
      for (int x = 0; x < m_width * NChannels; ++x) {
        const int i = (z * m_width * m_height + y * m_width) * NChannels + x;
        auto expected =
            importProbeOnly() ? m_inputData[i] : (m_inputData[i] * 2);
        auto actual = rowData[x];
#ifdef VERBOSE_PRINT
        if (printedValidationValues < validationPrintLimit) {
          std::cout << "[dx12-trace] validate linear=" << i << " z=" << z
                    << " y=" << y << " xChannel=" << x
                    << " input=" << printableValue(m_inputData[i])
                    << " expected=" << printableValue(expected)
                    << " actual=" << printableValue(actual) << "\n";
          ++printedValidationValues;
        }
#endif
        if (actual != expected) {
          validated = false;
#ifdef VERBOSE_PRINT
          std::cout << "Result mismatch at " << i << "! Expected: " << expected
                    << ", Actual: " << actual << std::endl;
#else
          goto done_validation;
#endif
        }
      }
    }
  }
done_validation:;

  // Unmap the readback buffer.
  D3D12_RANGE emptyRange{0, 0};
  readbackBuffer->Unmap(0, &emptyRange);

  // Signal the fence to wait upon before we can clean up DX12 later.
  ThrowIfFailed(
      dx12CommandQueue->Signal(m_dx12Fence.Get(), m_sharedFenceValue));

  return validated;
}

template <int NDims, typename DType, int NChannels>
void DX12InteropTest<NDims, DType, NChannels>::waitDX12Fence(
    DWORD timeoutMilliseconds) {
  // Check the current value of the fence to check if
  // GPU has finished executing the command list.
#ifdef VERBOSE_PRINT
  std::cout << "[dx12-trace] waitDX12Fence: target=" << m_sharedFenceValue
            << " completed=" << m_dx12Fence->GetCompletedValue()
            << " timeoutMs=" << timeoutMilliseconds << "\n";
#endif
  if (m_dx12Fence->GetCompletedValue() < m_sharedFenceValue) {
    // If not, set value fence is to set on completion.
    ThrowIfFailed(m_dx12Fence->SetEventOnCompletion(m_sharedFenceValue,
                                                    m_dx12FenceEvent));
    // Wait for fence to be triggered.
    WaitForSingleObject(m_dx12FenceEvent, timeoutMilliseconds);
  }
#ifdef VERBOSE_PRINT
  std::cout << "[dx12-trace] waitDX12Fence completed: target="
            << m_sharedFenceValue
            << " completed=" << m_dx12Fence->GetCompletedValue() << "\n";
#endif
}

template <int NDims, typename DType, int NChannels>
void DX12InteropTest<NDims, DType, NChannels>::cleanupDX12() {
  // Wait for the command list to finish execution.
  waitDX12Fence();

  // Clean up opened handles
  if (m_sharedSemaphoreHandle != INVALID_HANDLE_VALUE)
    CloseNTHandle(m_sharedSemaphoreHandle);
  CloseNTHandle(m_sharedMemoryHandle);
  CloseHandle(m_dx12FenceEvent);

  // ComPtr handles will be destroyed automatically.
}

template <int NDims, typename DType, int NChannels>
static bool
runTest(DX12SYCLDevice &device, sycl::image_channel_type channelType,
        sycl::range<NDims> globalSize, sycl::range<NDims> localSize) {

  // Skip unorm_int8 tests for Level Zero backend
  if (channelType == sycl::image_channel_type::unorm_int8 &&
      device.getSyclQueue().get_device().get_backend() ==
          sycl::backend::ext_oneapi_level_zero) {
    std::cout << "Skipping unorm_int8 test for Level Zero backend.\n";
    return true;
  }

  syclexp::image_descriptor syclImageDesc{globalSize, NChannels, channelType};

  // Verify ability to allocate the above image descriptor.
  // E.g. LevelZero does not support `unorm` channel types.
  if (!bindless_helpers::memoryAllocationSupported(
          syclImageDesc, syclexp::image_memory_handle_type::opaque_handle,
          device.getSyclQueue())) {
    // We cannot allocate the image memory, skip the test.
    std::cout << "Memory allocation unsupported. Skipping test.\n";
    return true;
  }

  DX12InteropTest<NDims, DType, NChannels> interopTestInstance(
      device, channelType, globalSize, localSize);

  interopTestInstance.initDX12Resources();
  interopTestInstance.callSYCLKernel();
  bool validated = interopTestInstance.validateOutput();
  interopTestInstance.cleanupDX12();

#ifdef VERBOSE_PRINT
  if (!validated) {
    std::cerr << "\tTest failed: NDims " << NDims << " NChannels " << NChannels
              << " image_channel_type "
              << bindless_helpers::channelTypeToString(channelType)
              << ", exiting\n";
    exit(-1);
  } else {
    std::cout << "\tTest passed: NDims " << NDims << " NChannels " << NChannels
              << " image_channel_type "
              << bindless_helpers::channelTypeToString(channelType) << "\n";
  }
#endif

  return validated;
}

int main() {
  DX12SYCLDevice device;

  bool validated = true;
  const bool Only2DUInt32Case = runOnly2DUint32Case();

#ifdef TEST_SMALL_IMAGE_SIZE
  const bool UseExtraSmallSize = envFlagIsSet("EXTRA_SMALL_SIZE");
  const size_t OneDImageSize = UseExtraSmallSize ? 4 : 1024;
  const size_t OneDLocalSize = UseExtraSmallSize ? 4 : 1024;
#else
  const size_t OneDImageSize = 1024;
  const size_t OneDLocalSize = 1024;
#endif

  sycl::range<1> globalSize1{OneDImageSize};
  sycl::range<1> localSize1{OneDLocalSize};

  if (!Only2DUInt32Case) {
    validated &= runTest<1, uint32_t, 1>(
        device, sycl::image_channel_type::unsigned_int32, globalSize1,
        localSize1);
    validated &= runTest<1, uint8_t, 4>(
        device, sycl::image_channel_type::unorm_int8, globalSize1, localSize1);
    validated &= runTest<1, float, 1>(device, sycl::image_channel_type::fp32,
                                      globalSize1, localSize1);
    validated &= runTest<1, sycl::half, 2>(
        device, sycl::image_channel_type::fp16, globalSize1, localSize1);
    validated &= runTest<1, sycl::half, 4>(
        device, sycl::image_channel_type::fp16, globalSize1, localSize1);
  }

#ifdef TEST_SMALL_IMAGE_SIZE
  const size_t SmallImageSize = UseExtraSmallSize ? 4 : 64;
  const size_t SmallLocalSize = UseExtraSmallSize ? 4 : 16;
  const size_t SmallLocalSizeY = UseExtraSmallSize ? 4 : 8;

  sycl::range<2> globalSize2[] = {{SmallImageSize, SmallImageSize},
                                  {SmallImageSize, SmallImageSize},
                                  {SmallImageSize, SmallImageSize},
                                  {SmallImageSize, SmallImageSize},
                                  {SmallImageSize, SmallImageSize}};
#else
  const size_t SmallLocalSize = 16;
  const size_t SmallLocalSizeY = 8;
  sycl::range<2> globalSize2[] = {
      {1024, 1024}, {1920, 1080}, {1920, 1080}, {2048, 2048}, {2048, 2048}};
#endif
  validated &=
      runTest<2, uint32_t, 1>(device, sycl::image_channel_type::unsigned_int32,
                              globalSize2[0], {SmallLocalSize, SmallLocalSize});
  if (!Only2DUInt32Case) {
    validated &= runTest<2, uint8_t, 4>(
        device, sycl::image_channel_type::unorm_int8, globalSize2[1],
        {SmallLocalSize, SmallLocalSizeY});
    validated &=
        runTest<2, float, 1>(device, sycl::image_channel_type::fp32,
                             globalSize2[2], {SmallLocalSize, SmallLocalSizeY});
    validated &= runTest<2, sycl::half, 2>(
        device, sycl::image_channel_type::fp16, globalSize2[3],
        {SmallLocalSize, SmallLocalSize});
    validated &= runTest<2, sycl::half, 4>(
        device, sycl::image_channel_type::fp16, globalSize2[4],
        {SmallLocalSize, SmallLocalSize});
  }

#ifdef TEST_SMALL_IMAGE_SIZE
  sycl::range<3> globalSize3[] = {{SmallImageSize, 16, 4},
                                  {SmallImageSize, 16, 4},
                                  {SmallImageSize, SmallImageSize, 4},
                                  {SmallImageSize, SmallImageSize, 4},
                                  {SmallImageSize, SmallImageSize, 4}};
#else
  sycl::range<3> globalSize3[] = {{1024, 1024, 16},
                                  {1920, 1080, 8},
                                  {1920, 1080, 8},
                                  {2048, 2048, 4},
                                  {2048, 2048, 4}};
#endif
  if (!Only2DUInt32Case) {
    validated &= runTest<3, uint32_t, 1>(
        device, sycl::image_channel_type::unsigned_int32, globalSize3[0],
        {SmallLocalSize, 16, 1});
    validated &= runTest<3, uint8_t, 4>(
        device, sycl::image_channel_type::unorm_int8, globalSize3[1],
        {SmallLocalSize, SmallLocalSizeY, 2});
    validated &= runTest<3, float, 1>(device, sycl::image_channel_type::fp32,
                                      globalSize3[2],
                                      {SmallLocalSize, SmallLocalSizeY, 1});
    validated &= runTest<3, sycl::half, 2>(
        device, sycl::image_channel_type::fp16, globalSize3[3],
        {SmallLocalSize, SmallLocalSize, 1});
    validated &= runTest<3, sycl::half, 4>(
        device, sycl::image_channel_type::fp16, globalSize3[4],
        {SmallLocalSize, SmallLocalSize, 1});
  }

  if (validated) {
    std::cout << "Test passed!" << std::endl;
    return 0;
  }

  std::cerr << "Test failed!" << std::endl;

  return 1;
}
