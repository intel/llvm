// REQUIRES: aspect-ext_oneapi_external_memory_import
// REQUIRES: vulkan

// RUN: %{build} %link-vulkan -o %t.out %if target-spir %{ -Wno-ignored-attributes %}
// RUN: %{run} %t.out
// RUN: %{run} %t.out --semaphores

// DMABUF not on Windows.
// RUN: %{run} %t.out --dmabuf
// RUN: %{run} %t.out --semaphores --dmabuf

// clang-format off
/*
  Vulkan/SYCL Buffer Interop Test (USM)

  This test does not use any image specific APIs.
  It is only testing the exportation of VkBuffer memory, and importing and
  mapping the VkBuffer memory into SYCL device memory. The imported SYCL device
  memory is then manipulated purely through pointers inside the kernel.

  Note that this test just checks the basics.  We have others that push the system harder. 


  clang++ -fsycl  -o vsb.bin vulkan_sycl_buffer.cpp -lvulkan -I$VULKAN_SDK/include -L$VULKAN_SDK/lib


  clang++ -fsycl  -o vsb.exe vulkan_sycl_buffer.cpp -Wno-ignored-attributes -lvulkan-1 -I$VULKAN_SDK/Include -L$VULKAN_SDK/Lib



  Features:
  - Creates Vulkan Storage Buffers (Input/Output)
  - Exports memory handles (Win32/FD/DMABUF)
  - Imports to SYCL via generic bindless_images extension (external_memory)
  - Maps to USM Pointers (map_external_linear_memory)
  - Kernel: Output[i] = Input[i] * 2
  - Verifies results on Host via Vulkan

  Usage:
    ./vsb.bin
    ./vsb.bin --semaphores
    ./vsb.bin --dmabuf              # note dmabuf is Linux only.
    ./vsb.bin --size 1024
*/
// clang-format on

#include "vulkan_setup.hpp"
#include <numeric>
#include <sycl/builtins.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>
#include <sycl/image.hpp>
#include <vector>

namespace syclexp = sycl::ext::oneapi::experimental;

// ---------------------------------------------------------
// DATA HELPERS
// ---------------------------------------------------------

// Upload: Host -> Staging -> Device
void uploadBufferData(VulkanContext &ctx, BufferResources &dst,
                      const std::vector<uint32_t> &data) {
  VkDeviceSize dataSize = data.size() * sizeof(uint32_t);

  VkBuffer stagingBuffer;
  VkDeviceMemory stagingMemory;
  VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = dataSize;
  bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  vkCreateBuffer(ctx.device, &bi, nullptr, &stagingBuffer);

  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(ctx.device, stagingBuffer, &req);
  VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = req.size;
  ai.memoryTypeIndex = findMemoryType(ctx.physicalDevice, req.memoryTypeBits,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  vkAllocateMemory(ctx.device, &ai, nullptr, &stagingMemory);
  vkBindBufferMemory(ctx.device, stagingBuffer, stagingMemory, 0);

  void *mapped;
  vkMapMemory(ctx.device, stagingMemory, 0, dataSize, 0, &mapped);
  memcpy(mapped, data.data(), (size_t)dataSize);
  vkUnmapMemory(ctx.device, stagingMemory);

  // One-shot Copy
  VkCommandPoolCreateInfo poolInfo = {
      VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  poolInfo.queueFamilyIndex = ctx.queueFamilyIndex;
  VkCommandPool pool;
  vkCreateCommandPool(ctx.device, &poolInfo, nullptr, &pool);
  VkCommandBuffer cmd;
  VkCommandBufferAllocateInfo alloc = {
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  alloc.commandPool = pool;
  alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc.commandBufferCount = 1;
  vkAllocateCommandBuffers(ctx.device, &alloc, &cmd);

  VkCommandBufferBeginInfo begin = {
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  vkBeginCommandBuffer(cmd, &begin);
  VkBufferCopy copyRegion = {0, 0, dataSize};
  vkCmdCopyBuffer(cmd, stagingBuffer, dst.buffer, 1, &copyRegion);
  vkEndCommandBuffer(cmd);

  VkSubmitInfo submit = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &cmd;
  vkQueueSubmit(ctx.queue, 1, &submit, VK_NULL_HANDLE);
  vkQueueWaitIdle(ctx.queue);

  vkDestroyCommandPool(ctx.device, pool, nullptr);
  vkDestroyBuffer(ctx.device, stagingBuffer, nullptr);
  vkFreeMemory(ctx.device, stagingMemory, nullptr);
}

// Download: Device -> Staging -> Host
std::vector<uint32_t> downloadBufferData(VulkanContext &ctx,
                                         BufferResources &src, size_t count) {
  VkDeviceSize dataSize = count * sizeof(uint32_t);
  std::vector<uint32_t> result(count);

  VkBuffer stagingBuffer;
  VkDeviceMemory stagingMemory;
  VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = dataSize;
  bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  vkCreateBuffer(ctx.device, &bi, nullptr, &stagingBuffer);

  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(ctx.device, stagingBuffer, &req);
  VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = req.size;
  ai.memoryTypeIndex = findMemoryType(ctx.physicalDevice, req.memoryTypeBits,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  vkAllocateMemory(ctx.device, &ai, nullptr, &stagingMemory);
  vkBindBufferMemory(ctx.device, stagingBuffer, stagingMemory, 0);

  VkCommandPoolCreateInfo poolInfo = {
      VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  poolInfo.queueFamilyIndex = ctx.queueFamilyIndex;
  VkCommandPool pool;
  vkCreateCommandPool(ctx.device, &poolInfo, nullptr, &pool);
  VkCommandBuffer cmd;
  VkCommandBufferAllocateInfo alloc = {
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  alloc.commandPool = pool;
  alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc.commandBufferCount = 1;
  vkAllocateCommandBuffers(ctx.device, &alloc, &cmd);

  VkCommandBufferBeginInfo begin = {
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  vkBeginCommandBuffer(cmd, &begin);

  // Barrier: SYCL Write -> Transfer Read
  VkBufferMemoryBarrier barrier = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
  barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.buffer = src.buffer;
  barrier.offset = 0;
  barrier.size = VK_WHOLE_SIZE;

  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1,
                       &barrier, 0, nullptr);

  VkBufferCopy copyRegion = {0, 0, dataSize};
  vkCmdCopyBuffer(cmd, src.buffer, stagingBuffer, 1, &copyRegion);
  vkEndCommandBuffer(cmd);

  VkSubmitInfo submit = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &cmd;
  vkQueueSubmit(ctx.queue, 1, &submit, VK_NULL_HANDLE);
  vkQueueWaitIdle(ctx.queue);

  void *mapped;
  vkMapMemory(ctx.device, stagingMemory, 0, dataSize, 0, &mapped);
  memcpy(result.data(), mapped, (size_t)dataSize);
  vkUnmapMemory(ctx.device, stagingMemory);

  vkDestroyCommandPool(ctx.device, pool, nullptr);
  vkDestroyBuffer(ctx.device, stagingBuffer, nullptr);
  vkFreeMemory(ctx.device, stagingMemory, nullptr);
  return result;
}

// ---------------------------------------------------------
// MAIN TEST
// ---------------------------------------------------------
int main(int argc, char **argv) {
  bool useSemaphores = false;
  bool useDmaBuf = false;
  size_t numElements = 1024;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--semaphores")
      useSemaphores = true;
    if (arg == "--dmabuf")
      useDmaBuf = true;
    if (arg == "--size" && i + 1 < argc)
      numElements = std::stoi(argv[++i]);
  }

#ifdef _WIN32
  if (useDmaBuf) {
    std::cerr
        << "WARNING: --dmabuf ignored on Windows. Using Opaque Win32 Handles."
        << std::endl;
    useDmaBuf = false;
  }
  VkExternalMemoryHandleTypeFlagBits currentHandleType =
      VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
  VkExternalMemoryHandleTypeFlagBits currentHandleType =
      useDmaBuf ? VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT
                : VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif

  std::cout << "Running SYCL Vulkan Buffer Test" << std::endl;
  std::cout << "Elements: " << numElements
            << " | Semaphores: " << (useSemaphores ? "ON" : "OFF")
            << " | Type: " << (useDmaBuf ? "DMA_BUF" : "OPAQUE") << std::endl;

  // VULKAN SETUP
  VulkanContext vkCtx = createVulkanContext();
  VkDeviceSize bufferSize = numElements * sizeof(uint32_t);

  // Create Input and Output Buffers
  BufferResources inBuf = createExportableBuffer(
      vkCtx, bufferSize,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      currentHandleType);
  BufferResources outBuf = createExportableBuffer(
      vkCtx, bufferSize,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      currentHandleType);

  // Create Semaphores
  VkSemaphore signalSem = VK_NULL_HANDLE;
  VkSemaphore waitSem = VK_NULL_HANDLE;
  if (useSemaphores) {
    signalSem = createExportableSemaphore(vkCtx); // SYCL waits on this
    waitSem = createExportableSemaphore(vkCtx);   // SYCL signals this
  }

  // Initialize Input Data
  std::vector<uint32_t> hostInput(numElements);
  std::iota(hostInput.begin(), hostInput.end(), 0);
  uploadBufferData(vkCtx, inBuf, hostInput);

  if (useSemaphores) {
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &signalSem;
    vkQueueSubmit(vkCtx.queue, 1, &si, VK_NULL_HANDLE);
  } else {
    vkQueueWaitIdle(vkCtx.queue);
  }

  // SYCL INTEROP
  try {
    sycl::queue q;
    auto device = q.get_device();
    auto context = q.get_context();

// Import Input Buffer
#ifdef _WIN32
    HANDLE inHandle = getMemHandle(vkCtx, inBuf.memory);
    syclexp::external_mem_descriptor<syclexp::resource_win32_handle> inDesc{
        inHandle, syclexp::external_mem_handle_type::win32_nt_handle,
        bufferSize};
#else
    int inFd = getMemFd(vkCtx, inBuf.memory, currentHandleType);
    auto syclHandleType = useDmaBuf
                              ? syclexp::external_mem_handle_type::dma_buf
                              : syclexp::external_mem_handle_type::opaque_fd;
    syclexp::external_mem_descriptor<syclexp::resource_fd> inDesc{
        inFd, syclHandleType, bufferSize};
#endif
    syclexp::external_mem inExtMem =
        syclexp::import_external_memory(inDesc, device, context);

// Import Output Buffer
#ifdef _WIN32
    HANDLE outHandle = getMemHandle(vkCtx, outBuf.memory);
    syclexp::external_mem_descriptor<syclexp::resource_win32_handle> outDesc{
        outHandle, syclexp::external_mem_handle_type::win32_nt_handle,
        bufferSize};
#else
    int outFd = getMemFd(vkCtx, outBuf.memory, currentHandleType);
    syclexp::external_mem_descriptor<syclexp::resource_fd> outDesc{
        outFd, syclHandleType, bufferSize};
#endif
    syclexp::external_mem outExtMem =
        syclexp::import_external_memory(outDesc, device, context);

    // Import Semaphores
    std::optional<syclexp::external_semaphore> extSignalSem;
    std::optional<syclexp::external_semaphore> extWaitSem;

    if (useSemaphores) {
#ifdef _WIN32
      auto waitDesc = syclexp::external_semaphore_descriptor<
          syclexp::resource_win32_handle>{
          getSemaphoreHandle(vkCtx, signalSem),
          syclexp::external_semaphore_handle_type::win32_nt_handle};
      auto signalDesc = syclexp::external_semaphore_descriptor<
          syclexp::resource_win32_handle>{
          getSemaphoreHandle(vkCtx, waitSem),
          syclexp::external_semaphore_handle_type::win32_nt_handle};
#else
      auto waitDesc =
          syclexp::external_semaphore_descriptor<syclexp::resource_fd>{
              getSemaphoreFd(vkCtx, signalSem),
              syclexp::external_semaphore_handle_type::opaque_fd};
      auto signalDesc =
          syclexp::external_semaphore_descriptor<syclexp::resource_fd>{
              getSemaphoreFd(vkCtx, waitSem),
              syclexp::external_semaphore_handle_type::opaque_fd};
#endif
      extWaitSem =
          syclexp::import_external_semaphore(waitDesc, device, context);
      extSignalSem =
          syclexp::import_external_semaphore(signalDesc, device, context);
    }

    // Map to USM
    uint32_t *inPtr = static_cast<uint32_t *>(
        syclexp::map_external_linear_memory(inExtMem, 0, bufferSize, q));
    uint32_t *outPtr = static_cast<uint32_t *>(
        syclexp::map_external_linear_memory(outExtMem, 0, bufferSize, q));

    // --- THE DANCE ---
    sycl::event waitEvent;
    if (useSemaphores) {
      waitEvent = q.submit([&](sycl::handler &h) {
        h.ext_oneapi_wait_external_semaphore(*extWaitSem);
      });
    }

    sycl::event kernelEvent = q.submit([&](sycl::handler &h) {
      if (useSemaphores)
        h.depends_on(waitEvent);
      h.parallel_for(sycl::range<1>(numElements), [=](sycl::item<1> item) {
        size_t id = item.get_id(0);
        outPtr[id] = inPtr[id] * 2;
      });
    });

    if (useSemaphores) {
      q.submit([&](sycl::handler &h) {
        h.depends_on(kernelEvent);
        h.ext_oneapi_signal_external_semaphore(*extSignalSem);
      });
    }

    q.wait();

    // Cleanup SYCL
    syclexp::unmap_external_linear_memory(inPtr, q);
    syclexp::unmap_external_linear_memory(outPtr, q);
    syclexp::release_external_memory(inExtMem, device, context);
    syclexp::release_external_memory(outExtMem, device, context);
    if (useSemaphores) {
      syclexp::release_external_semaphore(*extWaitSem, device, context);
      syclexp::release_external_semaphore(*extSignalSem, device, context);
    }

  } catch (sycl::exception &e) {
    std::cerr << "SYCL Exception: " << e.what() << std::endl;
    return 1;
  }

  // VERIFY
  std::vector<uint32_t> result = downloadBufferData(vkCtx, outBuf, numElements);
  int errors = 0;
  for (size_t i = 0; i < numElements; ++i) {
    uint32_t expected = i * 2;
    if (result[i] != expected) {
      if (errors++ < 10)
        std::cerr << "Mismatch at " << i << " Got " << result[i] << " Exp "
                  << expected << std::endl;
    }
  }

  if (errors == 0)
    std::cout << "SUCCESS! All " << numElements << " elements verified."
              << std::endl;
  else
    std::cout << "FAILURE! " << errors << " mismatches." << std::endl;

  cleanupBuffer(vkCtx, inBuf);
  cleanupBuffer(vkCtx, outBuf);

  if (useSemaphores) {
    vkDestroySemaphore(vkCtx.device, signalSem, nullptr);
    vkDestroySemaphore(vkCtx.device, waitSem, nullptr);
  }
  vkDestroyDevice(vkCtx.device, nullptr);
  vkDestroyInstance(vkCtx.instance, nullptr);

  return (errors == 0) ? 0 : 1;
}