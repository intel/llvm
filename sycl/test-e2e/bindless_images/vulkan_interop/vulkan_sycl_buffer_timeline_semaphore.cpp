// REQUIRES: aspect-ext_oneapi_external_memory_import
// REQUIRES: aspect-ext_oneapi_external_semaphore_import
// REQUIRES: vulkan

// hangs on linux
// UNSUPPORTED: linux
// UNSUPPORTED-TRACKER: GSD-12371

//
// UNSUPPORTED: windows && gpu-intel-gen12
// UNSUPPORTED-TRACKER: URLZA-723

// RUN: %{build} %link-vulkan -o %t.out %if target-spir %{ -Wno-ignored-attributes %}
// RUN: %{run} %t.out --no-sem
// RUN: %{run} %t.out

// clang-format off
/*
  Vulkan/SYCL Buffer + Timeline Semaphore Interop Stress Test

  Iteratively round-trips data through a Vulkan buffer and a SYCL kernel,
  synchronized via a single timeline semaphore with monotonically increasing
  values.  No image APIs are used — only VkBuffer memory exported and mapped
  to SYCL USM pointers.

  Semaphore protocol:
    One timeline semaphore, initial value 0.
    Vulkan signals odd values: 2*i - 1  (for iteration i = 1..N)
    SYCL   signals even values: 2*i

  Flow (per iteration i):
    1. Vulkan: fill input staging with value i, copy to device buffer,
               signal sem = 2*i-1
    2. SYCL:   wait sem >= 2*i-1, kernel: out[j] = in[j] * 2,
               signal sem = 2*i
    3. Vulkan:  device-wait sem >= 2*i (via readback submit + fence),
                readback output, verify out[j] == i*2


  Linux:   clang++ -fsycl -o vsbt.bin vulkan_sycl_buffer_timeline_semaphore.cpp -lvulkan -I$VULKAN_SDK/include -L$VULKAN_SDK/lib
  Windows: clang++ -fsycl -o vsbt.exe vulkan_sycl_buffer_timeline_semaphore.cpp -Wno-ignored-attributes -lvulkan-1 -I$VULKAN_SDK/Include -L$VULKAN_SDK/Lib

  Usage:
    ./vsbt.bin                  # 100 iterations
    ./vsbt.bin --iterations 50  # override iteration count
    ./vsbt.bin --no-sem         # baseline (host barriers, no semaphores)
    ./vsbt.bin --size 2048      # override element count
*/
// clang-format on

#include "vulkan_setup.hpp"
#include <iostream>
#include <string>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>
#include <vector>

namespace syclexp = sycl::ext::oneapi::experimental;

// ---------------------------------------------------------
// MAIN
// ---------------------------------------------------------
int main(int argc, char **argv) {
  bool useSemaphores = true;
  int iterations = 100;
  size_t numElements = 1024;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--no-sem")
      useSemaphores = false;
    if (arg == "--iterations" && i + 1 < argc)
      iterations = std::stoi(argv[++i]);
    if (arg == "--size" && i + 1 < argc)
      numElements = std::stoi(argv[++i]);
  }

  VkDeviceSize bufferSize = numElements * sizeof(uint32_t);

  std::cout << "Running SYCL Vulkan Buffer + Timeline Semaphore Stress Test"
            << std::endl;
  std::cout << "Elements: " << numElements << " | Iterations: " << iterations
            << " | Semaphores: " << (useSemaphores ? "ON" : "OFF") << std::endl;

  // VULKAN SETUP
  VulkanContext vkCtx = createVulkanContext();

  // Exportable device-local buffers
  BufferResources inBuf = createExportableBuffer(
      vkCtx, bufferSize,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  BufferResources outBuf = createExportableBuffer(
      vkCtx, bufferSize,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

  // Host-visible staging buffers (reused across iterations)
  BufferResources inStaging =
      createStagingBuffer(vkCtx, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
  BufferResources outStaging =
      createStagingBuffer(vkCtx, bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT);

  // Single timeline semaphore: one monotonic sequence for both directions.
  // Vulkan signals odd values (2*i-1), SYCL signals even values (2*i).
  VkSemaphore sem = VK_NULL_HANDLE;
  if (useSemaphores) {
    sem = createExportableTimelineSemaphore(vkCtx);
  }

  // Command pool + command buffer (resettable)
  VkCommandPool pool;
  {
    VkCommandPoolCreateInfo poolInfo = {
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.queueFamilyIndex = vkCtx.queueFamilyIndex;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK(vkCreateCommandPool(vkCtx.device, &poolInfo, nullptr, &pool));
  }
  VkCommandBuffer cmd;
  {
    VkCommandBufferAllocateInfo alloc = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    alloc.commandPool = pool;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(vkCtx.device, &alloc, &cmd));
  }

  // Fence for host synchronization of readback
  VkFence fence;
  {
    VkFenceCreateInfo fenceInfo = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VK_CHECK(vkCreateFence(vkCtx.device, &fenceInfo, nullptr, &fence));
  }

  // SYCL INTEROP
  try {
    sycl::queue q;
    auto device = q.get_device();
    auto context = q.get_context();

    std::cout << "[SYCL] Device: "
              << device.get_info<sycl::info::device::name>() << std::endl;

// Import input buffer
#ifdef _WIN32
    HANDLE inHandle = getMemHandle(vkCtx, inBuf.memory);
    syclexp::external_mem_descriptor<syclexp::resource_win32_handle> inDesc{
        inHandle, syclexp::external_mem_handle_type::win32_nt_handle,
        bufferSize};
#else
    int inFd = getMemFd(vkCtx, inBuf.memory);
    syclexp::external_mem_descriptor<syclexp::resource_fd> inDesc{
        inFd, syclexp::external_mem_handle_type::opaque_fd, bufferSize};
#endif
    syclexp::external_mem inExtMem =
        syclexp::import_external_memory(inDesc, device, context);

// Import output buffer
#ifdef _WIN32
    HANDLE outHandle = getMemHandle(vkCtx, outBuf.memory);
    syclexp::external_mem_descriptor<syclexp::resource_win32_handle> outDesc{
        outHandle, syclexp::external_mem_handle_type::win32_nt_handle,
        bufferSize};
#else
    int outFd = getMemFd(vkCtx, outBuf.memory);
    syclexp::external_mem_descriptor<syclexp::resource_fd> outDesc{
        outFd, syclexp::external_mem_handle_type::opaque_fd, bufferSize};
#endif
    syclexp::external_mem outExtMem =
        syclexp::import_external_memory(outDesc, device, context);

    // Import timeline semaphore (single handle for both wait and signal)
    syclexp::external_semaphore syclSem{};
    if (useSemaphores) {
#ifdef _WIN32
      HANDLE semHandle = getSemaphoreHandle(vkCtx, sem);
      auto semDesc = syclexp::external_semaphore_descriptor<
          syclexp::resource_win32_handle>{
          semHandle,
          syclexp::external_semaphore_handle_type::timeline_win32_nt_handle};
#else
      int semFd = getSemaphoreFd(vkCtx, sem);
      auto semDesc =
          syclexp::external_semaphore_descriptor<syclexp::resource_fd>{
              semFd, syclexp::external_semaphore_handle_type::timeline_fd};
#endif
      syclSem = syclexp::import_external_semaphore(semDesc, device, context);
    }

    // Map to USM pointers
    uint32_t *inPtr = static_cast<uint32_t *>(
        syclexp::map_external_linear_memory(inExtMem, 0, bufferSize, q));
    uint32_t *outPtr = static_cast<uint32_t *>(
        syclexp::map_external_linear_memory(outExtMem, 0, bufferSize, q));

    std::cout << "[Test] Starting " << iterations << " iteration stress test..."
              << std::endl;

    for (int i = 1; i <= iterations; ++i) {
      uint64_t vkSignalVal = (uint64_t)(2 * i - 1); // Vulkan signals odd
      uint64_t syclSignalVal = (uint64_t)(2 * i);   // SYCL signals even

      //  Vulkan: fill staging, copy to device, signal sem(2i-1)
      {
        void *mapped;
        vkMapMemory(vkCtx.device, inStaging.memory, 0, bufferSize, 0, &mapped);
        auto *data = static_cast<uint32_t *>(mapped);
        for (size_t j = 0; j < numElements; ++j)
          data[j] = (uint32_t)i;
        vkUnmapMemory(vkCtx.device, inStaging.memory);
      }

      vkResetCommandBuffer(cmd, 0);
      VkCommandBufferBeginInfo begin = {
          VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      vkBeginCommandBuffer(cmd, &begin);

      VkBufferCopy copyRegion = {0, 0, bufferSize};
      vkCmdCopyBuffer(cmd, inStaging.buffer, inBuf.buffer, 1, &copyRegion);

      // Barrier: transfer write -> shader read
      VkBufferMemoryBarrier bar = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
      bar.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      bar.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
      bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      bar.buffer = inBuf.buffer;
      bar.offset = 0;
      bar.size = VK_WHOLE_SIZE;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 1,
                           &bar, 0, nullptr);

      vkEndCommandBuffer(cmd);

      VkTimelineSemaphoreSubmitInfo timelineInfo = {
          VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
      timelineInfo.signalSemaphoreValueCount = 1;
      timelineInfo.pSignalSemaphoreValues = &vkSignalVal;

      VkSubmitInfo uploadSub = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
      uploadSub.pNext = &timelineInfo;
      uploadSub.commandBufferCount = 1;
      uploadSub.pCommandBuffers = &cmd;
      if (useSemaphores) {
        uploadSub.signalSemaphoreCount = 1;
        uploadSub.pSignalSemaphores = &sem;
      }
      VK_CHECK(vkQueueSubmit(vkCtx.queue, 1, &uploadSub, VK_NULL_HANDLE));
      vkQueueWaitIdle(vkCtx.queue);
      std::cout << "  [" << i << "] Vulkan upload done" << std::flush;

      // SYCL: wait sem(2i-1), kernel, signal sem(2i)
      if (useSemaphores) {
        std::cout << ", SYCL sem-wait(" << vkSignalVal << ")..." << std::flush;
        q.ext_oneapi_wait_external_semaphore(syclSem, vkSignalVal);
        std::cout << "ok" << std::flush;
      }

      q.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::range<1>(numElements), [=](sycl::item<1> item) {
          size_t id = item.get_id(0);
          outPtr[id] = inPtr[id] * 2;
        });
      });

      if (useSemaphores) {
        std::cout << ", SYCL sem-signal(" << syclSignalVal << ")..."
                  << std::flush;
        q.ext_oneapi_signal_external_semaphore(syclSem, syclSignalVal);
        std::cout << "ok" << std::flush;
      }
      q.wait();
      std::cout << ", SYCL done" << std::flush;

      // Vulkan: device-wait sem(2i) via readback submit + fence
      vkResetCommandBuffer(cmd, 0);
      vkBeginCommandBuffer(cmd, &begin);

      VkBufferMemoryBarrier readBar = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
      readBar.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
      readBar.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
      readBar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      readBar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      readBar.buffer = outBuf.buffer;
      readBar.offset = 0;
      readBar.size = VK_WHOLE_SIZE;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1,
                           &readBar, 0, nullptr);

      VkBufferCopy readCopy = {0, 0, bufferSize};
      vkCmdCopyBuffer(cmd, outBuf.buffer, outStaging.buffer, 1, &readCopy);
      vkEndCommandBuffer(cmd);

      // Device-side semaphore wait + fence
      VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
      VkTimelineSemaphoreSubmitInfo readTimeline = {
          VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
      readTimeline.waitSemaphoreValueCount = 1;
      readTimeline.pWaitSemaphoreValues = &syclSignalVal;

      VkSubmitInfo readSub = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
      readSub.commandBufferCount = 1;
      readSub.pCommandBuffers = &cmd;
      if (useSemaphores) {
        readSub.pNext = &readTimeline;
        readSub.waitSemaphoreCount = 1;
        readSub.pWaitSemaphores = &sem;
        readSub.pWaitDstStageMask = &waitStage;
      }
      vkResetFences(vkCtx.device, 1, &fence);
      VK_CHECK(vkQueueSubmit(vkCtx.queue, 1, &readSub, fence));
      std::cout << ", vk-fence..." << std::flush;
      VkResult fenceRes =
          vkWaitForFences(vkCtx.device, 1, &fence, VK_TRUE, 5000000000ULL);
      if (fenceRes == VK_TIMEOUT) {
        std::cerr << std::endl
                  << "TIMEOUT on fence! sem wait for " << syclSignalVal
                  << std::endl;
        if (useSemaphores) {
          uint64_t curVal;
          vkGetSemaphoreCounterValue(vkCtx.device, sem, &curVal);
          std::cerr << "  sem counter = " << curVal << std::endl;
        }
        return 1;
      }
      VK_CHECK(fenceRes);
      std::cout << "ok" << std::flush;

      // Verify
      {
        void *mapped;
        vkMapMemory(vkCtx.device, outStaging.memory, 0, bufferSize, 0, &mapped);
        auto *data = static_cast<uint32_t *>(mapped);
        uint32_t expected = (uint32_t)i * 2;
        int errors = 0;
        for (size_t j = 0; j < numElements; ++j) {
          if (data[j] != expected) {
            if (errors++ < 5)
              std::cerr << "  [" << j << "]: got " << data[j] << " expected "
                        << expected << std::endl;
          }
        }
        vkUnmapMemory(vkCtx.device, outStaging.memory);
        if (errors > 0) {
          std::cerr << std::endl
                    << "FAILURE at iteration " << i << ": " << errors
                    << " mismatches" << std::endl;
          return 1;
        }
      }

      if (i % 25 == 0 || i == 1)
        std::cout << " PASS" << std::endl;
      else
        std::cout << " ok" << std::endl;
    }

    std::cout << "SUCCESS! All " << iterations << " iterations passed."
              << std::endl;

    // Cleanup SYCL
    syclexp::unmap_external_linear_memory(inPtr, q);
    syclexp::unmap_external_linear_memory(outPtr, q);
    if (useSemaphores) {
      syclexp::release_external_semaphore(syclSem, device, context);
    }
    syclexp::release_external_memory(inExtMem, device, context);
    syclexp::release_external_memory(outExtMem, device, context);

  } catch (sycl::exception &e) {
    std::cerr << "SYCL Exception: " << e.what() << std::endl;
    return 1;
  }

  // Cleanup Vulkan
  if (sem != VK_NULL_HANDLE)
    vkDestroySemaphore(vkCtx.device, sem, nullptr);

  vkDestroyFence(vkCtx.device, fence, nullptr);
  vkDestroyCommandPool(vkCtx.device, pool, nullptr);
  cleanupBuffer(vkCtx, inBuf);
  cleanupBuffer(vkCtx, outBuf);
  cleanupBuffer(vkCtx, inStaging);
  cleanupBuffer(vkCtx, outStaging);
  vkDestroyDevice(vkCtx.device, nullptr);
  vkDestroyInstance(vkCtx.instance, nullptr);

  return 0;
}
