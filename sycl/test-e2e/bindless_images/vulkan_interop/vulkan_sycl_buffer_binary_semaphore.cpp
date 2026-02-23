// REQUIRES: aspect-ext_oneapi_external_memory_import
// REQUIRES: aspect-ext_oneapi_external_semaphore_import
// REQUIRES: vulkan

// UNSUPPORTED: linux
// UNSUPPORTED-TRACKER: GSD-12371

// RUN: %{build} %link-vulkan -o %t.out %if target-spir %{ -Wno-ignored-attributes %}
// RUN: %{run} %t.out --no-sem
// RUN: %{run} %t.out --dual-sem
// RUN: %{run} %t.out

// clang-format off
/*
  Vulkan/SYCL Buffer + Binary Semaphore Interop Stress Test

  Iteratively round-trips data through a Vulkan buffer and a SYCL kernel,
  synchronized via binary semaphore(s).  No image APIs are used — only
  VkBuffer memory exported and mapped to SYCL USM pointers.

  Semaphore topology is selectable:
    Default (single):  one binary semaphore alternates roles each half-cycle.
    --dual-sem:        two unidirectional binary semaphores (A: Vk->SYCL, B: SYCL->Vk)
                       each semaphore is only ever signaled by one side and consumed by the other.

  Flow (per iteration i):
    1. Vulkan: fill input staging with value i, copy to device buffer, signal vkToSyclSem
    2. SYCL:   wait vkToSyclSem (consumes), kernel: out[j] = in[j] * 2, signal syclToVkSem
    3. Vulkan:  device-wait syclToVkSem (via readback submit + fence), readback output, verify out[j] == i*2

  Build (manual):
    Linux:   clang++ -fsycl -o vsbb.bin vulkan_sycl_buffer_binary_semaphore.cpp -lvulkan -I$VULKAN_SDK/include -L$VULKAN_SDK/lib
    Windows: clang++ -fsycl -o vsbb.exe vulkan_sycl_buffer_binary_semaphore.cpp -Wno-ignored-attributes -lvulkan-1 -I$VULKAN_SDK/Include -L$VULKAN_SDK/Lib

  Usage:
    ./vsbb.bin                  # single binary sem, 100 iterations
    ./vsbb.bin --dual-sem       # two binary sems, 100 iterations
    ./vsbb.bin --iterations 50  # override iteration count
    ./vsbb.bin --no-sem         # baseline (host barriers, no semaphores)
    ./vsbb.bin --size 2048      # override element count
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
  bool dualSem = false;
  int iterations = 100;
  size_t numElements = 1024;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--no-sem")
      useSemaphores = false;
    if (arg == "--dual-sem")
      dualSem = true;
    if (arg == "--iterations" && i + 1 < argc)
      iterations = std::stoi(argv[++i]);
    if (arg == "--size" && i + 1 < argc)
      numElements = std::stoi(argv[++i]);
  }

  VkDeviceSize bufferSize = numElements * sizeof(uint32_t);

  std::string modeStr = "NO-SEM (host barriers)";
  if (useSemaphores)
    modeStr = dualSem ? "DUAL binary semaphores" : "SINGLE binary semaphore";

  std::cout << "Running SYCL Vulkan Buffer + Binary Semaphore Stress Test"
            << std::endl;
  std::cout << "Elements: " << numElements << " | Iterations: " << iterations
            << " | Mode: " << modeStr << std::endl;

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

  // Binary semaphores.
  // vkToSyclSem: Vulkan signals, SYCL consumes.
  // syclToVkSem: SYCL signals, Vulkan consumes.
  // In single-sem mode, both point to the same VkSemaphore.
  VkSemaphore vkToSyclSem = VK_NULL_HANDLE;
  VkSemaphore syclToVkSem = VK_NULL_HANDLE;
  if (useSemaphores) {
    vkToSyclSem = createExportableSemaphore(vkCtx);
    syclToVkSem = dualSem ? createExportableSemaphore(vkCtx) : vkToSyclSem;
  }

  // Command pool + command buffers (resettable)
  VkCommandPool pool;
  {
    VkCommandPoolCreateInfo poolInfo = {
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.queueFamilyIndex = vkCtx.queueFamilyIndex;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK(vkCreateCommandPool(vkCtx.device, &poolInfo, nullptr, &pool));
  }
  VkCommandBuffer cmds[2];
  {
    VkCommandBufferAllocateInfo alloc = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    alloc.commandPool = pool;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = 2;
    VK_CHECK(vkAllocateCommandBuffers(vkCtx.device, &alloc, cmds));
  }
  VkCommandBuffer &cmdUpload = cmds[0];
  VkCommandBuffer &cmdReadback = cmds[1];

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

    // Import semaphore(s)
    syclexp::external_semaphore syclWaitSem{}; // waits on vkToSyclSem
    syclexp::external_semaphore syclSigSem{};  // signals syclToVkSem
    if (useSemaphores) {
#ifdef _WIN32
      HANDLE waitHandle = getSemaphoreHandle(vkCtx, vkToSyclSem);
      auto waitDesc = syclexp::external_semaphore_descriptor<
          syclexp::resource_win32_handle>{
          waitHandle, syclexp::external_semaphore_handle_type::win32_nt_handle};
      syclWaitSem =
          syclexp::import_external_semaphore(waitDesc, device, context);

      if (dualSem) {
        HANDLE sigHandle = getSemaphoreHandle(vkCtx, syclToVkSem);
        auto sigDesc = syclexp::external_semaphore_descriptor<
            syclexp::resource_win32_handle>{
            sigHandle,
            syclexp::external_semaphore_handle_type::win32_nt_handle};
        syclSigSem =
            syclexp::import_external_semaphore(sigDesc, device, context);
      } else {
        syclSigSem = syclWaitSem; // same handle in single-sem mode
      }
#else
      int waitFd = getSemaphoreFd(vkCtx, vkToSyclSem);
      auto waitDesc =
          syclexp::external_semaphore_descriptor<syclexp::resource_fd>{
              waitFd, syclexp::external_semaphore_handle_type::opaque_fd};
      syclWaitSem =
          syclexp::import_external_semaphore(waitDesc, device, context);

      if (dualSem) {
        int sigFd = getSemaphoreFd(vkCtx, syclToVkSem);
        auto sigDesc =
            syclexp::external_semaphore_descriptor<syclexp::resource_fd>{
                sigFd, syclexp::external_semaphore_handle_type::opaque_fd};
        syclSigSem =
            syclexp::import_external_semaphore(sigDesc, device, context);
      } else {
        syclSigSem = syclWaitSem; // same handle in single-sem mode
      }
#endif
    }

    // Map to USM pointers
    uint32_t *inPtr = static_cast<uint32_t *>(
        syclexp::map_external_linear_memory(inExtMem, 0, bufferSize, q));
    uint32_t *outPtr = static_cast<uint32_t *>(
        syclexp::map_external_linear_memory(outExtMem, 0, bufferSize, q));

    std::cout << "[Test] Starting " << iterations << " iteration stress test..."
              << std::endl;

    for (int i = 1; i <= iterations; ++i) {

      // --- 1. Vulkan: fill staging, copy to device, signal vkToSyclSem ---
      {
        void *mapped;
        vkMapMemory(vkCtx.device, inStaging.memory, 0, bufferSize, 0, &mapped);
        auto *data = static_cast<uint32_t *>(mapped);
        for (size_t j = 0; j < numElements; ++j)
          data[j] = (uint32_t)i;
        vkUnmapMemory(vkCtx.device, inStaging.memory);
      }

      vkResetCommandBuffer(cmdUpload, 0);
      VkCommandBufferBeginInfo begin = {
          VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      vkBeginCommandBuffer(cmdUpload, &begin);

      VkBufferCopy copyRegion = {0, 0, bufferSize};
      vkCmdCopyBuffer(cmdUpload, inStaging.buffer, inBuf.buffer, 1,
                      &copyRegion);

      // Barrier: transfer write -> shader read
      VkBufferMemoryBarrier bar = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
      bar.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      bar.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
      bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      bar.buffer = inBuf.buffer;
      bar.offset = 0;
      bar.size = VK_WHOLE_SIZE;
      vkCmdPipelineBarrier(cmdUpload, VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 1,
                           &bar, 0, nullptr);

      vkEndCommandBuffer(cmdUpload);

      VkSubmitInfo uploadSub = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
      uploadSub.commandBufferCount = 1;
      uploadSub.pCommandBuffers = &cmdUpload;
      if (useSemaphores) {
        uploadSub.signalSemaphoreCount = 1;
        uploadSub.pSignalSemaphores = &vkToSyclSem;
      }
      VK_CHECK(vkQueueSubmit(vkCtx.queue, 1, &uploadSub, VK_NULL_HANDLE));
      vkQueueWaitIdle(vkCtx.queue);
      std::cout << "  [" << i << "] Vulkan upload done" << std::flush;

      // --- 2. SYCL: wait vkToSyclSem -> kernel -> signal syclToVkSem ---
      if (useSemaphores) {
        std::cout << ", SYCL sem-wait..." << std::flush;
        q.ext_oneapi_wait_external_semaphore(syclWaitSem);
        std::cout << "ok" << std::flush;
      }

      q.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::range<1>(numElements), [=](sycl::item<1> item) {
          size_t id = item.get_id(0);
          outPtr[id] = inPtr[id] * 2;
        });
      });

      if (useSemaphores) {
        std::cout << ", SYCL sem-signal..." << std::flush;
        q.ext_oneapi_signal_external_semaphore(syclSigSem);
        std::cout << "ok" << std::flush;
      }
      q.wait();
      std::cout << ", SYCL done" << std::flush;

      // --- 3. Vulkan: device-wait syclToVkSem + readback via fence ---
      vkResetCommandBuffer(cmdReadback, 0);
      vkBeginCommandBuffer(cmdReadback, &begin);

      VkBufferMemoryBarrier readBar = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
      readBar.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
      readBar.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
      readBar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      readBar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      readBar.buffer = outBuf.buffer;
      readBar.offset = 0;
      readBar.size = VK_WHOLE_SIZE;
      vkCmdPipelineBarrier(cmdReadback, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1,
                           &readBar, 0, nullptr);

      VkBufferCopy readCopy = {0, 0, bufferSize};
      vkCmdCopyBuffer(cmdReadback, outBuf.buffer, outStaging.buffer, 1,
                      &readCopy);
      vkEndCommandBuffer(cmdReadback);

      VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
      VkSubmitInfo readSub = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
      readSub.commandBufferCount = 1;
      readSub.pCommandBuffers = &cmdReadback;
      if (useSemaphores) {
        readSub.waitSemaphoreCount = 1;
        readSub.pWaitSemaphores = &syclToVkSem;
        readSub.pWaitDstStageMask = &waitStage;
      }
      vkResetFences(vkCtx.device, 1, &fence);
      VK_CHECK(vkQueueSubmit(vkCtx.queue, 1, &readSub, fence));
      std::cout << ", vk-fence..." << std::flush;
      VkResult fenceRes =
          vkWaitForFences(vkCtx.device, 1, &fence, VK_TRUE, 5000000000ULL);
      if (fenceRes == VK_TIMEOUT) {
        std::cerr << std::endl
                  << "TIMEOUT on fence at iteration " << i << std::endl;
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
      syclexp::release_external_semaphore(syclWaitSem, device, context);
      if (dualSem)
        syclexp::release_external_semaphore(syclSigSem, device, context);
    }
    syclexp::release_external_memory(inExtMem, device, context);
    syclexp::release_external_memory(outExtMem, device, context);

  } catch (sycl::exception &e) {
    std::cerr << "SYCL Exception: " << e.what() << std::endl;
    return 1;
  }

  // Cleanup Vulkan
  vkDestroyFence(vkCtx.device, fence, nullptr);
  if (vkToSyclSem != VK_NULL_HANDLE)
    vkDestroySemaphore(vkCtx.device, vkToSyclSem, nullptr);
  if (dualSem && syclToVkSem != VK_NULL_HANDLE)
    vkDestroySemaphore(vkCtx.device, syclToVkSem, nullptr);
  vkDestroyCommandPool(vkCtx.device, pool, nullptr);
  cleanupBuffer(vkCtx, inBuf);
  cleanupBuffer(vkCtx, outBuf);
  cleanupBuffer(vkCtx, inStaging);
  cleanupBuffer(vkCtx, outStaging);
  vkDestroyDevice(vkCtx.device, nullptr);
  vkDestroyInstance(vkCtx.instance, nullptr);

  return 0;
}
