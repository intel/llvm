// REQUIRES: aspect-ext_oneapi_external_semaphore_import
// REQUIRES: windows
// REQUIRES: vulkan

// The driver does not yet support importing Vulkan-created named timeline
// semaphores by name on Windows. Nor do we yet knjow the driver version where
// that is coming. But it's best to get this under test now.
// XFAIL: *
// XFAIL-TRACKER: GSD-12837

// RUN: %{build} %link-vulkan -o %t.out %if target-spir %{ -Wno-ignored-attributes %}
// RUN: %{run} %t.out

// clang-format off
/*
  Vulkan/SYCL Timeline Semaphore by Name Interop Test

  Verifies that a Vulkan-created named timeline semaphore can be imported
  into SYCL via resource_win32_name + timeline_win32_nt_handle.  This is
  the same-vendor Intel-Vulkan -> Intel-L0 lane: Intel Vulkan registers
  timeline semaphore names as DXGK sync objects, which NEO opens via
  D3DKMT_OPENSYNCOBJECTNTHANDLEFROMNAME.  Import success proves the name
  survived SYCL -> UR -> L0 -> NEO and that the underlying sync object is
  reachable from L0's opener.

  A second import with a non-ASCII name (\u00E9 + \u6C14) additionally exercises
  the UTF-16 -> UTF-8 -> UTF-16 round-trip through UR's WideCharToMultiByte
  conversion and NEO's MultiByteToWideChar decode.

  A cross-API signal/wait round-trip after the main import proves the two
  sides are operating on the *same* kernel object, not that the name
  merely resolved to something.

  Build (SYCLOS on Windows):
  clang++.exe -fsycl  -o vwnts.exe vulkan_win32_named_timeline_semaphore.cpp -lvulkan-1 -I$VULKAN_SDK/Include -L$VULKAN_SDK/Lib
*/
// clang-format on

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include "vulkan_setup.hpp"

#include <cstdio>
#include <iostream>
#include <string>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>
#include <sycl/properties/queue_properties.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

struct NamedTimelineSemaphore {
  VkSemaphore semaphore = VK_NULL_HANDLE;
  std::wstring name;
  // Retain one HANDLE reference so the NT name persists across the test.
  HANDLE keepAliveHandle = nullptr;
};

// Vulkan-side counterpart of the D3D12 test's createNamedExportableFence:
// build a timeline semaphore whose export info registers an NT name.
static NamedTimelineSemaphore
createNamedTimelineSemaphore(VulkanContext &ctx, const wchar_t *name,
                             uint64_t initialValue = 0) {
  NamedTimelineSemaphore res;
  res.name = name;

  VkSemaphoreTypeCreateInfo typeInfo{};
  typeInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
  typeInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  typeInfo.initialValue = initialValue;

  VkExportSemaphoreWin32HandleInfoKHR win32Info{};
  win32Info.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR;
  win32Info.dwAccess = GENERIC_ALL;
  win32Info.name = name;
  win32Info.pNext = &typeInfo;

  VkExportSemaphoreCreateInfo exportInfo{};
  exportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
  exportInfo.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
  exportInfo.pNext = &win32Info;

  VkSemaphoreCreateInfo semaphoreInfo{};
  semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  semaphoreInfo.pNext = &exportInfo;

  VK_CHECK(
      vkCreateSemaphore(ctx.device, &semaphoreInfo, nullptr, &res.semaphore));

  res.keepAliveHandle = getSemaphoreHandle(ctx, res.semaphore);
  return res;
}

// wcout depends on console code page + CRT mode and can fail silently on
// non-Latin-1 codepoints. Print via narrow stdout with \uXXXX escapes.
static void printWideName(const wchar_t *name) {
  for (const wchar_t *p = name; *p; ++p) {
    if (*p >= 0x20 && *p < 0x7F) {
      std::cout << static_cast<char>(*p);
    } else {
      char buf[8];
      std::snprintf(buf, sizeof(buf), "\\u%04X",
                    static_cast<unsigned>(*p) & 0xFFFFu);
      std::cout << buf;
    }
  }
}

int main() {
  std::cout << "Running SYCL Vulkan Timeline Semaphore by Name Test\n";

  VulkanContext vkCtx = createVulkanContext();

  NamedTimelineSemaphore mainSem = createNamedTimelineSemaphore(
      vkCtx, L"Global\\SYCLTestVulkanTimelineName");
  std::cout << "[Vulkan] Created named timeline semaphore: ";
  printWideName(mainSem.name.c_str());
  std::cout << std::endl;

  // Non-ASCII name — exercises UTF-16 -> UTF-8 -> UTF-16 through import.
  // (U+00E9 = 2-byte UTF-8; U+6C14 = 3-byte UTF-8.) Universal character
  // escapes so MSVC source-charset handling can't reinterpret.
  NamedTimelineSemaphore utf16Sem = createNamedTimelineSemaphore(
      vkCtx, L"Global\\SYCLTestVulkanTimelineChiqu\u00E9\u6C14");
  std::cout << "[Vulkan] Created named timeline semaphore: ";
  printWideName(utf16Sem.name.c_str());
  std::cout << std::endl;

  int retCode = 0;

  try {
    sycl::property_list qProps{
        sycl::property::queue::in_order{},
        sycl::ext::intel::property::queue::immediate_command_list{}};
    sycl::queue q{qProps};
    auto device = q.get_device();
    auto context = q.get_context();

    std::cout << "[SYCL] Device: "
              << device.get_info<sycl::info::device::name>() << std::endl;

    std::cout << "[SYCL] Importing timeline semaphore by name\n";
    auto semDesc =
        syclexp::external_semaphore_descriptor<syclexp::resource_win32_name>{
            {(const void *)mainSem.name.c_str()},
            syclexp::external_semaphore_handle_type::timeline_win32_nt_handle};
    syclexp::external_semaphore syclSem =
        syclexp::import_external_semaphore(semDesc, device, context);

    // Import success on a non-ASCII name proves encoding fidelity; no need
    // to signal/wait on it — the main-name path below covers sync mechanics.
    std::cout << "[SYCL] Importing UTF-16 non-ASCII named timeline\n";
    auto utf16SemDesc =
        syclexp::external_semaphore_descriptor<syclexp::resource_win32_name>{
            {(const void *)utf16Sem.name.c_str()},
            syclexp::external_semaphore_handle_type::timeline_win32_nt_handle};
    syclexp::external_semaphore utf16SyclSem =
        syclexp::import_external_semaphore(utf16SemDesc, device, context);
    syclexp::release_external_semaphore(utf16SyclSem, device, context);
    std::cout << "[SYCL] UTF-16 named timeline round-trip OK\n";

    // Cross-API round-trip: prove Vulkan and SYCL see the same underlying
    // object, not just that a name resolves to something on each side.
    std::cout << "[Test] Cross-API signal/wait\n";
    VkSemaphoreSignalInfo signalInfo{};
    signalInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
    signalInfo.semaphore = mainSem.semaphore;
    signalInfo.value = 1;
    VK_CHECK(vkSignalSemaphore(vkCtx.device, &signalInfo));

    q.ext_oneapi_wait_external_semaphore(syclSem, uint64_t{1});
    q.ext_oneapi_signal_external_semaphore(syclSem, uint64_t{2});
    q.wait_and_throw();

    VkSemaphoreWaitInfo waitInfo{};
    waitInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    waitInfo.semaphoreCount = 1;
    waitInfo.pSemaphores = &mainSem.semaphore;
    uint64_t waitValue = 2;
    waitInfo.pValues = &waitValue;
    VkResult r = vkWaitSemaphores(vkCtx.device, &waitInfo, 5'000'000'000ull);
    if (r != VK_SUCCESS) {
      std::cerr
          << "[FAIL] Vulkan did not observe SYCL's timeline signal (result="
          << r << ")" << std::endl;
      retCode = 1;
    } else {
      std::cout << "[Test] Vulkan observed SYCL signal — same object\n";
    }

    syclexp::release_external_semaphore(syclSem, device, context);
    if (retCode == 0)
      std::cout << "SUCCESS" << std::endl;
  } catch (sycl::exception &e) {
    std::cerr << "SYCL Exception: " << e.what() << std::endl;
    retCode = 1;
  }

  vkDestroySemaphore(vkCtx.device, mainSem.semaphore, nullptr);
  vkDestroySemaphore(vkCtx.device, utf16Sem.semaphore, nullptr);
  if (mainSem.keepAliveHandle)
    CloseHandle(mainSem.keepAliveHandle);
  if (utf16Sem.keepAliveHandle)
    CloseHandle(utf16Sem.keepAliveHandle);
  cleanupVulkanContext(vkCtx);
  return retCode;
}
