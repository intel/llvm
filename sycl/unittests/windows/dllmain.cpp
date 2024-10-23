//==----- dllmain.cpp --- verify behaviour of lib on process termination ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/*
 * This test calls DllMain on Windows. This means, the process performs actions
 * which are required for library unload. That said, the test requires to be a
 * distinct binary executable.
 */

#include <helpers/UrMock.hpp>
#include <sycl/sycl.hpp>

#include <gtest/gtest.h>

#ifdef _WIN32
#include <windows.h>

extern "C" BOOL WINAPI DllMain(HINSTANCE hinstDLL, DWORD fdwReason,
                               LPVOID lpReserved);

static std::atomic<int> TearDownCalls{0};

// Before the port this was an override for LoaderTearDown, UR's mock
// functionality can't override loader functions but AdapterRelease is called
// in the runtime in the same place as LoaderTearDown
ur_result_t redefinedAdapterRelease(void *) {
  fprintf(stderr, "intercepted tear down\n");
  ++TearDownCalls;

  return UR_RESULT_SUCCESS;
}
#endif

TEST(Windows, DllMainCall) {
#ifdef _WIN32
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();
  mock::getCallbacks().set_before_callback("urAdapterRelease",
                                           &redefinedAdapterRelease);

  // Teardown calls are only expected on sycl.dll library unload, not when
  // process gets terminated.
  // The first call to DllMain is to simulate library unload. The second one
  // is to simulate process termination
  fprintf(stderr, "Call DllMain for the first time\n");
  DllMain((HINSTANCE)0, DLL_PROCESS_DETACH, (LPVOID)NULL);

  int TearDownCallsDone = TearDownCalls.load();

  EXPECT_NE(TearDownCallsDone, 0);

  fprintf(stderr, "Call DllMain for the second time\n");
  DllMain((HINSTANCE)0, DLL_PROCESS_DETACH, (LPVOID)0x01);

  EXPECT_EQ(TearDownCalls.load(), TearDownCallsDone);
#endif
}
