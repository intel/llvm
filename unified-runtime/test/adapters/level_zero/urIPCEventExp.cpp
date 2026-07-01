// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// REQUIRES: v2
// RUN: %with-v2 ./ipc_event-test

#include "ipc_event_fixtures.h"

using urIPCEventExpTest = uur::event::urIPCEventTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE(urIPCEventExpTest);

// Get returns a non-null buffer and a non-zero size.
TEST_P(urIPCEventExpTest, GetReturnsBuffer) {
  void *data = nullptr;
  size_t size = 0;
  ASSERT_SUCCESS(urIPCGetEventHandleExp(event, &data, &size));
  ASSERT_NE(data, nullptr);
  ASSERT_GT(size, 0u);
  EXPECT_SUCCESS(urIPCPutEventHandleExp(context, data));
}

// Producer Get -> consumer Open in the same process succeeds and yields a
// non-null event.
TEST_P(urIPCEventExpTest, OpenSucceedsSameProcess) {
  void *data = nullptr;
  size_t size = 0;
  ASSERT_SUCCESS(urIPCGetEventHandleExp(event, &data, &size));

  ur_event_handle_t imported = nullptr;
  ASSERT_SUCCESS(urIPCOpenEventHandleExp(context, data, size, &imported));
  ASSERT_NE(imported, nullptr);

  EXPECT_SUCCESS(urEventRelease(imported));
  EXPECT_SUCCESS(urIPCPutEventHandleExp(context, data));
}

// The producer event was signaled in SetUp, so a wait on the imported event
// must complete.
TEST_P(urIPCEventExpTest, WaitOnImportedSucceeds) {
  void *data = nullptr;
  size_t size = 0;
  ASSERT_SUCCESS(urIPCGetEventHandleExp(event, &data, &size));

  ur_event_handle_t imported = nullptr;
  ASSERT_SUCCESS(urIPCOpenEventHandleExp(context, data, size, &imported));

  EXPECT_SUCCESS(urEventWait(1, &imported));

  EXPECT_SUCCESS(urEventRelease(imported));
  EXPECT_SUCCESS(urIPCPutEventHandleExp(context, data));
}

// The imported event is a normal ur_event_handle_t and participates in the
// usual reference counting.
TEST_P(urIPCEventExpTest, ImportedEventRetainRelease) {
  void *data = nullptr;
  size_t size = 0;
  ASSERT_SUCCESS(urIPCGetEventHandleExp(event, &data, &size));

  ur_event_handle_t imported = nullptr;
  ASSERT_SUCCESS(urIPCOpenEventHandleExp(context, data, size, &imported));

  ASSERT_SUCCESS(urEventRetain(imported));
  EXPECT_SUCCESS(urEventRelease(imported));
  EXPECT_SUCCESS(urEventRelease(imported));

  EXPECT_SUCCESS(urIPCPutEventHandleExp(context, data));
}

// Multiple Get/Open/Release/Put round trips on the same producer event each
// yield a fresh non-null imported event.
TEST_P(urIPCEventExpTest, MultipleRoundTrips) {
  for (int i = 0; i < 3; ++i) {
    void *data = nullptr;
    size_t size = 0;
    ASSERT_SUCCESS(urIPCGetEventHandleExp(event, &data, &size));

    ur_event_handle_t imported = nullptr;
    ASSERT_SUCCESS(urIPCOpenEventHandleExp(context, data, size, &imported));
    ASSERT_NE(imported, nullptr);

    EXPECT_SUCCESS(urEventRelease(imported));
    EXPECT_SUCCESS(urIPCPutEventHandleExp(context, data));
  }
}

// The same handle can be opened more than once; each open yields an event that
// shares state with the producer (so each completes a wait).
TEST_P(urIPCEventExpTest, MultipleOpensOfSameHandle) {
  void *data = nullptr;
  size_t size = 0;
  ASSERT_SUCCESS(urIPCGetEventHandleExp(event, &data, &size));

  ur_event_handle_t first = nullptr;
  ur_event_handle_t second = nullptr;
  ASSERT_SUCCESS(urIPCOpenEventHandleExp(context, data, size, &first));
  ASSERT_SUCCESS(urIPCOpenEventHandleExp(context, data, size, &second));
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  ASSERT_NE(first, second);

  EXPECT_SUCCESS(urEventWait(1, &first));
  EXPECT_SUCCESS(urEventWait(1, &second));

  EXPECT_SUCCESS(urEventRelease(first));
  EXPECT_SUCCESS(urEventRelease(second));
  EXPECT_SUCCESS(urIPCPutEventHandleExp(context, data));
}

// Releasing the imported event before Put on the handle is fine, and Put does
// not affect the still-alive producer event.
TEST_P(urIPCEventExpTest, PutLeavesProducerAlive) {
  void *data = nullptr;
  size_t size = 0;
  ASSERT_SUCCESS(urIPCGetEventHandleExp(event, &data, &size));

  ur_event_handle_t imported = nullptr;
  ASSERT_SUCCESS(urIPCOpenEventHandleExp(context, data, size, &imported));
  EXPECT_SUCCESS(urEventRelease(imported));
  EXPECT_SUCCESS(urIPCPutEventHandleExp(context, data));

  // The producer event (released by the fixture TearDown) is still usable.
  EXPECT_SUCCESS(urEventWait(1, &event));
}
