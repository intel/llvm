# Overview

This extension enables detection of assert failure of kernel.

# New enum value

`ze_result_t` enumeration should be augmented with `ZE_RESULT_ASSERT_FAILED`
enum element. This enum value indicated a detected assert failure at
device-side.

# Changed API

```
ze_event_handle_t Event; // describes an event of kernel been submitted previously
ze_result Result = zeEventQueryStatus(Event);
```

If kernel failed an assertion `zeEventQueryStatus` should return
`ZE_RESULT_ASSERT_FAILED`.

