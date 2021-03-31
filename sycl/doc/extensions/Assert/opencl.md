# Overview

This extension enables detection of assert failure of kernel.

# New error code

`CL_ASSERT_FAILURE` is added to indicate a detected assert failure at
device-side.

# Changed API

```
cl_event Event; // describes an event of kernel been submitted previously
cl_int Result;
size_t ResultSize;

clGetEventInfo(Event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(Result), &Result, &ResultSize);
```

If kernel failed an assertion `clGetEventInfo` should put `CL_ASSERT_FAILURE`
in `Result`.

