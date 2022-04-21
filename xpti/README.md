# XPTI proxy library

Implementation of the instrumentation stub library to support SYCL
Instrumentation. The stub library checks for two things before it can
successfully dispatch event streams:

1. Environment variable that indicates that tracing has been enabled.

   This is defined by the variable `XPTI_TRACE_ENABLE`. The possible
   values taken by this environment variable are:

   To enable: `XPTI_TRACE_ENABLE=1` or `XPTI_TRACE_ENABLE=true`

   To disable: `XPTI_TRACE_ENABLE=0` or `XPTI_TRACE_ENABLE=false`

2. Environment variable that points to the XPTI dispatcher so the stub
   library can dynamically load it and dispatch the calls to the dispatcher.
   `XPTI_FRAMEWORK_DISPATCHER=/path/to/dispatcher.[so,dll,dylib]`

The stub library requires both of these to be set for it to successfully
dispatch the calls for the event streams. The dispatcher is required for
tool developers to implement subscribers and register them with the dispatcher.
