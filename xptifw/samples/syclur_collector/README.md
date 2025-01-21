# Example SYCL UR Layer Collector

The SYCL UR layer collector demonstrates the creation of a subscriber and prints
of the data received from SYCL UR layer stream. In order to obtain the data from
an application instrumented with XPTI, the following steps must be performed.

1. Set the environment variable that indicates that tracing has been enabled.

   This is defined by the variable `XPTI_TRACE_ENABLE`. The possible
   values taken by this environment variable are:

   To enable: `XPTI_TRACE_ENABLE=1` or `XPTI_TRACE_ENABLE=true`

   To disable: `XPTI_TRACE_ENABLE=0` or `XPTI_TRACE_ENABLE=false`

2. Set the environment variable that points to the XPTI framework dispatcher so
   the stub library can dynamically load it and dispatch the calls to the
   dispatcher.
   `XPTI_FRAMEWORK_DISPATCHER=/path/to/libxptifw.[so,dll,dylib]`

3. Set the environment variable that points to the subscriber, which in this
  case is `libsyclur_collector.[so,dll,dylib]`.

     `XPTI_SUBSCRIBERS=/path/to/libxpti_syclur_collector.[so,dll,dylib]`
For more detail on the framework, the tests that are provided and their usage,
please consult the [XPTI Framework library documentation](doc/XPTI_Framework.md).
