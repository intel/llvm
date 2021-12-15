# ITT Mapping collector

The ITT mapping collector demonstrates the creation of a subscriber that map
XPTI calls to ITT API so tools that rely on ITT to build their trace database
can still consume XPTI traces without having to write an XPTI collector.

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
  case is `libitt_mapper.[so,dll,dylib]`.

     `XPTI_SUBSCRIBERS=/path/to/libitt_mapper.[so,dll,dylib]`

For more detail on the framework, the tests that are provided and their usage,
please consult the [XPTI Framework library documentation](doc/XPTI_Framework.md).

> **NOTE:** The ITT API supported by this sample is a subset of the ITT API
> specification that is necessary to map the XPTI API calls. An ITT library
> must be created as a shared object to allow other tools to replace the dummy
> implementation in this sample with real implementations that tools have.