# Conformance tests

At this point, conformance tests include matches for individual adapters
that allow you to ignore errors from the listed tests.
This solution allows regular execution of cts tests on GHA
and prevents further errors.
In the future, when all bugs are fixed, and the tests pass,
this solution will no longer be necessary.
When you fix any test, the match file must be updated
Empty match files indicate that there are no failing tests
in a particular group for the corresponding adapter.

## How to set test device/platform name or limit the test devices/platforms count

To limit how many devices/platforms you want to run the conformance and
adapters tests on, use CMake option UR_TEST_DEVICES_COUNT or
UR_TEST_PLATFORMS_COUNT. If you want to run the tests on
all available devices/platforms, set 0. The default value is 1.
If you run binaries for the tests, you can use the parameter
`--platforms_count=COUNT` or `--devices_count=COUNT`.
To set test device/platform name you want to run the tests on, use
parameter `--platform=NAME` or `--device=NAME`.

## Known failures

The `UUR_KNOWN_FAILURE_ON` macro can be used to skip tests on devices where the
test is known to fail. This can be done in the following situations.

For all devices in an adapter:

```cpp
UUR_KNOWN_FAILURE_ON(uur::LevelZero{});
```

By substring match of the device name within and adapter:

```cpp
UUR_KNOWN_FAILURE_ON(uur::OpenCL{"Intel(R) UHD Graphics 770"});
```

In certain test suits, where there is no access to a device, the platform name
is used instead:

```cpp
UUR_KNOWN_FAILURE_ON(uur::CUDA{"NVIDIA CUDA BACKEND"});
```

When neither device or platform is available in a test suite, the name is
ignored and only the adapter backend is used to determine if the test is a
known failure.

The macro is variadic making it possible to specify known failures for multiple
adapters in a single place and multiple names can also be provided per adapter:

```cpp
UUR_KNOWN_FAILURE_ON(
uur::OpenCL{
    "Intel(R) UHD Graphics 750",
    "Intel(R) UHD Graphics 770",
},
uur::HIP{"Radeon RX 7700"},
uur::NativeCPU{});
```

The following adapter matcher objects are available:

* `uur::OpenCL`
* `uur::LevelZero`
* `uur::LevelZeroV2`
* `uur::CUDA`
* `uur::HIP`
* `uur::NativeCPU`

## Writing a new CTS test

If you're writing a new CTS test you'll need to make use of the existing test
fixtures and instantiation macros to access the adapters available for testing.
The definitions for these can all be found in
[fixtures.h](https://github.com/oneapi-src/unified-runtime/blob/main/test/conformance/testing/include/uur/fixtures.h).

There are five "base" fixtures in that header that each correspond to an
instantiation macro - specific macros are needed due to how gtest handles
parameterization and printing. All of these make use of gtest's [value
parameterization](http://google.github.io/googletest/advanced.html#how-to-write-value-parameterized-tests)
to instantiate the tests for all the available backends. Two of the five base
fixtures have a wrapper to allow for tests defining their own parameterization.

The base fixtures and their macros are detailed below. Tests inheriting from
fixtures other than the base ones (`urContextTest`, etc.) must be instantiated
with the macro that corresponds to whatever base class is ultimately being
inherited from. In the case of `urContextTest` we can see that it inherits
directly from `urDeviceTest`, so we should use the `urDeviceTest` macro. For
other fixtures you'll need to follow the inheritance back a few steps to figure
it out.

### `urAdapterTest`

This fixture is intended for tests that will be run once for each adapter
available. The only data member it provides is a `ur_adapter_handle_t`.

Instantiated with the `UUR_INSTANTIATE_ADAPTER_TEST_SUITE` macro.

### `urPlatformTest`

This fixture is intended for tests that will be run once for each platform
available (note the potentially one-to-many relationship between adapters and
platforms). The only data member it provides is a `ur_platform_handle_t`.

Instantiated with the `UUR_INSTANTIATE_PLATFORM_TEST_SUITE` macro.

### `urDeviceTest`

This fixture is intended for tests that will be run once for each device
available. It provides a `ur_adapter_handle_t`, `ur_platform_handle_t` and a
`ur_device_handle_t` (the former two corresponding to the latter).

Instantiated with the `UUR_INSTANTIATE_DEVICE_TEST_SUITE` macro.

### `urPlatformTestWithParam`

This fixture functions the same as `urPlatformTest` except it allows value
parameterization via a template parameter. Note the parameter you specify is
accessed with `getParam()` rather than gtest's `GetParam()`. A platform handle
is provided the same way it is in the non-parameterized variant.

Instantiated with the `UUR_PLATFORM_TEST_SUITE_WITH_PARAM` macro, which, in
addition to the fixture, takes a `::testing::Values` list of parameter values
and a printer (more detail about that below).

### `urDeviceTestWithParam`

As with the parameterized platform fixture this functions identically to
`urDeviceTest` with the addition of the template parameter for
parameterization.

Instantiated with the `UUR_DEVICE_TEST_SUITE_WITH_PARAM` macro.

### Parameter printers

When instantiating tests based on the parameterized fixtures you need to
provide a printer function along with the value list. This determines how the
parameter values are incorporated into the test name. If your parameter type
has a `<<` operator defined for it, you can simply use the
`platformTestWithParamPrinter`/`deviceTestWithParamPrinter` helper functions for
this.

If you find yourself needing to write a custom printer function, bear in mind
that due to the parameterization wrapper it'll need to deal with a tuple
containing the platform/device information and your parameter. You should
reference the implementations of `platformTestWithParamPrinter` and
`deviceTestWithParamPrinter` to see how this is handled.
