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