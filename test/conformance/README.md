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

## How to limit the test devices count

To limit how many devices you want to run the CTS on,
use CMake option UR_TEST_DEVICES_COUNT. If you want to run
the tests on all available devices, set 0.
The default value is 1.