# Basic tests

In order to capture the cost of various API calls in the framework and test the
correctness of the API, a set of basic tests have been created. They primarily
fall under two categories:

1. Semantic tests: These tests perform correctness checks on the API call to
ensure the right data is being retrieved. The semantic tests are categorized
into string table tests, trace point tests and notification tests.

2. Performance tests: These test attempt to capture the average cost of various
operations that are a part of creating trace points in applications. The tests
are categorized into data structure tests and instrumentation tests.

For more detail on the framework, the tests that are provided and their usage,
please consult the [XPTI Framework library documentation](doc/XPTI_Framework.md).
