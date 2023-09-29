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