SYCL-related tests directory.

 - Basic - tests used for sanity testing. Building, executing and checks are defined using insource comments with LIT syntax.
 - External - contains infrastructure for running tests which sources are stored outside of this repository
 - MultiSource - SYCL related tests which depend on multiple source file.
 - SingleSource - SYCL tests with single source file.
 - Parallel - Tests which produce high-parallel load on taret device. It is recommended to run such tests in 1 thread.
