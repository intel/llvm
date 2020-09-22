# A brief overview of kernel/program caching mechanism.

The cache is employed when one submits kernel for execution or builds program or
kernel with SYCL API. At the same time programs and kernels are cached only when
they're built from C++ source, i.e. `program::build_with_kernel_type<>()` and
`program::get_kernel<>()` methods are employed. This restriction is implemented
via use of `program_impl::is_cacheable_with_options()` and
`program_impl::is_cacheable()` methods. The latter method only returns a boolean
flag which is set to false on default and is set to true in a single use-case.
One can find use-cases and cache filling in the [unit-tests](https://github.com/intel/llvm/blob/sycl/sycl/unittests/kernel-and-program/Cache.cpp).

How does it work, i.e. at which point is the cache employed? At some point of
`ExecCGCommand`'s enqueue process the program manager's method will be called:
either `ProgramManager::getBuildPIProgram` or
`ProgramManager::getOrCreateKernel`. Now, both these methods will call template
function [`getOrBuild`](../source/detail/program_manager/program_manager.cpp#L149)
with multiple lambdas passed to it:
 - Acquire function;
 - GetCache function;
 - Build function.

Acquire function returns a locked version of cache. Locking is employed for
thread safety. The threads are blocked only for insert-or-acquire attempt, i.e.
when calling to `map::insert` in [`getOrBuild`](../source/detail/program_manager/program_manager.cpp#L149)
function. The rest of operation is done with the help of atomics and condition
variables (plus a mutex for proper work of condition variable).

GetCache function returns a reference to mapping `key->value` out of locked
instance of cache. We will see rationale behind it a bit later.

Build function actually builds the kernel or program.

When we say "cache" we think about mapping of some key to value. These maps are
contained within [KernelProgramCache](https://github.com/intel/llvm/blob/sycl/sycl/source/detail/kernel_program_cache.hpp) 
class instance which on its own belongs to `context_impl` class instance.
Kernel cache is per program mapping of kernel name plus device pair to
`BuildResult<PiKernel>`<sup>[1](#remove-pointer)</sup>. When `getOrBuild`
function is called the key for kernel cache is pair/tuple of kernel name and
device. Program cache maps triple (spec consts, kernel set id, device) to
`BuildResult<PiProgram>`<sup>[1](#remove-pointer)</sup>.

Now, we have a helper [Locked](https://github.com/intel/llvm/blob/sycl/sycl/include/CL/sycl/detail/locked.hpp)
class. It's to use RAII to make code look cleaner a bit. Acquire function/lambda
will return a specialization of Locked class for reference of proper mapping.
Now, GetCache function will return the mapping to be employed i.e. it'll fetch
mapping of kernel name plus device to `BuildResult` for proper program as
`getOrBuild` will work with mapping of key (whichever it is) to `BuildResult`
specialization.

`BuildResult` structure contains three information fields - pointer to built
kernel/program, build error (if applicable) and current build status
(either of "in progress", "succeeded", "failed").

Now, how `getOrBuild` works?
First, we fetch the cache with sequential calls to Acquire and GetCache
functions. Then, we check if we're the first ones who build this kernel/program.
This is achieved with attempt to insert another key-value pair into the map.
At this point we try to insert `BuildResult` stub instance with status equal to
"in progress" which will allow other threads to know that someone is (i.e.
we're) building the object (i.e. kernel or program) now. If insertion fails we
will wait for building thread to finish with call to `waitUntilBuilt` function.
This function will throw stored exception<sup>[2](#exception-data)</sup> upon
build failure. This allows waiting threads to result the same as the building
thread. Special case of the failure is when build result doesn't contain the
error (i.e. the error wasn't of `cl::sycl::exception` type) and the pointer to
object in `BuildResult` instance is nil. In this case the building thread has
finished build process and returned the user an error. Though, this error could
be of spurious/sporadic nature. Hence, the waiting thread will try to build the
same object once more.

`BuildResult` structure also contains synchronization objects: mutex and
condition variable. We employ them to signal waiting threads that the build
process for this kernl/program is finished (either successfuly or with a
failure).


<a name="remove-pointer">1</a>: The use of `std::remove_pointer` was omitted in sake of
simplicity here.

<a name="exception-data">2</a>: Actually, we store contents of the exception: its message and
error code.

