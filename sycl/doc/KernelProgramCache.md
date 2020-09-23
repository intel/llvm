# A brief overview of kernel/program caching mechanism.

## Rationale behind caching

*Use-case #1.* Looped enqueue of the same kernel:
```C++
  using namespace cl::sycl::queue;

  queue Q;
  std::vector<buffer> Bufs;

  ...
  // initialize Bufs with some number of buffers
  ...

  for (size_t Idx = 0; Idx < Bufs.size(); ++Idx) {
    Q.submit([&](handler &CGH) {
      auto Acc = Bufs[Idx].get_access<access::mode::read_write>(CGH);

      CGH.parallel_for<class TheKernel>(
          range<2>{N, M}, [=](item<2> Item) { ... });
    });
  }
```

*Use-case #2.* Enqueue of multiple kernels within a single program<sup>[1](#what-is-program)</sup>:
```C++
  using namespace cl::sycl::queue;

  queue Q;

  Q.submit([&](handler &CGH) {
    ...

    CGH.parallel_for<class TheKernel_1>(
        range<2>{N_1, M_1}, [=](item<2> Item) { ... });
  });
  Q.submit([&](handler &CGH) {
    ...

    CGH.parallel_for<class TheKernel_2>(
        range<2>{N_2, M_2}, [=](item<2> Item) { ... });
  });
  Q.submit([&](handler &CGH) {
    ...

    CGH.parallel_for<class TheKernel_3>(
        range<2>{N_3, M_3}, [=](item<2> Item) { ... });
  });
  ...
  Q.submit([&](handler &CGH) {
    ...

    CGH.parallel_for<class TheKernel_K>(
        range<2>{N_K, M_K}, [=](item<2> Item) { ... });
  });
```

Both these use-cases will need to built the program or kernel multiple times.
When JIT is employed this process may take quite a lot of time.

In order to eliminate this waste of run-time we introduce a kernel/program
caching. The cache is per-context and it caches underlying objects of non
interop kernels and programs which are built with no options.

<a name="what-is-program">1</a>: Here we use the term "program" in the same
sense as OpenCL does i.e. a set of kernels.


## Data structure of cache

The cache stores underlying PI objects of `cl::sycl::program` and
`cl::sycl::kernel` in a per-context data storage. The storage consists of two
maps: one is for programs and the other is for kernels.

Programs mapping's key consists of three components:
kernel set id<sup>[1](#what-is-ksid)</sup>, specialized constants, device this
program is built for.

Kernels mapping's key consists of three components too: program the kernel
belongs to, kernel name<sup>[2](#what-is-kname)</sup>, device the program is
built for.

<a name="what-is-ksid">1</a>: Kernel set id is merely a number of translation
unit which contains at least one kernel.
<a name="what-is-kname">2</a>: Kernel name is mangled class name which is
provided to methods of `cl::sycl::handler` (e.g. `parallel_for` or
`single_task`).


## Points of improvement (things to do)

 - Implement LRU policy on cached objects. See [issue](https://github.com/intel/llvm/issues/2517).
 - Allow for caching of objects built with some build options.
 - Employ the same built object for multiple devices of the same ISA,
   capabilities and so on. *NOTE:* It's not really known if it's possible to
   check if two distinct devices are *exactly* the same.
 - Improve testing: cover real use-cases. See currently covered cases [here](https://github.com/intel/llvm/blob/sycl/sycl/unittests/kernel-and-program/Cache.cpp).


## Implementation details

The caches are represented with instance of [`KernelProgramCache`](https://github.com/intel/llvm/blob/sycl/sycl/source/detail/kernel_program_cache.hpp)
class. The class is instantiated in a per-context manner.

The `KernelProgramCache` is the storage descrived above.


### When does the cache come at work?

The cache is employed when one submits kernel for execution or builds program or
kernel with SYCL API. That means that the cache works when either user
explicitly calls `program::build_with_kernel_type<>()`/`program::get_kernel<>()`
methods or SYCL RT builds or gets the required kernel. Cacheability of an object
is verified with `program_impl::is_cacheable()` method. SYCL RT will check if
program is cacheable and will get the kernel with call to
`ProgramManager::getOrCreateKernel()` method.


*NOTE:* a kernel is only cacheable if and only if the program it belongs to is
cacheable. On the other hand if the program is cacheable, then each and every
kernel of this program will be cached also.


Invoked by user `program::build_with_kernel_type<>()` and
`program::get_kernel<>()` methods will call either
`ProgramManager::getBuildPIProgram()` or `ProgramManager::getOrCreateKernel()`
method respectively. Now, both these methods will call template
function [`getOrBuild()`](../source/detail/program_manager/program_manager.cpp#L149)
with multiple lambdas passed to it:
 - Acquire function;
 - GetCache function;
 - Build function.

*Acquire* function returns a locked version of cache. Locking is employed for
thread safety. The threads are blocked only for insert-or-acquire attempt, i.e.
when calling to `map::insert` in [`getOrBuild`](../source/detail/program_manager/program_manager.cpp#L149)
function. The rest of operation is done with the help of atomics and condition
variables (plus a mutex for proper work of condition variable).

*GetCache* function returns a reference to mapping `key->value` out of locked
instance of cache. We will see rationale behind it a bit later.

*Build* function actually builds the kernel or program.

Caching isn't done:
 - when program is built out of source i.e. with
   `program::build_with_source()` or `program::compile_with_source()` method;
 - when program is result of linking of multiple programs.


### Thread-safety

Why do we need thread safety here? It's quite possible to have a use-case when
the `cl::sycl::context` is shared across multiple threads (e.g. via sharing a
queue). Possibility of enqueueing multiple cacheable kernels simultaneously
within multiple threads makes us to provide thread-safety for the cache.

It's worth of noting that we don't cache the PI resource (kernel or program)
on it's own. Instead we augment the resource with the status of build process.
Hence, what is cached is a wrapper structure `BuildResult` which contains three
information fields - pointer to built resource, build error (if applicable) and
current build status (either of "in progress", "succeeded", "failed").

One can find definition of `BuildResult` template in [KernelProgramCache](https://github.com/intel/llvm/blob/sycl/sycl/source/detail/kernel_program_cache.hpp).

Pointer to built resource and build result are both atomic variables. Atomicity
of these variables allows one to hold lock on cache for quite a short time and
perform the rest of build/wait process without unwanted need of other threads to
wait on lock availability.

A specialization of helper class [Locked](https://github.com/intel/llvm/blob/sycl/sycl/include/CL/sycl/detail/locked.hpp)
for reference of proper mapping is returned by Acquire function. The use of this
class implements RAII to make code look cleaner a bit. Now, GetCache function
will return the mapping to be employed i.e. it'll fetch mapping of kernel name
plus device to `BuildResult` for proper program as `getOrBuild` will work with
mapping of key (whichever it is) to `BuildResult` specialization. The structure
is specialized with either `PiKernel` or `PiProgram`<sup>[1](#remove-program)</sup>.


### Core of caching mechanism

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


<a name="remove-pointer">1</a>: The use of `std::remove_pointer` was omitted in
sake of simplicity here.

<a name="exception-data">2</a>: Actually, we store contents of the exception:
its message and error code.

