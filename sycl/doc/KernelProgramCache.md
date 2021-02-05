# A brief overview of kernel and program caching mechanism.

## Rationale behind caching

During SYCL program execution SYCL runtime will create internal objects
representing kernels and programs, it may also invoke JIT compiler to bring
kernels in a program to executable state. Those runtime operations are quite
expensive, and in some cases caching approach can be employed to eliminate
redundant kernel or program object re-creation and online recompilation. Few
examples below illustrate scenarios where such optimization is possible.

*Use-case #1.* Submission of the same kernel in a loop:
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

*Use-case #2.* Submission of multiple kernels within a single program<sup>[1](#what-is-program)</sup>:
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

In both cases SYCL runtime will need to build the program and kernels multiple
times, which may involve JIT compilation and take quite a lot of time.

In order to eliminate this waste of run-time we introduce a kernel and program
caching. The cache is per-context and it caches underlying objects of non
interop kernels and programs.

Also JIT compilation for cases when an application contains huge amount of
device code (big kernels or multiple kernels) may take significant time. The
kernels and programs are rebuilt on every program restart. AOT compilation
can be used to avoid that but it ties application to specific backend runtime
version(s) and predefined HW configuration(s). As a general solution it is
reasonable to have kernel/program cache which works between application
restarts (e.g. cache on disk for device code built for specific HW/SW
configuration).

<a name="what-is-program">1</a>: Here "program" means an internal SYCL runtime
object corresponding to a device code module or native binary defining a set of
SYCL kernels and/or device functions.


## Data structure of cache

The cache is split into two levels:
 - in-memory cache which is used during application runtime for device code
 which has been already loaded and built for target device.
 - on-disk cache which is used to store device binaries between application
 executions.

### In-memory cache

The cache stores underlying PI objects behind `cl::sycl::program` and
`cl::sycl::kernel` user-level objects in a per-context data storage. The storage
consists of two maps: one is for programs and the other is for kernels.

The programs map's key consists of four components:
 - kernel set id<sup>[1](#what-is-ksid)</sup>,
 - specialization constants values,
 - device this program is built for,
 - build options id <sup>[2](#what-is-bopts)</sup>.

The kernels map's key consists of two components:
 - program the kernel, belongs to,
 - kernel name<sup>[3](#what-is-kname)</sup>.

<a name="what-is-ksid">1</a>: Kernel set id is an ordinal number of the device
binary image the kernel is contained in.
<a name="what-is-bopts">2</a>: Hash for the string representation of build
options set in application or environment variables. The both compile and link
options are used. There are three sources of build options:
 - from device image (pi_device_binary_struct::CompileOptions,
 pi_device_binary_struct::LinkOptions);
 - environment variables (SYCL_PROGRAM_COMPILE_OPTIONS,
 SYCL_PROGRAM_LINK_OPTIONS);
 - options passed through SYCL API.
<a name="what-is-kname">3</a>: Kernel name is a kernel ID mangled class' name
which is provided to methods of `cl::sycl::handler` (e.g. `parallel_for` or
`single_task`).

### On-disk cache

The cache is hidden behind in-memory cache and stores the same underlying PI
object behind `cl::sycl::program` user-level objects in a per-context data
storage.
The storage is organized as a map for storing device code image. It uses
different keys to address difference in SYCL objects ids between applications
runs as well as the fact that the same kernel name can be used in different
SYCL applications.

The programs map's key consists of four components:
 - device image id<sup>[1](#what-is-diid)</sup>,
 - specialization constants values,
 - device id<sup>[2](#what-is-did)</sup> this program is built for,
 - build options id<sup>[3](#what-is-bopts)</sup>.
<a name="what-is-diid">1</a>: Hash out of  first 10 kB (for performance reasons) of the
device code image used as input for the build.
<a name="what-is-did">2</a>: Hash out of the string which is concatenation of values for
`info::platform::name`, `info::device::name`, `info::device::version`,
`info::device::driver_version` parameters to differentiate different HW and SW
installed on the same host as well as SW/HW upgrades.
<a name="what-is-bopts">3</a>: Hash for the string representation of build
options set in application or environment variables. The both compile and link
options are used. There are three sources of build options:
 - from device image (pi_device_binary_struct::CompileOptions,
 pi_device_binary_struct::LinkOptions);
 - environment variables (SYCL_PROGRAM_COMPILE_OPTIONS,
 SYCL_PROGRAM_LINK_OPTIONS);
 - options passed through SYCL API.

## Environment variables
i
The following environment variables affect cache mechanism:
`DPCPP_CACHE_DIR`=/path/to/cache/location (default values are `%AppData%\Intel\dpcpp_program_cache` for Windows and `$HOME/intel/dpcpp_program_cache`)
`DPCPP_CACHE_ENABLED`=switching on-disc cache ON/OFF (default value is ON to expose possible issue ASAP, may be switch to OFF before the release)
`DPCPP_CACHE_THRESHOLD`=cache clean up threshold (TBD: cleanup strategy)
`DPCPP_CACHE_MIN_DEVICE_IMAGE_SIZE`=min size of device code image which is reasonable
to cache on disk because disk access operation may take more time than do JIT
compilation for it (default - TBD).
`DPCPP_CACHE_MAX_DEVICE_IMAGE_SIZE`=too big kernels may overload disk too fast
(default - TBD)

## Points of improvement (things to do)

 - Implement LRU policy on cached objects. See [issue](https://github.com/intel/llvm/issues/2517).
 - Employ the same built object for multiple devices of the same ISA,
   capabilities and so on. *NOTE:* It's not really known if it's possible to
   check if two distinct devices are *exactly* the same. Probably this should be
   an improvement request for plugins. By now it is assumed that two devices with the same device id <a name="what-is-did">2</a> are the same.
 - Improve testing: cover real use-cases. See currently covered cases [here](https://github.com/intel/llvm/blob/sycl/sycl/unittests/kernel-and-program/Cache.cpp).

## Implementation details

The caches are represented with instance of [`KernelProgramCache`](https://github.com/intel/llvm/blob/sycl/sycl/source/detail/kernel_program_cache.hpp)
class. The runtime creates one instance of the class per distinct SYCL context
(A context object which is a result of copying another context object isn't
"distinct", as it corresponds to the same underlying internal object
representing a context).

The `KernelProgramCache` is essentially a pair of maps as described above.

### When does the cache come at work?

The cache is used when one submits a kernel for execution or builds program with
SYCL API. That means that the cache works when either user explicitly calls
`program::build_with_kernel_type<>()`/`program::get_kernel<>()` methods or SYCL
RT builds a program or gets the required kernel as needed during application
execution. Cacheability of an object can be tested with
`program_impl::is_cacheable()` method. SYCL RT will only try to insert cacheable
programs or kernels into the cache. This is done as a part of
`ProgramManager::getOrCreateKernel()` method.

*NOTE:* a kernel is only cacheable if and only if the program it belongs to is
cacheable. On the other hand if the program is cacheable, then each and every
kernel of this program will be cached also.

All requests to build a program or to create a kernel - whether they originate
from explicit user API calls or from internal SYCL runtime execution logic - end
up with calling the function [`getOrBuild()`](https://github.com/intel/llvm/blob/sycl/sycl/source/detail/program_manager/program_manager.cpp#L149)
with number of lambda functions passed as arguments:
 - Acquire function;
 - GetCache function;
 - Build function.

*Acquire* function returns a locked version of cache. Locking is employed for
thread safety. The threads are blocked only for insert-or-acquire attempt, i.e.
when calling to `map::insert` in [`getOrBuild`](https://github.com/intel/llvm/blob/sycl/sycl/source/detail/program_manager/program_manager.cpp#L149)
function. The rest of operation is done with the help of atomics and condition
variables (plus a mutex for proper work of condition variable).

*GetCache* function returns a reference to mapping `key->value` out of locked
instance of cache. We will see rationale behind it a bit later.

*Build* function actually builds the kernel or program.

Caching isn't done:
 - when program is built out of source with `program::build_with_source()` or
   `program::compile_with_source()` method;
 - when program is a result of linking multiple programs (TBD: if it is
possible to remove this restriction).


### Thread-safety

Why do we need thread safety here? It is quite possible to have a use-case when
the `cl::sycl::context` is shared across multiple threads (e.g. via sharing a
queue). Possibility of enqueueing multiple cacheable kernels simultaneously
from multiple threads requires us to provide thread-safety for the caching
mechanisms.

It is worth of noting that we don't cache the PI resource (kernel or program)
by itself. Instead we augment the resource with the status of build process.
Hence, what is cached is a wrapper structure `BuildResult` which contains three
information fields - pointer to built resource, build error (if applicable) and
current build status (either of "in progress", "succeeded", "failed").

One can find definition of `BuildResult` template in [KernelProgramCache](https://github.com/intel/llvm/blob/sycl/sycl/source/detail/kernel_program_cache.hpp).

The built resource access synchronization approach aims at minimizing the time
any thread holds the global lock guarding the maps to improve performance. To
achieve that, the global lock is acquired only for the duration of the global
map access. Actual build of the program happens outside of the lock, so other
threads can request or build other programs in the meantime. A thread requesting
a `BuildResult` instance via `getOrBuild` can go one of three ways:
 A) Build result is **not** available, it is the first thread to request it.
    Current thread will then execute the build letting others wait for the
    result using the per-build result condition variable kept in `BuildResult`'s
    `MBuildCV` field.
 B) Build result is **not** available, another thread is already building the
    result. Current thread will then wait for the result using the `MBuildCV`
    condition variable.
 C) Build result **is** available. The thread simply takes it from the `Ptr`
    field w/o using any mutexes or condition variables.

As noted before, access to `BuildResult` instance fields may occur from
different threads simultaneously, but the global lock is no longer held. So, to
make it safe and to make sure only one thread builds the requested program, the
following is done:
 - program build state is reflected in the `State` field, threads use
   compare-and-swap technique to compete who will do the build and become thread
   A. Threads C will find 'DONE' in this field and immediately return the with
   built result at hand.
 - thread A and thread(s) B use the `MBuildCV` conditional variable field and
   `MBuildResultMutex`  mutex field guarding that variable to implement the
   "single producer-multiple consumers scheme".
 - the build result itself appears in the 'Ptr' field when available.
All fields are atomic because they can be accessed from multiple threads.

A specialization of helper class [Locked](https://github.com/intel/llvm/blob/sycl/sycl/include/CL/sycl/detail/locked.hpp)
for reference of proper mapping is returned by Acquire function. The use of this
class implements RAII to make code look cleaner a bit. Now, GetCache function
will return the mapping to be employed that includes the 3 components: kernel
name, device as well as any specialization constants values. These get added to
`BuildResult` and are cached. The `BuildResult` structure is specialized with
either `PiKernel` or `PiProgram`<sup>[1](#remove-pointer)</sup>.

### Inter-process safety

For on-disc cache there might be access collisions for accessing the same file
from different instance of SYCL applications:
- write collision happens when 2 instances of the same application are started
to write to the same cache file;
- read collision may happen if one application is writing to the file and the
other instance of the application is trying to read from it while write
operation is not finished.
To avoid collisions the file system entries are locked for read-write access
until write operation is finished. e.g if new file or directory should be
created/deleted parent directory is locked, file is created in locked state,
then the directory and the file are unlocked.
Advisory locking <sup>[2](#advisory-lock)</sup> is used to ensure that the
user/OS tools are able to manage files.

### Hash function

STL hash function specialized for std::string is going to be used:
`template<>  struct hash<std::string>`
TBD: may be it is reasonable to use own implementation for hash function to
avoid issues when different  C++ library implementation is used producing
different hashes. But at first look it is not an issue because it may happen
only DPC++ compiler/runtime change and that means change of device code image.

### Core of caching mechanism

Now, let us see how 'getOrBuild' function works.
First, we fetch the cache with sequential calls to Acquire and GetCache
functions. Then, we check if this is the first attempt to build this kernel or
program. This is achieved with an attempt to insert another key-value pair into
the map. At this point we try to insert `BuildResult` stub instance with status
equal to "in progress" which will allow other threads to know that someone is
(i.e. we're) building the object (i.e. kernel or program) now. If insertion
fails, we will wait for building thread to finish with call to `waitUntilBuilt`
function. This function will throw stored exception<sup>[3](#exception-data)</sup>
upon build failure. This allows waiting threads to see the same result as the
building thread. Special case of the failure is when build result doesn't
contain the error (i.e. the error wasn't of `cl::sycl::exception` type) and the
pointer to object in `BuildResult` instance is nil. In this case, the building
thread has finished the build process and has returned an error to the user.
But this error may be sporadic in nature and may be spurious. Hence, the waiting
thread will try to build the same object once more.

`BuildResult` structure also contains synchronization objects: mutex and
condition variable. We employ them to signal waiting threads that the build
process for this kernel/program is finished (either successfully or with a
failure).

<a name="remove-pointer">1</a>: The use of `std::remove_pointer` was omitted for
the sake of simplicity here.
<a name="advisory-lock">2</a> Advisory locks work only when a process
explicitly acquires and releases locks, and are ignored if a process is not aware
of locks.
<a name="exception-data">3</a>: Actually, we store contents of the exception:
its message and error code.

### On-disk cache storage structure

The device code image are stored on filesystem using structure below:
<cache_root>/<device_id>/<device_image_id>_<spec_constants_values>_<build_options>.bin

### Cache cleanup

Cache cleanup mechanism is required to avoid resources overfloat both for
memory and disk. The general idea is to delete items following LRU (least
recently used) startegy.

#### In-memory cache cleanup

It is initiated on program/kernel maps access/add item operation. The items
exceeding storage threshold are deleted.

#### On-disk cache cleanup mechanism

On-disk cache is going to be purged basing on file last access (read/write) date
(access time). On SYCL application shutdown phase cache clean up process is
initiated which walk through cache directories and check:
 - if the file is locked, go to the next file;
 - otherwise check file access time:
   - if file access time is above threshold, delete the file and remove parent
   directory while they are empty;
   - otherwise do nothing.
