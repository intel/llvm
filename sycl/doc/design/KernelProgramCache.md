# A brief overview of kernel and program caching mechanism

## Rationale behind caching

During SYCL program execution, the SYCL runtime will create internal objects
representing kernels and programs, it may also invoke JIT compiler to bring
kernels in a program to executable state. Those runtime operations are quite
expensive, and in some cases caching approach can be employed to eliminate
redundant kernel or program object re-creation and online recompilation. Few
examples below illustrate scenarios where such optimization is possible.

*Use-case #1.* Submission of the same kernel in a loop:

```C++
  using namespace sycl;

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
  using namespace sycl;

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

In both cases, the SYCL runtime will need to build the program and kernels multiple
times, which may involve JIT compilation and take quite a lot of time.

In order to eliminate this waste of run-time we introduce a kernel and program
caching. The cache is per-context and it caches underlying objects of non
interop kernels and programs.

*Use-case #3.* Rebuild of all programs on SYCL application restart:
JIT compilation for cases when an application contains huge amount of device
code (big kernels or multiple kernels) may take significant time. The kernels
and programs are rebuilt on every program restart. AOT compilation can be used
to avoid that but it ties application to specific backend runtime version(s) and
predefined HW configuration(s). As a general solution it is reasonable to have
program persistent cache which works between application restarts (e.g. cache
on disk for device code built for specific HW/SW configuration).

(what-is-program)=
<a name="what-is-program">1</a>: Here "program" means an internal SYCL runtime
object corresponding to a device code module or native binary defining a set of
SYCL kernels and/or device functions.

## Data structure of cache

The cache is split into two levels:

- in-memory cache which is used during application runtime for device code
  which has been already loaded and built for target device.
- persistent (on-disk) cache which is used to store device binaries between
  application executions.

### In-memory cache

The cache stores the underlying UR objects behind `sycl::program` and `sycl::kernel`
user-level objects in a per-context data storage. The storage consists of three 
maps: one is for programs and the other two are for kernels.

The programs map's key consists of four components:

- ID of the device image containing the program,
- specialization constants values,
- the set of devices this program is built for.

The kernels map's key consists of two components:

- the program the kernel belongs to,
- kernel name<sup>[3](#what-is-kname)</sup>.

The third map, called Fast Kernel Cache, is used as an optimization to reduce the
number of lookups in the kernels map. Its key consists of the following components:

- specialization constants values,
- the UR handle of the device this kernel is built for,
- kernel name<sup>[3](#what-is-kname)</sup>.

(what-is-kname)=
<a name="what-is-kname">3</a>: Kernel name is a kernel ID mangled class' name
which is provided to methods of `sycl::handler` (e.g. `parallel_for` or
`single_task`).

### Persistent cache

The cache works behind in-memory cache and stores the same underlying UR
object behind `sycl::program` user-level objects in a per-context data storage.
The storage is organized as a map for storing device code image. It uses
different keys to address difference in SYCL objects ids between applications
runs as well as the fact that the same kernel name can be used in different
SYCL applications.

The programs map's key consists of four components:

- device image id<sup>[1](#what-is-diid)</sup>,
- specialization constants values,
- device id<sup>[2](#what-is-did)</sup> this program is built for,
- build options id<sup>[3](#what-is-bopts)</sup>.

Hashes are used for fast built device image identification and shorten
filenames on disk. Once cache directory on disc is identified (see
[Persistent cache storage structure](#persistent-cache-storage-structure)
for detailed directory structure) full key values are compared with the ones
stored on disk (in every <n>.src file located in the cache item directory):

- if they match the built image is loaded from corresponding <n>.bin file;
- otherwise image build is done and new cache item is created on disk
  containing 2 files: <max_n+1>.src for key values and <max_n+1>.bin for
  built image.

(what-is-diid)=
<a name="what-is-diid">1</a>: Hash out of the device code image used as input
for the build.

(what-is-did)=
<a name="what-is-did">2</a>: Hash out of the string which is concatenation of
values for `info::platform::name`, `info::device::name`,
`info::device::version`, `info::device::driver_version` parameters to
differentiate different HW and SW installed on the same host as well as SW/HW
upgrades.

(what-is-bopts)=
<a name="what-is-bopts">3</a>: Hash for the concatenation of build options (both
compile and link options) set in application or environment variables. There are
three sources of build options:

- from device image (sycl_device_binary_struct::CompileOptions,
  sycl_device_binary_struct::LinkOptions);
- environment variables (SYCL_PROGRAM_COMPILE_OPTIONS,
  SYCL_PROGRAM_LINK_OPTIONS);
- options passed through SYCL API.

## Cache configuration

The following environment variables affect the cache behavior:

| Environment variable | Values | Description |
| -------------------- | ------ | ----------- |
| `SYCL_CACHE_DIR` | Path | Path to persistent cache root directory. Default values are `%AppData%\libsycl_cache` for Windows and `$XDG_CACHE_HOME/libsycl_cache` on Linux, if `XDG_CACHE_HOME` is not set then `$HOME/.cache/libsycl_cache`. When none of the environment variables are set SYCL persistent cache is disabled. |
| `SYCL_CACHE_PERSISTENT` | '1' or '0' | Controls persistent device compiled code cache. Turns it on if set to '1' and turns it off if set to '0'. When cache is enabled SYCL runtime will try to cache and reuse JIT-compiled binaries. Default is off. |
| `SYCL_CACHE_IN_MEM` | '1' or '0' | Enable ('1') or disable ('0') in-memory caching of device compiled code. When cache is enabled SYCL runtime will try to cache and reuse JIT-compiled binaries. Default is '1'. |
| `SYCL_IN_MEM_CACHE_EVICTION_THRESHOLD` | Positive integer  | `SYCL_IN_MEM_CACHE_EVICTION_THRESHOLD` accepts an integer that specifies the maximum size of the in-memory program cache in bytes. Eviction is performed when the cache size exceeds the threshold. The default value is 0 which means that eviction is disabled.  |
| `SYCL_CACHE_EVICTION_DISABLE` | Any(\*) | Switches persistent cache eviction off when the variable is set. |
| `SYCL_CACHE_MAX_SIZE` | Positive integer | Persistent cache eviction is triggered once total size of cached images exceeds the value in megabytes (default - 8 192 for 8 GB). Set to 0 to disable size-based cache eviction. |
| `SYCL_CACHE_THRESHOLD` | Positive integer | Persistent cache eviction threshold in days (default value is 7 for 1 week). Set to 0 for disabling time-based cache eviction. |
| `SYCL_CACHE_MIN_DEVICE_IMAGE_SIZE` | Positive integer | Minimum size of device code image in bytes which is reasonable to cache on disk because disk access operation may take more time than do JIT compilation for it. Applicable only for persistent cache. Default value is 0 to cache all images. |
| `SYCL_CACHE_MAX_DEVICE_IMAGE_SIZE` | Positive integer | Maximum size of device image in bytes which is cached. Caching big kernels may overload the disk very fast. Applicable only for persistent cache. Default value is 1 GB. |


## Implementation details

The caches are represented with instance of
[`KernelProgramCache`](https://github.com/intel/llvm/blob/sycl/sycl/source/detail/kernel_program_cache.hpp)
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
up with calling the function
[`getOrBuild()`](https://github.com/intel/llvm/blob/sycl/sycl/source/detail/program_manager/program_manager.cpp)
with number of lambda functions passed as arguments:

- Acquire function;
- GetCache function;
- Build function.

*Acquire* function returns a locked version of cache. Locking is employed for
thread safety. The threads are blocked only for insert-or-acquire attempt, i.e.
when calling to `map::insert` in
[`getOrBuild()`](https://github.com/intel/llvm/blob/sycl/sycl/source/detail/program_manager/program_manager.cpp)
function. The rest of operation is done with the help of atomics and condition
variables (plus a mutex for proper work of condition variable).

*GetCache* function returns a reference to mapping `key->value` out of locked
instance of cache. We will see rationale behind it a bit later.

*Build* function actually builds the kernel or program.

### Thread-safety

Why do we need thread safety here? It is quite possible to have a use-case when
the `sycl::context` is shared across multiple threads (e.g. via sharing a
queue). Possibility of enqueueing multiple cacheable kernels simultaneously
from multiple threads requires us to provide thread-safety for the caching
mechanisms.

It is worth of noting that we don't cache the UR resource (kernel or program)
by itself. Instead we augment the resource with the status of build process.
Hence, what is cached is a wrapper structure `BuildResult` which contains three
information fields - pointer to built resource, build error (if applicable) and
current build status (either of "in progress", "succeeded", "failed").

One can find definition of `BuildResult` template in
[KernelProgramCache](https://github.com/intel/llvm/blob/sycl/sycl/source/detail/kernel_program_cache.hpp).

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
  `MBuildResultMutex` mutex field guarding that variable to implement the
  "single producer-multiple consumers scheme".
- the build result itself appears in the 'Ptr' field when available.

All fields are atomic because they can be accessed from multiple threads.

A specialization of helper class
[Locked](https://github.com/intel/llvm/blob/sycl/sycl/include/sycl/detail/locked.hpp)
for reference of proper mapping is returned by Acquire function. The use of this
class implements RAII to make code look cleaner a bit. Now, GetCache function
will return the mapping to be employed that includes the 3 components: kernel
name, device as well as any specialization constants values. These get added to
`BuildResult` and are cached. The `BuildResult` structure is specialized with
either `ur_kernel_handle_t` or `ur_program_handle_t`<sup>[1](#remove-pointer)</sup>.

### Hash function

STL hash function specialized for std::string is going to be used:
`template<>  struct hash<std::string>`

### Core of caching mechanism

Now, let us see how 'getOrBuild' function works.
First, we fetch the cache with sequential calls to Acquire and GetCache
functions. Then, we check if this is the first attempt to build this kernel or
program. This is achieved with an attempt to insert another key-value pair into
the map. At this point we try to insert `BuildResult` stub instance with status
equal to "in progress" which will allow other threads to know that someone is
(i.e. we're) building the object (i.e. kernel or program) now. If insertion
fails, we will wait for building thread to finish with call to `waitUntilBuilt`
function. This function will throw stored exception<sup>[2](#exception-data)</sup>
upon build failure. This allows waiting threads to see the same result as the
building thread. Special case of the failure is when build result doesn't
contain the error (i.e. the error wasn't of `sycl::exception` type) and the
pointer to object in `BuildResult` instance is nil. In this case, the building
thread has finished the build process and has returned an error to the user.
But this error may be sporadic in nature and may be spurious. Hence, the waiting
thread will try to build the same object once more.

`BuildResult` structure also contains synchronization objects: mutex and
condition variable. We employ them to signal waiting threads that the build
process for this kernel/program is finished (either successfully or with a
failure).

(remove-pointer)=
<a name="remove-pointer">1</a>: The use of `std::remove_pointer` was omitted for
the sake of simplicity here.

(exception-data)=
<a name="exception-data">2</a>: Actually, we store contents of the exception:
its message and error code.

### Persistent cache storage structure

The device code image are stored on file system using structure below:

```bash
<cache_root>/
  <device_hash>/
    <device_image_hash>/
      <spec_constants_values_hash>/
        <build_options_hash>/
          <n>.src
          <n>.bin
```

- `<cache_root>` - root directory storing cache files, that depends on
  environment variables (see SYCL_CACHE_DIR description in the
  [Cache configuration](#cache-configuration));
- `<device_hash>` - hash out of device information used to identify target
  device;
- `<device_image_hash>` - hash made out of device image used as input for the
  JIT compilation;
- `<spec_constants_values_hash>` - hash for specialization constants values;
- `<build_options_hash>` - hash for all build options;
- `<n>` - sequential number of hash collisions. When hashes matches for the
  specific build but full values don't, new cache item is added with incremented
  value (enumeration started from 0).

Two files per cache item are stored on disk:

- `<n>.src` contains full values for build parameters (device information,
  specialization constant values, build options, device image) which is used to
  resolve hash collisions and analysis of cached items.
- `<n>.bin` contains built device code.

### Inter-process safety

For on-disk cache there might be access collisions for accessing the same file
from different instances of SYCL applications:

- write collision happens when 2 instances of the same application are started
  to write to the same cache file/directory;
- read collision may happen if one application is writing to the file and the
  other instance of the application is trying to read from it while write
  operation is not finished.

To avoid collisions the file system entries are locked for read-write access
until write operation is finished. e.g if new file or directory should be
created/deleted parent directory is locked, file is created in locked state,
then the directory and the file are unlocked.

To address cases with high lock rate (multiple copies of the SYCL applications
are run in parallel and use the same cache directory) nested directories
representing cache keys are used to minimize locks down to applications which
build the same device with the same parameters. Directory is locked for minimum
time because it can be unlocked once subdirectory is created. If file is created
in a directory, the directory should be locked until file creation is done.

Advisory locking <sup>[1](#advisory-lock)</sup> is used to ensure that the
user/OS tools are able to manage files.

(advisory-lock)=
<a name="advisory-lock">1.</a> Advisory locks work only when a process
explicitly acquires and releases locks, and are ignored if a process is not
aware of locks.

### Cache eviction mechanism

Cache eviction mechanism is required to avoid resources overflow both for
memory and disk. The general idea is to delete items following cache size or
LRU (least recently used) strategy both for in-memory and persistent cache.

#### In-memory cache eviction

Eviction in in-memory cache is disabled by default but can be controlled by SYCL_IN_MEM_CACHE_EVICTION_THRESHOLD
environment variable. The threshold is set in bytes and when the cache size exceeds the threshold the eviction process is initiated. The eviction process is based on LRU strategy. The cache is walked through and the least recently used items are deleted until the cache size is below the threshold.
To implement eviction for in-memory cache efficiently, we store the programs in a linked-list, called the eviction list. When the program is first added to the cache, it is also added to the back of the eviction list. When a program is fetched from cache, we move the program to the end of the eviction list. This way, we ensure that the programs at the beginning of the eviction list are always the least recently used.
When adding a new program to cache, we check if the size of the program cache exceeds the threshold, if so, we iterate through the eviction list starting from the front and delete the programs until the cache size is below the threshold. When a program is deleted from the cache, we also evict its corresponding kernels from both of the kernel caches.

***If the application runs out-of-memory,*** either due to cache eviction being disabled or the cache eviction threshold being too high, we will evict all the items from program and kernel caches.

#### Persistent cache eviction

Persistent cache eviction can be enabled using the SYCL_CACHE_MAX_SIZE environment variable and is based on the LRU strategy.

- A new file, called `cache_size.txt`, is created at the root of the persistent cache directory. This file contains the total size of the cache in bytes. When a new item is added to the cache, the size of the item is added to the total size in the `cache_size.txt` file. When the total size exceeds the threshold, the eviction process is initiated.

- Whenever a cache entry is added or accessed, the corresponding cache item directory is updated with the current time. This is done by creating a new file, called `<entry name>_access_time.txt`, in the cache item directory. This file contains the current time in nanoseconds since the epoch. When the eviction process is initiated, we use this file to determine the last access time of the cache item.

- When a new item is added to the cache, we check if the total size exceeds the threshold. If so, we iterate through the cache item directories and delete the least recently accessed items until the total size is below half the cache size.

Note that once the eviction is triggered, the cache size is reduced to half the cache size to avoid frequent eviction.


## Cache limitations

The caching isn't done when:

- when program is built out of source with `program::build_with_source()` or
  `program::compile_with_source()` method;
- when program is a result of linking multiple programs;
- program is built using interoperability methods.

## Points of improvement (things to do)

- Employ the same built object for multiple devices of the same ISA,
  capabilities and so on. *NOTE:* It's not really known if it's possible to
  check if two distinct devices are *exactly* the same. Probably this should be
  an improvement request for the UR adapters. By now it is assumed that two
  devices with the same device id <a name="what-is-did">2</a> are the same.
- Improve testing: cover real use-cases. See currently covered cases
  [here](https://github.com/intel/llvm/blob/sycl/sycl/unittests/kernel-and-program/Cache.cpp).
- Implement tool for exploring cache items (initially it is possible using OS
  utilities).
