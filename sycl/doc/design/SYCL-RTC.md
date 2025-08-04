# SYCL Runtime Compilation

SYCL-RTC means using the
[`kernel_compiler`](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_kernel_compiler.asciidoc)
extension to wrap a SYCL source string comprised of kernel definitions in the
[free-function syntax](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/proposed/sycl_ext_oneapi_free_function_kernels.asciidoc)
into a `kernel_bundle` in the `ext_oneapi_source` state, which is then compiled
into `exectuable` state by the extension's `build(...)` function. The feature is
backed by an implementation inside the `sycl-jit` library, which exposes the
modular, LLVM-based compiler tech behind DPC++ to be called by the SYCL runtime.
This document gives an overview of the design.

```c++
#include <sycl/sycl.hpp>
namespace syclexp = sycl::ext::oneapi::experimental;

// ...

std::string sycl_source = R"""(
  #include <sycl/sycl.hpp>
  
  extern "C" SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((
    sycl::ext::oneapi::experimental::nd_range_kernel<1>))
  void vec_add(float* in1, float* in2, float* out){
    size_t id = sycl::ext::oneapi::this_work_item::get_nd_item<1>()
                .get_global_linear_id();
    out[id] = in1[id] + in2[id];
  }
)""";

sycl::queue q;

auto source_bundle = syclexp::create_kernel_bundle_from_source(
  q.get_context(), syclexp::source_language::sycl, sycl_source);

// This design document explains what happens on the next line.
auto exec_bundle = syclexp::build(source_bundle);
```

## File-and-process-based prototype

The
[first implementation](https://github.com/intel/llvm/blob/03cb2b25026f060149eb94c85b228e5b3a780588/sycl/source/detail/kernel_compiler/kernel_compiler_sycl.cpp#L254)
of the `build(...)` function wrote the source string into a temporary file,
invoked DPC++ on it with the `-fsycl-dump-device-code` flag to dump the device
code to another file in SPIR-V format, and finally loaded that file back into
the runtime, from where it was executed. 

## The rationale for an in-memory compilation pipeline

Invoking the DPC++ executable as outlined in the previous section worked
reasonably well to implement the basic `kernel_compiler` extension, but we
observed several shortcomings:

- Functional completeness: Emitting a single SPIR-V file is sufficient for
  simple kernels, but more advanced device code may result in multiple *device
  images* comprised of SPIR-V binaries and accompanying metadata (*runtime
  properties*) that needs to be communicated to the runtime.
- Robustness: Reading multiple dependent files from a temporary directory can be
  be fragile.
- Performance: Multiple processes are launched by the compiler driver, and file
  I/O operations have a non-negligible overhead. The `-fsycl-dump-device-code`
  required the presence of a dummy `main()` to be added to the source string,
  and caused an unnecessary host compilation to be performed.
- Security: Reading executable code from disk is a security concern, and users
  of an RTC-enabled application may be unaware that a compilation writing
  intermediate files is happening in the background.

These challenges ultimately motivated the design of the **in-memory compilation
pipeline** based on the `sycl-jit` library which is now the default approach in
DPC++ and the oneAPI product distribution since the 2025.2 release. This new
approach leverages **modular compiler technology** to produce a faster, more
feature-rich, more robust and safer implementation of the `kernel_compiler`
extension.

The individual steps in the pipeline (frontend, device library linking,
`sycl-post-link` and target format translation) are now invoked programmatically
via an API inside the same process, and intermediate results are passed along as
objects in memory. The code can be found in the
[`compileSYCL(...)`](https://github.com/intel/llvm/blob/cc966df07d29db75d07f969f044c0491819bd930/sycl-jit/jit-compiler/lib/rtc/RTC.cpp#L57)
function.
 
## Using the LibTooling API to compile the source string to an `llvm::Module`

[LibTooling](https://clang.llvm.org/docs/LibTooling.html) is a high-level API to
write standalone tools based on Clang, such as linters, refactoring tools or
static analysers. To use it, one defines a *tool action* to run on a set of
files in a *virtual filesystem overlay*, which the frontend then processes
according to a *compilation command database*.

For SYCL-RTC, the filesystem overlay is populated with files containing the
source string and any virtual `include_files` (defined via the homonymous
property). The compilation command is static and puts the frontend into
`-fsycl-device-only` mode. Any user-given options (from the `build_options`
property) are appended. Lastly, the implementation defines a custom tool action
which runs the frontend until LLVM codegen, and then obtains ownership of the
LLVM module.

This might be a slightly unusual way to use of LibTooling, but we found it works
great for SYCL-RTC. The next sections explain the
[`jit_compiler::compileDeviceCode(...)`](https://github.com/intel/llvm/blob/cc966df07d29db75d07f969f044c0491819bd930/sycl-jit/jit-compiler/lib/rtc/DeviceCompilation.cpp#L418)
function in more detail.

### Step 1: Determine the path of the compiler installation

To set up up working frontend invocation, we need to know where to find
supplemental files such as the SYCL headers. Normally, these paths are
determined relative to the compiler executable, however in our case, the
executable is actually the RTC-enabled application, which can reside in an
arbitrary location. Instead, we use OS-specific logic inside
[`getDPCPPRoot()`](https://github.com/intel/llvm/blob/cc966df07d29db75d07f969f044c0491819bd930/sycl-jit/jit-compiler/lib/rtc/DeviceCompilation.cpp#L112)
to determine the location of the shared library `sycl-jit.so` (or `.dll` on
Windows) which contains the SYCL-RTC implementation. From its location, we can
derive the compiler installation's root directory.

### Step 2: Collect command-line arguments

The next step is to collect the command-line arguments for the frontend
invocation. The
[`adjustArgs(...)`](https://github.com/intel/llvm/blob/cc966df07d29db75d07f969f044c0491819bd930/sycl-jit/jit-compiler/lib/rtc/DeviceCompilation.cpp#L320)
function relies on Clang's option handling infrastructure to set the required
options to enter the device compilation mode (`-fsycl-device-only`), set up the
compiler environment, and select the target. Finally, any user-specified
arguments passed via the
[`build_options`](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_kernel_compiler.asciidoc#new-properties-for-the-build-and-compile-functions)
property are appended to the list of command-line arguments.

### Step 3: Configure the `ClangTool`

Once we know the required command-line arguments, we can set up the compilation
command database and an
[instance](https://github.com/intel/llvm/blob/cc966df07d29db75d07f969f044c0491819bd930/sycl-jit/jit-compiler/lib/rtc/DeviceCompilation.cpp#L433)
of the
[`ClangTool`](https://github.com/intel/llvm/blob/cc966df07d29db75d07f969f044c0491819bd930/clang/include/clang/Tooling/Tooling.h#L317)
class, which provides the entry point to the LibTooling interface. As we'll be
translating only a single file containing the source string, we construct a
[`FixedCompilationDatabase`](https://github.com/intel/llvm/blob/cc966df07d29db75d07f969f044c0491819bd930/clang/include/clang/Tooling/CompilationDatabase.h#L154)
relative to the current working directory.

To implement the `kernel_compiler` extension cleanly, we need to capture all
output (e.g. warnings and errors) from the frontend. The
[`ClangDiagnosticsWrapper`](https://github.com/intel/llvm/blob/cc966df07d29db75d07f969f044c0491819bd930/sycl-jit/jit-compiler/lib/rtc/DeviceCompilation.cpp#L274)
class configures a
[`TextDiagnosticsPrinter`](https://github.com/intel/llvm/blob/cc966df07d29db75d07f969f044c0491819bd930/clang/include/clang/Frontend/TextDiagnosticPrinter.h#L27)
to append all messages to a string maintained by our implementation to collect
all output produced during the runtime compilation.

The configuration of the `ClangTool` instance continues in the
[`setupTool`](https://github.com/intel/llvm/blob/cc966df07d29db75d07f969f044c0491819bd930/sycl-jit/jit-compiler/lib/rtc/DeviceCompilation.cpp#L353)
function. First, we redirect all output to our diagnostics wrapper. Then, we
[set up the overlay
filesystem](https://github.com/intel/llvm/blob/cc966df07d29db75d07f969f044c0491819bd930/sycl-jit/jit-compiler/lib/rtc/DeviceCompilation.cpp#L361-L364)
with a file named `rtc_<n>.cpp` (*n* is incremented for each use of the
`kernel_compiler` extension's `build(...)` function) in the current directory
with the contents of the source string. Each of the virtual header files that
the application defined via the
[`include_files`](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_kernel_compiler.asciidoc#new-properties-for-the-create_kernel_bundle_from_source-function)
property becomes also a file in the overlay filesystem, using the path specified
in the property.

The `ClangTool` class exposes so-called argument adjusters, which are intended
to modify the command-line arguments coming from the compilation command
database. We have to
[clear](https://github.com/intel/llvm/blob/cc966df07d29db75d07f969f044c0491819bd930/sycl-jit/jit-compiler/lib/rtc/DeviceCompilation.cpp#L368)
the default adjusters defined by the class, because one of them injects the
`-fsyntax-only` flag, which would conflict with the `-fsycl-device-only` flag we
need for SYCL-RTC. Finally, we
[add](https://github.com/intel/llvm/blob/cc966df07d29db75d07f969f044c0491819bd930/sycl-jit/jit-compiler/lib/rtc/DeviceCompilation.cpp#L371)
an argument adjuster ourselves to overwrite the name of executable in the
invocation. Again, this is to help the correct detection of the environment, by
making the invocation as similar as possible to a normal use of DPC++.

### Step 4: Run an action

The last step is to define a
[`ToolAction`](https://github.com/intel/llvm/blob/cc966df07d29db75d07f969f044c0491819bd930/clang/include/clang/Tooling/Tooling.h#L80)
to be executed on the source files. Clang conveniently provides the
[`EmitLLVMAction`](https://github.com/intel/llvm/blob/cc966df07d29db75d07f969f044c0491819bd930/clang/include/clang/CodeGen/CodeGenAction.h#L103),
which runs the frontend up until the LLVM IR code generation, which is exactly
what we need. However, LibTooling does not provides a helper to wrap it in a
`ToolAction`, so we need to define and run our own
[`GetLLVMModuleAction`](https://github.com/intel/llvm/blob/cc966df07d29db75d07f969f044c0491819bd930/sycl-jit/jit-compiler/lib/rtc/DeviceCompilation.cpp#L241). 

We extracted common boilerplate code to configure a
[`CompilerInstance`](https://github.com/intel/llvm/blob/cc966df07d29db75d07f969f044c0491819bd930/clang/include/clang/Frontend/CompilerInstance.h#L81)
in the
[`RTCActionBase`](https://github.com/intel/llvm/blob/cc966df07d29db75d07f969f044c0491819bd930/sycl-jit/jit-compiler/lib/rtc/DeviceCompilation.cpp#L176)
class. Inside  the `GetLLVMModuleAction`, we instantiate and execute the
aforementioned `EmitLLVMAction`, and, in case the translation was successful,
[obtains ownership](https://github.com/intel/llvm/blob/cc966df07d29db75d07f969f044c0491819bd930/sycl-jit/jit-compiler/lib/rtc/DeviceCompilation.cpp#L255)
of the constructed `llvm::Module` from it.

Finally, the call to
[`Action.takeModule()`](https://github.com/intel/llvm/blob/cc966df07d29db75d07f969f044c0491819bd930/sycl-jit/jit-compiler/lib/rtc/DeviceCompilation.cpp#L442)
transfers ownership again to the caller of `compileDeviceCode`. Note that this
simple mechanism works because we know that there is only a single compilation
happening for every instance of the `ClangTool` and hence our
`GetLLVMModuleAction` class.

## Caching

The implementation optionally uses the runtime's
[persistent cache](https://intel.github.io/llvm/design/KernelProgramCache.html#persistent-cache)
to elide recurring invocations of the frontend, which we observed to be the most
expensive (in terms of runtime overhead) phase of our compilation pipeline.

### Overall design

We cache only the frontend invocation, meaning that after a successful
translation, we store the LLVM IR module obtained via LibTooling on disk in the
Bitcode format using built-in utilities. In case of a cache hit in a later
runtime compilation, we load the module from disk and feed it into the device
linking phase. The rationale for this design was that were no utilities to save
and restore the linked and post-processed device images to disk at the time (the
[SYCLBIN](https://intel.github.io/llvm/design/SYCLBINDesign.html) infrastructure
was added later), and caching these steps would have resulted only in marginal
further runtime savings.

### Cache key considerations

The main challenge is to define a robust cache key. Because code compiled via
SYCL-RTC can `#include` header files defined via the `include_files` property as
well as from the filesystem, e.g. `sycl.hpp` from the DPC++ installation or user
libraries, it is not sufficient to look only at the source string. In order to
make the cache as conservative as possible (cache collisions are unlikely but
mathematically possible), we decided to compute a hash value of the
*preprocessed* source string, i.e. with all `#include` directives resolved. We
additionally compute a hash value of the rendered command-line arguments, and
append it to the hash of the preprocessed source to obtain the final cache key. 

### Implementation notes

The cache key computation is implemented in the
[`jit_compiler::calculateHash(...)`](https://github.com/intel/llvm/blob/cc966df07d29db75d07f969f044c0491819bd930/sycl-jit/jit-compiler/lib/rtc/DeviceCompilation.cpp#L381)
function. We are again relying on LibTooling to invoke the preprocessor -
handily, Clang provides  a
[`PreprocessorFrontendAction`](https://github.com/intel/llvm/blob/cc966df07d29db75d07f969f044c0491819bd930/clang/include/clang/Frontend/FrontendAction.h#L294)
that we extend to tailor to our use-case. We choose
[BLAKE3](https://en.wikipedia.org/wiki/BLAKE_(hash_function)) as the hash
algorithm because its proven in similar contexts (most notably,
[ccache](https://ccache.dev)) and available as a utility in the LLVM ecosystem.
As the output is a byte array, we apply Base64 encoding to obtain a character
string for use with the persistent cache.

## Device library linking and SYCL-specific transformations

With an LLVM IR module in hand, obtained either from the frontend or the cache,
the next steps in the compilation pipeline are simple.

The device library linking is done by the
[`jit_compiler::linkDeviceLibraries(...)`](https://github.com/intel/llvm/blob/cc966df07d29db75d07f969f044c0491819bd930/sycl-jit/jit-compiler/lib/rtc/DeviceCompilation.cpp#L566)
function. These libraries provide primitives for a variety of extra
functionality, such as an extended set of math functions and support for
`bfloat16` arithmetic, and are available as Bitcode files inside the DPC++
installation or the vendor toolchain, so we just use LLVM utilities to load them
into memory and link them to the module representing the runtime-compiled
kernels.

For the SYCL-specific post-processing, implemented in
[`jit_compiler::performPostLink(...)`](https://github.com/intel/llvm/blob/cc966df07d29db75d07f969f044c0491819bd930/sycl-jit/jit-compiler/lib/rtc/DeviceCompilation.cpp#L750),
we can reuse modular analysis and transformation passes in the
[`SYCLLowerIR`](https://github.com/intel/llvm/tree/sycl/llvm/lib/SYCLLowerIR)
component. The main tasks for the post-processing passes is to split the device
code module into smaller units (either as requested by the user, or required by
the ESIMD mode), and to compute the properties that need to be passed to the
SYCL runtime when the device images are loaded.

## Translation to the target format

The 
[final phase](https://github.com/intel/llvm/blob/cc966df07d29db75d07f969f044c0491819bd930/sycl-jit/jit-compiler/lib/rtc/RTC.cpp#L139)
in the pipeline is to translate the LLVM IR modules resulting from the previous
phase into a device-specific target format that can be handled by the runtime.
For Intel CPUs and GPUs, that's binary SPIR-V. For AMD and NVIDIA GPUs, we emit
AMDGCN and PTX assembly, respectively. Over time, we created our own set of
[utilities](https://github.com/intel/llvm/blob/cc966df07d29db75d07f969f044c0491819bd930/sycl-jit/jit-compiler/lib/translation/Translation.h)
to facilitate the translation. Internally, we dispatch the task to either the
SPIR-V translator (a copy of which is maintained inside the DPC++ repository),
or use vendor-specific backends that are part of LLVM to generate the
third-party GPU code.

## Third-party hardware support

SYCL-RTC works for AMD and NVIDIA GPUs, too. The usage of the `kernel_compiler`
extension remains the same for SYCL devices representing such a third-party GPU.
The concrete GPU architecture is queried via the environment variable
`SYCL_JIT_AMDGCN_PTX_TARGET_CPU` when executing the RTC-enabled application. For
AMD GPUs, it is **mandatory** to set it. For NVIDIA GPUs, it is highly
recommended to change it from the conservative default architecture (`sm_50`). 

```shell
$ clang++ -fsycl myapp.cpp -o myapp
$ SYCL_JIT_AMDGCN_PTX_TARGET_CPU=sm_90 ./myapp
```

A list of values that can be set as the target CPU can be found in the
[documentation of the `-fsycl-targets=`
option](https://intel.github.io/llvm/UsersManual.html#generic-options) (leave
out the `amd_gpu_` and `nvidia_gpu_` prefixes).

At the moment, the support is available in [daily
builds](https://github.com/intel/llvm/releases) of the open-source version of
DPC++.

## Further reading

- Technical presentation at IWOCL 2025: *Fast In-Memory Runtime Compilation of
  SYCL Code*:
  [Slides](https://www.iwocl.org/wp-content/uploads/iwocl-2025-julian-oppermann-fast-in-memory-runtime.pdf)
  [Video Recording](https://youtu.be/X9mS8xetZJY)
- Blog post:
  [*SYCL Runtime Compilation: A New Way to Specialise Kernels Using C++ Metaprogramming*](https://codeplay.com/portal/blogs/2025/07/08/sycl-runtime-compilation)
