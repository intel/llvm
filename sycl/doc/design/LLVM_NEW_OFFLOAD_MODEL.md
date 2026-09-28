# New Offload Model in Clang

This document describes the "New Offload Model" introduced in Clang for handling offloading compilation to heterogeneous programming models. This model is now the default for OpenMP, CUDA, HIP in LLVM 23. We are working to make it default for SYCL offloading.

## Table of Contents

- [Introduction](#introduction)
- [High-Level Overview](#high-level-overview)
- [Comparison with Old Model](#comparison-with-old-model)
- [Compilation Pipeline](#compilation-pipeline)
- [Detailed Step-by-Step Walkthrough](#detailed-step-by-step-walkthrough)
- [Key Data Structures](#key-data-structures)
- [The clang-linker-wrapper Tool](#the-clang-linker-wrapper-tool)
- [Command Line Examples](#command-line-examples)
- [References](#references)

## Introduction

The New Offload Model is a unified approach to handling device code compilation and linking across different offloading models (OpenMP, CUDA, HIP, SYCL). It centralizes device linking logic in a single tool called **clang-linker-wrapper**, which wraps the host linker and performs device-specific linking operations transparently.

**Key Features:**
- Unified device linking infrastructure across all offloading models
- Transparent wrapper around the host linker
- Support for Link-Time Optimization (LTO) on device code
- Relocatable linking support for shipping libraries with embedded device code
- Standardized binary format for embedded device images

**Status:**
- Can be explicitly enabled with `--offload-new-driver` flag
- Old model can still be used with `--no-offload-new-driver` (deprecated)

## High-Level Overview

### Compilation Flow

```
┌─────────────────┐
│  Source File    │
│   (e.g., .cpp)  │
└────────┬────────┘
         │
         v
┌─────────────────────────────────────────────────────────┐
│  HOST COMPILATION                                        │
│  - Compile for host architecture                         │
│  - Lower offload directives to metadata                  │
│  - Output: host.bc (bitcode)                            │
└────────┬────────────────────────────────────────────────┘
         │
         v
┌─────────────────────────────────────────────────────────┐
│  DEVICE COMPILATION (per device target)                 │
│  - Compile for device architecture                       │
│  - Extract kernels and device code                       │
│  - Link device runtime libraries                         │
│  - Output: device.o or device.bc                        │
└────────┬────────────────────────────────────────────────┘
         │
         v
┌─────────────────────────────────────────────────────────┐
│  CREATE FAT OBJECT                                       │
│  - Embed device image in .llvm.offloading section       │
│  - Use OffloadBinary format (magic: 0x10FF10AD)         │
│  - Compile host code to object                           │
│  - Output: fat.o (host object + embedded device code)   │
└────────┬────────────────────────────────────────────────┘
         │
         v
┌─────────────────────────────────────────────────────────┐
│  CLANG-LINKER-WRAPPER                                    │
│  1. Scan input files for .llvm.offloading sections      │
│  2. Extract all device images                            │
│  3. Group by (triple, arch, offload-kind)               │
│  4. Link device code for each target                     │
│  5. Bundle linked images (create fat binary)            │
│  6. Wrap bundled images with registration code          │
│  7. Invoke host linker with wrapped images              │
└────────┬────────────────────────────────────────────────┘
         │
         v
┌─────────────────┐
│  Final Binary   │
│  (executable or │
│   shared lib)   │
└─────────────────┘
```

## Comparison with Old Model

| Aspect | Old Model | New Model |
|--------|-----------|-----------|
| **Device Linking** | Per-model tools (`llvm-link`, `ld.lld`) | Unified `clang-linker-wrapper` |
| **Binary Format** | Tool-specific (bundler, file-table) | Standardized `OffloadBinary` format |
| **Tools** | `clang-offload-bundler`, `clang-offload-wrapper` | `clang-linker-wrapper`, `llvm-offload-binary` |
| **Embedding** | Various methods | `.llvm.offloading` ELF section |
| **LTO Support** | Limited | Full support across all models |
| **Relocatable Link** | Not supported | Full support (`-r` flag) |
| **Code Generation** | Driver orchestration | Linker-wrapper orchestration |

## Compilation Pipeline

### Phase 1: Device Compilation

For each target device architecture, Clang compiles the source file to create device code.

**Key Flags:**
- `-fopenmp-is-target-device` (OpenMP)
- `-fsycl-device-only` (SYCL)
- `-x cuda` (CUDA)
- `-x hip` (HIP)

**Device-Specific Passes:**
- Kernel outlining
- Address space inference
- Device-specific optimizations
- Link device runtime libraries

### Phase 2: Embedding Device Code

Device images are embedded into the host object file using the `-fembed-offload-object` flag in clang.

**Binary Format (`OffloadBinary`):**
- Magic bytes: `0x10FF10AD`
- Contains metadata: triple, arch, offload kind, image type
- Stores device image as binary blob
- Multiple images can be embedded in same section

**Section:** `.llvm.offloading` (ELF section with `SHF_EXCLUDE` flag)

### Phase 3: Linker Wrapper Execution

The `clang-linker-wrapper` tool:

1. **Scans** all input `.o` files for `.llvm.offloading` sections
2. **Extracts** `OffloadBinary` entries from each file
3. **Groups** device code by (offload-kind, triple, architecture)
4. **Links** device code for each unique target
5. **Wraps** linked device images with runtime registration code
6. **Invokes** host linker with both host objects and wrapped device objects

## Detailed Step-by-Step Walkthrough

### Step 1: Compile Source for Host and Device

**Command:**
```bash
clang++ -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda example.cpp -c -o example.o
```

**What Happens:**

1. **Host Compilation** ([clang/lib/CodeGen/CGStmtOpenMP.cpp](https://github.com/llvm/llvm-project/blob/main/clang/lib/CodeGen/CGStmtOpenMP.cpp))
   - Lower `#pragma omp target` to runtime calls
   - Create offloading entry metadata in IR
   - Store in `omp_offload.info` metadata node
   - Output: `example_host.bc`

2. **Device Compilation** ([clang/lib/CodeGen/CodeGenModule.cpp](https://github.com/llvm/llvm-project/blob/main/clang/lib/CodeGen/CodeGenModule.cpp))
   - Extract device kernels
   - Compile with `-fopenmp-is-target-device`
   - Link OpenMP device runtime (`libomptarget-nvptx64.bc`)
   - Output: `example_device.o`

3. **Create Fat Object** ([clang/lib/Driver/ToolChains/Clang.cpp](https://github.com/llvm/llvm-project/blob/main/clang/lib/Driver/ToolChains/Clang.cpp))
   ```cpp
   // In clang driver:
   // Add -fembed-offload-object=<device-image>
   ```
   - Embed device object using `-fembed-offload-object`
   - Create `OffloadBinary` structure
   - Place in `.llvm.offloading` section
   - Compile host bitcode to object

**Result:** `example.o` contains both host code and embedded device image

### Step 2: Linking with clang-linker-wrapper

**Command:**
```bash
clang++ -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda example.o -o example.out
```

**What Happens:**

The driver invokes `clang-linker-wrapper` instead of the linker directly:

```bash
clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu \
  --linker-path=/usr/bin/ld.lld -- \
  <standard linker arguments> example.o
```

#### 2.1: Extract Device Code

**Function:** [`getDeviceInput()`](https://github.com/llvm/llvm-project/blob/main/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L1186)

```cpp
Expected<SmallVector<SmallVector<OffloadFile>>>
getDeviceInput(const ArgList &Args) {
  // Scan all input files
  for (const opt::Arg *Arg : Args.filtered(OPT_INPUT, OPT_library)) {
    // Read file
    MemoryBuffer::getFileOrSTDIN(*Filename);

    // Extract offload binaries from .llvm.offloading section
    extractOffloadBinaries(Buffer, Binaries);
  }

  // Group by compatible targets
  MapVector<OffloadFile::TargetID, SmallVector<OffloadFile>> InputFiles;
  // ...
}
```

**What it does:**
1. Scans each `.o` file for `.llvm.offloading` section
2. Parses `OffloadBinary` format (magic bytes `0x10FF10AD`)
3. Extracts metadata: offload kind, triple, arch, image type
4. Collects all device images by target ID

#### 2.2: Link Device Code

**Function:** [`linkDevice()`](https://github.com/llvm/llvm-project/blob/main/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L626)

```cpp
Expected<StringRef> linkDevice(ArrayRef<StringRef> InputFiles,
                               const ArgList &Args,
                               uint16_t ActiveOffloadKindMask) {
  const llvm::Triple Triple(Args.getLastArgValue(OPT_triple_EQ));

  // Dispatch to appropriate linker based on target
  switch (Triple.getArch()) {
  case Triple::nvptx64:
  case Triple::amdgcn:
  case Triple::spirv64:
    return generic::clang(InputFiles, Args, ActiveOffloadKindMask);
  }
}
```

**For NVIDIA GPU:**
```bash
# Invokes clang internally:
clang --target=nvptx64-nvidia-cuda \
  -march=sm_80 \
  -o linked.device.img \
  device1.o device2.o device3.o
```

**For SYCL:** (special handling)
```cpp
if (ActiveOffloadKindMask & OFK_SYCL) {
  CmdArgs.push_back("--sycl-link");
  CmdArgs.append({"-Xlinker", "-triple=<target>"});
  CmdArgs.append({"-Xlinker", "-arch=<arch>"});
}
```
This invokes `clang-sycl-linker` via the `--sycl-link` flag.

**For AMD GPU:**
```bash
clang --target=amdgcn-amd-amdhsa \
  -mcpu=gfx906 \
  -flto \
  -o linked.device.img \
  device1.bc device2.bc device3.bc
```

#### 2.3: Bundle Device Images

**Why Bundling is Needed:**

After linking device code for each target, we may have multiple device images (e.g., one for `sm_75`, one for `sm_80`, one for `gfx906`). Each runtime has its own expected format for storing multiple device architectures in a single binary. **Bundling** is the process of packaging these multiple device images into a single fat binary that the runtime can parse to select the appropriate architecture at load time.

Think of it as creating a "multi-architecture container" where:
- The host program loads one fat binary
- The runtime inspects the system (which GPU is present)
- The runtime extracts and loads only the matching device image

Different offload models use different bundling formats because they have different runtime requirements and toolchain ecosystems.

**Function:** [`bundleLinkedOutput()`](https://github.com/llvm/llvm-project/blob/main/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L880)

**OpenMP - Simple OffloadBinary Wrapping:**
```cpp
Expected<SmallVector<std::unique_ptr<MemoryBuffer>>>
bundleOpenMP(ArrayRef<OffloadingImage> Images) {
  // Each image wrapped individually as OffloadBinary format
  // libomptarget expects this format
  for (const OffloadingImage &Image : Images)
    Buffers.emplace_back(
        MemoryBuffer::getMemBufferCopy(OffloadBinary::write(Image)));
}
```
*OpenMP uses the simple OffloadBinary format since libomptarget is designed to parse it.*

**SYCL - Pre-Bundled Format:**
```cpp
Expected<SmallVector<std::unique_ptr<MemoryBuffer>>>
bundleSYCL(ArrayRef<OffloadingImage> Images) {
  // clang-sycl-linker already created the bundled format
  // SYCL runtime expects a specific binary layout created by the linker
  // Pass through as-is
  for (const OffloadingImage &Image : Images) {
    StringRef S(Image.Image->getBufferStart(),
                Image.Image->getBufferSize());
    Buffers.emplace_back(MemoryBuffer::getMemBufferCopy(S));
  }
}
```
*SYCL's device linker (`clang-sycl-linker`) already produces the final bundled format, so no additional bundling is needed.*

**CUDA - NVIDIA fatbinary:**
```cpp
Expected<SmallVector<std::unique_ptr<MemoryBuffer>>>
bundleCuda(ArrayRef<OffloadingImage> Images, const ArgList &Args) {
  // Use NVIDIA's fatbinary tool to create proprietary fat binary format
  // CUDA driver expects this specific format
  SmallVector<std::pair<StringRef, StringRef>> InputFiles;
  for (const OffloadingImage &Image : Images)
    InputFiles.emplace_back(Image.Image->getBufferIdentifier(),
                           Image.StringData.lookup("arch"));

  return nvptx::fatbinary(InputFiles, Args);
}
```

The `nvptx::fatbinary()` function invokes NVIDIA's `fatbinary` tool:
```bash
fatbinary --create output.fatbin \
  --image3=kind=elf,sm=80,file=sm80.o \
  --image3=kind=elf,sm=75,file=sm75.o \
  --image3=kind=elf,sm=86,file=sm86.o
```
*CUDA runtime requires NVIDIA's proprietary fatbinary format, which packages multiple architectures with metadata.*

**HIP/AMD - clang-offload-bundler:**
```cpp
Expected<SmallVector<std::unique_ptr<MemoryBuffer>>>
bundleHIP(ArrayRef<OffloadingImage> Images, const ArgList &Args) {
  // Use clang-offload-bundler to create HIP fat binary
  // ROCm runtime expects this bundled format
  SmallVector<std::tuple<StringRef, StringRef, StringRef>> InputFiles;
  for (const OffloadingImage &Image : Images)
    InputFiles.emplace_back(
        std::make_tuple(Image.Image->getBufferIdentifier(),
                       Image.StringData.lookup("triple"),
                       Image.StringData.lookup("arch")));

  return amdgcn::fatbinary(InputFiles, Args);
}
```

The `amdgcn::fatbinary()` function (located at line 446) invokes `clang-offload-bundler`:
```bash
clang-offload-bundler \
  -type=o \
  -bundle-align=4096 \
  -targets=host-x86_64-unknown-linux-gnu,hip-amdgcn-amd-amdhsa-gfx906,hip-amdgcn-amd-amdhsa-gfx908 \
  -input=/dev/null \
  -input=gfx906.o \
  -input=gfx908.o \
  -output=output.hipfb
```
*HIP uses the offload bundler to create a fat binary containing host stub and multiple AMD GPU architectures. The bundler creates a special ELF with multiple architecture sections.*

**Key Differences:**
| Offload Kind | Bundler Tool | Output Format | Why This Format? |
|--------------|--------------|---------------|------------------|
| OpenMP | OffloadBinary::write() | OffloadBinary format | Platform-independent, libomptarget native |
| SYCL | clang-sycl-linker | SYCL-specific | Pre-bundled by device linker |
| CUDA | fatbinary (NVIDIA) | CUDA fatbinary | Required by CUDA driver API |
| HIP | clang-offload-bundler | HIP bundle format | Compatible with ROCm runtime |

#### 2.4: Wrap Device Images with Registration Code

**Function:** [`wrapDeviceImages()`](https://github.com/llvm/llvm-project/blob/main/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L729)

After bundling, we need to **wrap** the fat binary with host-side code that registers it with the offload runtime. This creates an object file containing:
1. The bundled device binary as raw data
2. Metadata structures describing the binary
3. Constructor/destructor functions to register/unregister with the runtime

```cpp
Expected<StringRef>
wrapDeviceImages(ArrayRef<std::unique_ptr<MemoryBuffer>> Buffers,
                 const ArgList &Args, OffloadKind Kind) {
  // Create LLVM IR module for host
  LLVMContext Context;
  Module M("offload.wrapper.module", Context);
  M.setTargetTriple(Triple(/* host triple */));

  switch (Kind) {
  case OFK_OpenMP:
    offloading::wrapOpenMPBinaries(M, BuffersToWrap,
                                   offloading::getOffloadEntryArray(M),
                                   /*Suffix=*/"",
                                   /*Relocatable=*/Args.hasArg(OPT_relocatable));
    break;
  case OFK_SYCL:
    offloading::wrapSYCLBinaries(M, BuffersToWrap.front(), Options);
    break;
  // ...
  }

  // Compile wrapper module to object file
  return compileModule(M, Kind);
}
```

**Generated Wrapper Structure (OpenMP example):**

The wrapper creates LLVM IR that contains:

1. **Device Binary Data:**
   ```llvm
   @__omp_offloading_binary = internal constant [N x i8] c"<binary data>"
   ```

2. **Device Image Descriptor:**
   ```c
   struct __tgt_device_image {
     void *ImageStart;  // Points to binary data
     void *ImageEnd;
     __tgt_offload_entry *EntriesBegin;  // Offload entry table
     __tgt_offload_entry *EntriesEnd;
   };
   ```

3. **Binary Descriptor:**
   ```c
   struct __tgt_bin_desc {
     int32_t NumDeviceImages;
     __tgt_device_image *DeviceImages;
     __tgt_offload_entry *HostEntriesBegin;
     __tgt_offload_entry *HostEntriesEnd;
   };
   ```

4. **Registration Functions:**
   ```c
   __attribute__((constructor))
   void __omp_offloading_descriptor_reg() {
     __tgt_register_lib(&__omp_offloading_descriptor);
   }

   __attribute__((destructor))
   void __omp_offloading_descriptor_unreg() {
     __tgt_unregister_lib(&__omp_offloading_descriptor);
   }
   ```

These constructors/destructors automatically register the device image with `libomptarget` when the program starts.

#### 2.5: Run Host Linker

**Function:** [`runLinker()`](https://github.com/llvm/llvm-project/blob/main/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp#L362)

Now that we have:
- Original host object files (e.g., `example_host.o`)
- Wrapped device images (e.g., `offload.wrapper.o` containing bundled device code + registration)

We invoke the standard host linker to create the final executable.

```cpp
Error runLinker(ArrayRef<StringRef> Files, const ArgList &Args) {
  // Get linker path from arguments
  StringRef LinkerPath = Args.getLastArgValue(OPT_linker_path_EQ);

  // Build argument list
  ArgStringList NewLinkerArgs;
  for (const opt::Arg *Arg : Args) {
    // Skip wrapper-only options
    if (Arg->getOption().hasFlag(WrapperOnlyOption))
      continue;

    Arg->render(Args, NewLinkerArgs);

    // Add wrapped device images after output argument
    if (Arg->getOption().matches(OPT_o))
      llvm::transform(Files, std::back_inserter(NewLinkerArgs),
                      [&](StringRef A) { return Args.MakeArgString(A); });
  }

  // Execute host linker
  executeCommands(LinkerPath, LinkerArgs);
}
```

**Effective command:**
```bash
ld.lld <standard args> \
  example.o \            # User's host code
  offload.wrapper.o \    # Wrapped device binary + registration code
  -lomptarget \          # Runtime library (for OpenMP)
  -o example
```

The final executable now contains:
- Host code
- Embedded fat device binary
- Registration code that runs at program startup

## The clang-linker-wrapper Tool

```cpp
int main(int Argc, char **Argv) {
  InitLLVM X(Argc, Argv);
  InitializeAllTargetInfos();
  InitializeAllTargets();

  // Parse arguments
  auto Args = Tbl.parseArgs(Argc, Argv, OPT_INVALID, Saver, ...);

  // Extract device input files from .llvm.offloading sections
  auto DeviceInputFiles = getDeviceInput(Args);

  // Link and wrap device images
  auto FilesOrErr = linkAndWrapDeviceFiles(*DeviceInputFiles, Args,
                                           Argv, Argc, !EmitFatbinOnly);

  // Run host linker with wrapped device objects
  runLinker(*FilesOrErr, Args);

  return EXIT_SUCCESS;
}
```

## Command Line Examples

### Basic OpenMP Offloading (NVIDIA)

```bash
# Single command (driver handles everything)
clang++ -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda \
  --offload-arch=sm_80 \
  example.cpp -o example

# Equivalent manual steps:
# 1. Compile to object with embedded device code
clang++ -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda \
  --offload-arch=sm_80 -c example.cpp -o example.o

# 2. Link (driver invokes clang-linker-wrapper)
clang++ -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda example.o -o example

# The driver actually runs something like:
clang-linker-wrapper \
  --host-triple=x86_64-unknown-linux-gnu \
  --linker-path=/usr/bin/ld.lld \
  -- \
  /usr/bin/ld.lld <args> example.o -lomptarget -o example
```

### Debugging with Save-Temps

```bash
# Save all intermediate files
clang++ -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda \
  --offload-arch=sm_80 \
  -Wl,--save-temps \
  example.cpp -o example

# This creates:
# - example.host.bc                    (host bitcode)
# - example-nvptx64-sm_80.bc          (device bitcode)
# - example-nvptx64-sm_80.o           (device object)
# - example-nvptx64-sm_80.img         (linked device image)
# - example.openmp.image.wrapper.bc   (wrapper IR)
# - example.openmp.image.wrapper.o    (wrapper object)
```

### Relocatable Linking

```bash
# Create relocatable object with embedded device code
clang++ -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda \
  --offload-arch=sm_80 \
  foo.cpp -c -o foo.o

# Perform device linking early and create merged object
clang++ -lomptarget.devicertl --offload-link -r foo.o -o merged.o

# Now merged.o can be distributed without requiring offload toolchain
ar rcs libfoo.a merged.o

# Link with regular compiler
g++ app.cpp -L. -lfoo -o app
```

### Override Device Image

```bash
# Normal compilation
clang++ -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda \
  -Wl,--save-temps \
  example.cpp -o example

# Modify the saved device IR
vim example-nvptx64-sm_80.bc  # (or use opt, llvm-link, etc.)

# Recompile device code
clang --target=nvptx64-nvidia-cuda -march=sm_80 \
  example-nvptx64-sm_80.bc -o modified.img

# Override with modified image
clang++ -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda \
  -Wl,--override-image=openmp=modified.img \
  example.cpp -o example
```

### Direct Fat Binary Output

```bash
# Create fat binary without host linking
clang++ -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda \
  --offload-arch=sm_80 \
  -Wl,--emit-fatbin-only \
  example.cpp -o example.fatbin

# This skips host linking and outputs the bundled device image directly
```

### Multiple Offload Kinds

```bash
# Hypothetical: OpenMP + CUDA in same binary
clang++ -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda \
  -x cuda --cuda-gpu-arch=sm_80 \
  mixed.cpp -o mixed

# The linker wrapper will:
# 1. Extract OpenMP device code
# 2. Extract CUDA device code
# 3. Link each separately
# 4. Wrap both with appropriate registration code
# 5. Link everything together
```

## References

### Documentation

- **Offloading Design:** [`clang/docs/OffloadingDesign.rst`](https://github.com/llvm/llvm-project/blob/main/clang/docs/OffloadingDesign.rst)
- **Linker Wrapper:** [`clang/docs/ClangLinkerWrapper.rst`](https://github.com/llvm/llvm-project/blob/main/clang/docs/ClangLinkerWrapper.rst)

### Source Code

#### Driver Integration
- **Clang Driver:** [`clang/lib/Driver/Driver.cpp`](https://github.com/llvm/llvm-project/blob/main/clang/lib/Driver/Driver.cpp) - Orchestrates compilation
- **Toolchain:** [`clang/lib/Driver/ToolChains/Clang.cpp`](https://github.com/llvm/llvm-project/blob/main/clang/lib/Driver/ToolChains/Clang.cpp) - Adds `-fembed-offload-object` flags

#### Linker Wrapper
- **Main Tool:** [`clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp`](https://github.com/llvm/llvm-project/blob/main/clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp)
- **Options:** [`clang/tools/clang-linker-wrapper/LinkerWrapperOpts.td`](https://github.com/llvm/llvm-project/blob/main/clang/tools/clang-linker-wrapper/LinkerWrapperOpts.td)

#### Binary Format
- **OffloadBinary:** [`llvm/include/llvm/Object/OffloadBinary.h`](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Object/OffloadBinary.h)
- **Implementation:** [`llvm/lib/Object/OffloadBinary.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Object/OffloadBinary.cpp)

#### Wrapper Generation
- **OpenMP Wrapper:** [`llvm/lib/Frontend/Offloading/OffloadWrapper.cpp`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Frontend/Offloading/OffloadWrapper.cpp)
- **Utility Functions:** [`llvm/include/llvm/Frontend/Offloading/Utility.h`](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Frontend/Offloading/Utility.h)

#### SYCL-Specific
- **SYCL Linker:** [`clang/tools/clang-sycl-linker/`](https://github.com/llvm/llvm-project/tree/main/clang/tools/clang-sycl-linker) (Intel fork)
- **SYCL Post Link:** [`llvm/tools/sycl-post-link/`](https://github.com/intel/llvm/tree/sycl/llvm/tools/sycl-post-link) (Intel fork)

#### Code Generation
- **OpenMP Codegen:** [`clang/lib/CodeGen/CGOpenMPRuntime.cpp`](https://github.com/llvm/llvm-project/blob/main/clang/lib/CodeGen/CGOpenMPRuntime.cpp)
- **CUDA Codegen:** [`clang/lib/CodeGen/CGCUDANV.cpp`](https://github.com/llvm/llvm-project/blob/main/clang/lib/CodeGen/CGCUDANV.cpp)

### Key Functions Reference

| Function | Location | Purpose |
|----------|----------|---------|
| `getDeviceInput()` | ClangLinkerWrapper.cpp | Extract device code from input files |
| `linkDevice()` | ClangLinkerWrapper.cpp | Link device code for target |
| `linkAndWrapDeviceFiles()` | ClangLinkerWrapper.cpp | Main processing loop |
| `wrapDeviceImages()` | ClangLinkerWrapper.cpp | Generate wrapper code |
| `bundleLinkedOutput()` | ClangLinkerWrapper.cpp | Create fat binary |
| `runLinker()` | ClangLinkerWrapper.cpp | Execute host linker |
| `extractOffloadBinaries()` | OffloadBinary.cpp | Parse `.llvm.offloading` section |
| `OffloadBinary::create()` | OffloadBinary.cpp | Deserialize OffloadBinary format |
| `OffloadBinary::write()` | OffloadBinary.cpp | Serialize OffloadBinary format |
| `wrapOpenMPBinaries()` | OffloadWrapper.cpp | Create OpenMP registration code |
| `wrapSYCLBinaries()` | OffloadWrapper.cpp | Create SYCL registration code |

### Related Tools

- **llvm-offload-binary:** Utility for creating/inspecting `OffloadBinary` files
- **llvm-objcopy:** Used for section manipulation in relocatable links
- **llvm-readelf:** Inspect `.llvm.offloading` sections
- **clang-offload-bundler:** Old model tool (deprecated)

