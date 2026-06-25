# Lifecycle of `--device-compiler` in clang-linker-wrapper

## What it is

`--device-compiler=` is a **wrapper-only option** for `clang-linker-wrapper`. It carries flags that should be forwarded to the *device* compilation (the per-target `clang` invocation that the wrapper spawns), without affecting the host link. It accepts two forms:

- `--device-compiler=<value>` → applies to **every** device target
- `--device-compiler=<triple>=<value>` → applies **only** to the matching target triple

The `clang++` driver populates it automatically.

---

## Stage 1 — User runs `clang++`

```bash
clang++ -fopenmp --offload-arch=gfx90a -Xoffload-compiler -ffast-math foo.cpp
```

`-Xoffload-compiler<triple> <arg>`:
Pass `<arg>` to the offload compilers or the ones identified by `-<triple>`

Open #1: is `Xsycl-target` intended for the same purpose as `-Xoffload-compiler`? Should we deprecate `Xsycl-target` in favor of `-Xoffload-compiler`?

Open #2: Per https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2025-1/xsycl-target.html , it seems `-Xsycl-target-backend`
is supposed to pass AOT/JIT compilation options. Is it the right direction to move parsing of `-Xsycl-target-backend` option from initial compilation step (at stage 2) to device compilation step (after `clang-linker-wrapper` called `clang++`)?

---

## Stage 2 — Driver constructs the `--device-compiler=` args

**`clang/lib/Driver/ToolChains/Clang.cpp`**, `LinkerWrapper::ConstructJob()`.

It loops over each offload kind and toolchain, collecting per-toolchain compiler args, then emits them prefixed with the target triple:

- **Forwarded compiler options**:
  ```cpp
  for (StringRef Arg : CompilerArgs)
    CmdArgs.push_back(Args.MakeArgString(
        "--device-compiler=" + TC->getTripleString() + "=" + Arg));
  ```
  
- **`-Xoffload-compiler` user passthrough** : translates `-Xoffload-compiler` into `--device-compiler=`, optionally prefixing a normalized triple.

Open #3: Today in intel/llvm we parse `-Xsycl-target-backend` at stage 2 and pass `ocloc`-specific options to `--device-compiler=`, which is not aligned with the rest of programming models, where normal compiler options are passed to ``--device-compiler=``

So the actual command the driver builds (visible with `clang++ -###`) looks like:

```
clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu \
  --linker-path=/usr/bin/ld \
  --device-compiler=nvptx64-nvidia-cuda=-g \
  --device-compiler=nvptx64-nvidia-cuda=-flto=full \
  --device-compiler=--cuda-path=/usr/local/cuda \
  <inputs> -o a.out
```

---

## Stage 3 — Option definition in the wrapper

**`clang/tools/clang-linker-wrapper/LinkerWrapperOpts.td`**:
```tablegen
def device_compiler_args_EQ : Joined<["--"], "device-compiler=">,
  Flags<[WrapperOnlyOption]>, MetaVarName<"<value> or <triple>=<value>">,
  HelpText<"Arguments to pass to the device compiler invocation">;
```
This generates the `OPT_device_compiler_args_EQ` enum used below.

---

## Stage 4 — Wrapper parses & filters per target

**`clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp`**, `getLinkerArgs()`. This builds a `DerivedArgList` *per device image*. The target triple/arch for the current image are set, then resolve the `--device-compiler=` values:

```cpp
for (StringRef Arg : Args.getAllArgValues(OPT_device_compiler_args_EQ)) {
  auto [Triple, Value] = Arg.split('=');
  llvm::Triple TT(Triple);
  if (TT.getArch() == Triple::ArchType::UnknownArch)        // no triple → global
    DAL.AddJoinedArg(nullptr, Tbl.getOption(OPT_compiler_arg_EQ),
                     Args.MakeArgString(Arg));
  else if (Value.empty())                                    // triple-shaped, no value
    DAL.AddJoinedArg(nullptr, Tbl.getOption(OPT_compiler_arg_EQ),
                     Args.MakeArgString(Triple));
  else if (Triple == DAL.getLastArgValue(OPT_triple_EQ))     // triple matches THIS target
    DAL.AddJoinedArg(nullptr, Tbl.getOption(OPT_compiler_arg_EQ),
                     Args.MakeArgString(Value));
}
```

Key point: `--device-compiler=` is *rewritten* into the internal device-only **`--compiler-arg=`** option (`OPT_compiler_arg_EQ`, defined `LinkerWrapperOpts.td`), keeping only the values relevant to the image being linked. The sibling `--device-linker=` is handled identically just above. `getLinkerArgs()` is called from the device-linking driver at `ClangLinkerWrapper.cpp`.

---

## Stage 5 — Wrapper spawns device `clang` with the flags

**`ClangLinkerWrapper.cpp`**, `generic::clang()` , reached via `linkDevice()`. The resolved `--compiler-arg=` values are appended to the device clang command at `611-614`:

```cpp
for (StringRef Arg : Args.getAllArgValues(OPT_linker_arg_EQ))
  CmdArgs.append({"-Xlinker", Args.MakeArgString(Arg)});
for (StringRef Arg : Args.getAllArgValues(OPT_compiler_arg_EQ))
  CmdArgs.push_back(Args.MakeArgString(Arg));
```

The command is then executed (`executeCommands()`), producing e.g.:
```
clang --target=nvptx64-nvidia-cuda -march=sm_70 ... -g -flto=full ...
```
---

## Stage 6 — What happens inside the spawned device `clang`

It starts its own driver pipeline: parse args → build a device toolchain → construct compile + link jobs → run them.

### The device clang driver pipeline

1. **Toolchain selection.** `--target=nvptx64-nvidia-cuda` → **`NVPTXToolChain`**
   (`clang/lib/Driver/ToolChains/Cuda.cpp` / `Cuda.h`). AMD target → `AMDGPUToolChain`;
   SPIR-V → `SPIRVToolChain`; SYCL takes the `--sycl-link` path

2. **Job construction.** The driver builds a link job whose tool is `clang-sycl-linker`

Open #4: This is where we could parse options and pass ocloc-specific options to like 
`clang-sycl-linker ocloc-options=...`

