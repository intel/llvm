set(LLVM_LINK_COMPONENTS
  BinaryFormat
  BitWriter
  Core
  IRReader
  MC
  Object
  Option
  ProfileData
  Support
  TargetParser
  WindowsDriver
  )

if(WIN32)
  # MSVCToolChain.cpp uses version.dll.
  set(system_libs version)
endif()

# This must be in sync with llvm/sycl/CMakeLists.txt.
SET_SOURCE_FILES_PROPERTIES( ToolChains/MSVC.cpp ToolChains/Clang.cpp
  PROPERTIES COMPILE_DEFINITIONS SYCL_MAJOR_VERSION="8" )

add_clang_library(clangDriver
  Action.cpp
  Compilation.cpp
  Distro.cpp
  Driver.cpp
  DriverOptions.cpp
  Job.cpp
  Multilib.cpp
  MultilibBuilder.cpp
  OffloadBundler.cpp
  OptionUtils.cpp
  Phases.cpp
  SanitizerArgs.cpp
  Tool.cpp
  ToolChain.cpp
  ToolChains/Arch/AArch64.cpp
  ToolChains/Arch/ARM.cpp
  ToolChains/Arch/CSKY.cpp
  ToolChains/Arch/LoongArch.cpp
  ToolChains/Arch/M68k.cpp
  ToolChains/Arch/Mips.cpp
  ToolChains/Arch/PPC.cpp
  ToolChains/Arch/RISCV.cpp
  ToolChains/Arch/Sparc.cpp
  ToolChains/Arch/SystemZ.cpp
  ToolChains/Arch/VE.cpp
  ToolChains/Arch/X86.cpp
  ToolChains/AIX.cpp
  ToolChains/AMDGPU.cpp
  ToolChains/AMDGPUOpenMP.cpp
  ToolChains/AVR.cpp
  ToolChains/BareMetal.cpp
  ToolChains/Clang.cpp
  ToolChains/CommonArgs.cpp
  ToolChains/CrossWindows.cpp
  ToolChains/CSKYToolChain.cpp
  ToolChains/Cuda.cpp
  ToolChains/Cygwin.cpp
  ToolChains/Darwin.cpp
  ToolChains/DragonFly.cpp
  ToolChains/Flang.cpp
  ToolChains/FreeBSD.cpp
  ToolChains/Fuchsia.cpp
  ToolChains/Gnu.cpp
  ToolChains/Haiku.cpp
  ToolChains/HIPUtility.cpp
  ToolChains/HIPAMD.cpp
  ToolChains/HIPSPV.cpp
  ToolChains/Hexagon.cpp
  ToolChains/HLSL.cpp
  ToolChains/Hurd.cpp
  ToolChains/Linux.cpp
  ToolChains/Managarm.cpp
  ToolChains/MipsLinux.cpp
  ToolChains/MinGW.cpp
  ToolChains/MSP430.cpp
  ToolChains/MSVC.cpp
  ToolChains/NaCl.cpp
  ToolChains/NetBSD.cpp
  ToolChains/OHOS.cpp
  ToolChains/OpenBSD.cpp
  ToolChains/PS4CPU.cpp
  ToolChains/RISCVToolchain.cpp
  ToolChains/Solaris.cpp
  ToolChains/SPIRV.cpp
  ToolChains/SPIRVOpenMP.cpp
  ToolChains/SYCL.cpp
  ToolChains/TCE.cpp
  ToolChains/UEFI.cpp
  ToolChains/VEToolchain.cpp
  ToolChains/WebAssembly.cpp
  ToolChains/XCore.cpp
  ToolChains/SYCL.cpp
  ToolChains/PPCLinux.cpp
  ToolChains/PPCFreeBSD.cpp
  ToolChains/InterfaceStubs.cpp
  ToolChains/ZOS.cpp
  Types.cpp
  XRayArgs.cpp

  DEPENDS
  ClangDriverOptions
  DeviceConfigFile
  # These generated headers are included transitively.
  target_parser_gen

  LINK_LIBS
  clangBasic
  ${system_libs}
  )
