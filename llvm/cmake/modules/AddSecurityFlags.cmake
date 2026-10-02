# Security hardening flags based on the Intel Secure Coding Standards for C/C++
# compilers. Enabled with -DEXTRA_SECURITY_FLAGS=default|sanitize and applied to
# the whole LLVM build through the global CMAKE_<LANG>_FLAGS and
# CMAKE_<TYPE>_LINKER_FLAGS.
#
# Notation used below:
#
#   GCC, Clang, icpx - compilers with GCC-style command line: GCC, Clang
#                      (including clang-cl) and icpx on Linux.
#   icx              - Intel compiler on Windows (cl-style command line).
#   MSVC             - Microsoft cl.exe.
#
# Added in both "default" and "sanitize" modes:
#
#   Control Flow Integrity (all builds):
#     GCC, Clang, icpx: -fcf-protection=full
#     icx:              /Qcf-protection:full
#     MSVC:             /guard:cf; linker /LTCG /CETCOMPAT
#
#   Format String Defense:
#     GCC, Clang, icpx: -Wformat -Wformat-security (all builds);
#                       -Werror=format-security (Release)
#     icx:              /Wformat /Wformat-security (all builds)
#     MSVC:             nothing
#
#   Inexecutable Stack:
#     GCC, Clang, icpx: linker -z noexecstack (Release)
#     icx, MSVC:        nothing
#
#   Position Independent Code (all builds):
#     GCC, Clang, icpx: -fPIC
#     MSVC:             /Gy
#     icx:              nothing
#
#   Position Independent Execution:
#     All compilers:    CMAKE_POSITION_INDEPENDENT_CODE must be ON, otherwise
#                       configuration fails; CMake then adds -fPIE/-pie for
#                       executables where supported.
#     MSVC:             linker /DYNAMICBASE (all builds), /NXCOMPAT (Release)
#
#   Stack Protection:
#     GCC, Clang, icpx: -fstack-protector (Debug);
#                       -fstack-protector-strong -fstack-clash-protection
#                       (Release)
#     MSVC:             /GS (all builds)
#     icx:              nothing
#
#   Pre-processor Macros (all compilers, non-Windows hosts only):
#     -D_FORTIFY_SOURCE=3 (=2 for GCC < 12) in all builds except Debug and
#     LLVM_USE_SANITIZER builds.
#     -D_GLIBCXX_ASSERTIONS in all builds if LLVM_ENABLE_ASSERTIONS is ON.
#   Read-only Relocation (all compilers, Unix hosts only, Release):
#     linker -z relro -z now
#
# Added only in "sanitize" mode (all builds):
#   Clang (including clang-cl): -fsanitize=cfi (compile and link)
#   Other compilers:            -fcf-protection=full -mcet (compile and link)
#
# Recommended by the standard, but not added here:
#   -Wall -Wextra -Wimplicit-fallthrough (GCC, Clang, icpx), /W4 (MSVC):
#     already added by HandleLLVMOptions (LLVM_ENABLE_WARNINGS=ON by default),
#     followed by -Wno-* / -wd* suppressions. This file is included after them,
#     so re-adding the flags here would re-enable the suppressed warnings for
#     Clang, icpx and MSVC and break the -Werror build.
#   -Wconversion (GCC, Clang, icpx): hundreds of warnings in the codebase, the
#     build fails under -Werror.
#   /Wall (icx), /sdl and /analyze (MSVC): the codebase does not build cleanly
#     with them under /WX, which --ci-defaults enables.
#   Spectre mitigations: -mfunction-return=thunk -mindirect-branch=thunk
#     -mindirect-branch-register (GCC), -mretpoline (Clang, icpx), /mretpoline
#     /Qspectre (icx, MSVC): significant performance impact.
#   -Wl,-z,nodlopen (GCC, Clang, icpx): UR adapters and sycl-jit are loaded
#     with dlopen.

macro(add_compile_option_ext flag name)
  cmake_parse_arguments(ARG "" "" "" ${ARGN})
  set(CHECK_STRING "${flag}")
  if(MSVC)
    set(CHECK_STRING "/WX ${CHECK_STRING}")
  else()
    set(CHECK_STRING "-Werror ${CHECK_STRING}")
  endif()

  check_c_compiler_flag("${CHECK_STRING}" "C_SUPPORTS_${name}")
  check_cxx_compiler_flag("${CHECK_STRING}" "CXX_SUPPORTS_${name}")
  if(C_SUPPORTS_${name} AND CXX_SUPPORTS_${name})
    message(STATUS "Building with ${flag}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flag}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${flag}")
    set(CMAKE_ASM_FLAGS "${CMAKE_ASM_FLAGS} ${flag}")
  else()
    message(WARNING "${flag} is not supported.")
  endif()
endmacro()

macro(add_link_option_ext flag name)
  include(CheckLinkerFlag)
  cmake_parse_arguments(ARG "" "" "" ${ARGN})
  check_linker_flag(CXX "${flag}" "LINKER_SUPPORTS_${name}")
  if(LINKER_SUPPORTS_${name})
    message(STATUS "Building with ${flag}")
    append("${flag}" ${ARG_UNPARSED_ARGUMENTS})
  else()
    message(WARNING "${flag} is not supported.")
  endif()
endmacro()

set(is_gcc FALSE)
set(is_clang FALSE)
set(is_msvc FALSE)
set(is_icpx FALSE)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(is_clang TRUE)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  set(is_gcc TRUE)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
  set(is_icpx TRUE)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  set(is_msvc TRUE)
endif()

# Compilers with GCC-style command line: gcc, clang, icpx on Linux.
set(is_gnu_like FALSE)
if(is_gcc
   OR is_clang
   OR (is_icpx AND NOT MSVC))
  set(is_gnu_like TRUE)
endif()

# Intel compiler with cl-style command line: icx on Windows.
set(is_icx_cl FALSE)
if(is_icpx AND MSVC)
  set(is_icx_cl TRUE)
endif()

set(is_release FALSE)
if(CMAKE_BUILD_TYPE MATCHES "Release")
  set(is_release TRUE)
endif()

macro(append_common_extra_security_flags)
  # Compiler Warnings and Error Detection: not added here, see the summary at the
  # top of the file.

  # Control Flow Integrity
  if(is_gnu_like)
    add_compile_option_ext("-fcf-protection=full" FCFPROTECTION)
  elseif(is_icx_cl)
    add_compile_option_ext("/Qcf-protection:full" FCFPROTECTION)
  elseif(is_msvc)
    add_link_option_ext("/LTCG" LTCG CMAKE_EXE_LINKER_FLAGS
                        CMAKE_MODULE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS)
    add_compile_option_ext("/guard:cf" GUARDCF)
    add_link_option_ext("/CETCOMPAT" CETCOMPAT CMAKE_EXE_LINKER_FLAGS
                        CMAKE_MODULE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS)
  endif()

  # Format String Defense
  if(is_gnu_like)
    add_compile_option_ext("-Wformat" WFORMAT)
    add_compile_option_ext("-Wformat-security" WFORMATSECURITY)
    if(is_release)
      add_compile_option_ext("-Werror=format-security" WERRORFORMATSECURITY)
    endif()
  elseif(is_icx_cl)
    add_compile_option_ext("/Wformat" WFORMAT)
    add_compile_option_ext("/Wformat-security" WFORMATSECURITY)
  endif()

  # Inexecutable Stack
  if(is_gnu_like AND is_release)
    add_link_option_ext(
      "-Wl,-z,noexecstack" NOEXECSTACK CMAKE_EXE_LINKER_FLAGS
      CMAKE_MODULE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS)
  endif()

  # Position Independent Code
  if(is_gnu_like)
    add_compile_option_ext("-fPIC" FPIC)
  elseif(is_msvc)
    add_compile_option_ext("/Gy" GY)
  endif()

  # Position Independent Execution
  # We rely on CMake to set the right -fPIE flags for us, but it must be
  # explicitly requested
  if (CMAKE_POSITION_INDEPENDENT_CODE)
    include(CheckPIESupported)
    check_pie_supported()
  else()
    message(FATAL_ERROR "To enable all necessary security flags, CMAKE_POSITION_INDEPENDENT_CODE must be set to ON")
  endif()

  if(is_msvc)
    add_link_option_ext("/DYNAMICBASE" DYNAMICBASE CMAKE_EXE_LINKER_FLAGS
                        CMAKE_MODULE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS)
    if(is_release)
      add_link_option_ext("/NXCOMPAT" NXCOMPAT CMAKE_EXE_LINKER_FLAGS
                          CMAKE_MODULE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS)
    endif()
  endif()

  # Stack Protection
  if(is_gnu_like)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
      add_compile_option_ext("-fstack-protector" FSTACKPROTECTOR)
    elseif(is_release)
      add_compile_option_ext("-fstack-protector-strong" FSTACKPROTECTORSTRONG)
      add_compile_option_ext("-fstack-clash-protection" FSTACKCLASHPROTECTION)
    endif()
  elseif(is_msvc)
    add_compile_option_ext("/GS" GS)
  endif()

  # Fortify Source (strongly recommended):
  if (NOT WIN32)
    # Strictly speaking, _FORTIFY_SOURCE is a glibc feature and not a compiler
    # feature. However, we experienced some issues (warnings about redefined macro
    # which are problematic under -Werror) when setting it to value '3' with older
    # gcc versions. Hence the check.
    # Value '3' became supported in glibc somewhere around gcc 12, so that is
    # what we are looking for.
    if (is_gcc AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 12)
      set(FORTIFY_SOURCE "-D_FORTIFY_SOURCE=2")
    else()
      # Assuming that the problem is not reproducible with other compilers
      set(FORTIFY_SOURCE "-D_FORTIFY_SOURCE=3")
    endif()

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
      message(WARNING "${FORTIFY_SOURCE} can only be used with optimization.")
      message(WARNING "${FORTIFY_SOURCE} is not supported.")
    else()
      # Sanitizers do not work with checked memory functions, such as
      # __memset_chk. We do not build release packages with sanitizers, so just
      # avoid -D_FORTIFY_SOURCE=N under LLVM_USE_SANITIZER.
      if(NOT LLVM_USE_SANITIZER)
        message(STATUS "Building with ${FORTIFY_SOURCE}")
        add_definitions(${FORTIFY_SOURCE})
      else()
        message(
          WARNING "${FORTIFY_SOURCE} dropped due to LLVM_USE_SANITIZER.")
      endif()
    endif()
  endif()

  if(LLVM_ON_UNIX)
    if(LLVM_ENABLE_ASSERTIONS)
      add_definitions(-D_GLIBCXX_ASSERTIONS)
    endif()

    if(is_release)
      # Full Relocation Read Only
      add_link_option_ext("-Wl,-z,relro" ZRELRO CMAKE_EXE_LINKER_FLAGS
                          CMAKE_MODULE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS)
      # Immediate Binding (Bindnow)
      add_link_option_ext("-Wl,-z,now" ZNOW CMAKE_EXE_LINKER_FLAGS
                          CMAKE_MODULE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS)
    endif()
  endif()
endmacro()

if(EXTRA_SECURITY_FLAGS)
  if(EXTRA_SECURITY_FLAGS STREQUAL "none")
    # No actions.
  elseif(EXTRA_SECURITY_FLAGS STREQUAL "default")
    append_common_extra_security_flags()
  elseif(EXTRA_SECURITY_FLAGS STREQUAL "sanitize")
    append_common_extra_security_flags()
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      add_compile_option_ext("-fsanitize=cfi" FSANITIZE_CFI)
      add_link_option_ext(
        "-fsanitize=cfi" FSANITIZE_CFI_LINK CMAKE_EXE_LINKER_FLAGS
        CMAKE_MODULE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS)
      # Recommended option although linking a DSO with SafeStack is not
      # currently supported by compiler.
      # add_compile_option_ext("-fsanitize=safe-stack" FSANITIZE_SAFESTACK)
      # add_link_option_ext("-fsanitize=safe-stack" FSANITIZE_SAFESTACK_LINK
      # CMAKE_EXE_LINKER_FLAGS CMAKE_MODULE_LINKER_FLAGS
      # CMAKE_SHARED_LINKER_FLAGS)
    else()
      add_compile_option_ext("-fcf-protection=full -mcet" FCF_PROTECTION)
      # need to align compile and link option set, link now is set
      # unconditionally
      add_link_option_ext(
        "-fcf-protection=full -mcet" FCF_PROTECTION_LINK CMAKE_EXE_LINKER_FLAGS
        CMAKE_MODULE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS)
    endif()
  else()
    message(
      FATAL_ERROR
        "Unsupported value of EXTRA_SECURITY_FLAGS: ${EXTRA_SECURITY_FLAGS}")
  endif()
endif()
