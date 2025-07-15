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

macro(append_common_extra_security_flags)
  # Compiler Warnings and Error Detection
  # Note: in intel/llvm we build both linux and win with --ci-defaults.
  # This flag also enables -Werror or /WX.
  if(is_gcc
     OR is_clang
     OR (is_icpx AND MSVC))
    add_compile_option_ext("-Wall" WALL)
    add_compile_option_ext("-Wextra" WEXTRA)
  elseif(is_icpx)
    add_compile_option_ext("/Wall" WALL)
  elseif(is_msvc)
    add_compile_option_ext("/W4" WALL)
  endif()

  if(CMAKE_BUILD_TYPE MATCHES "Release")
    if(is_gcc
       OR is_clang
       OR (is_icpx AND MSVC))
      add_compile_option_ext("-Wconversion" WCONVERSION)
      add_compile_option_ext("-Wimplicit-fallthrough" WIMPLICITFALLTHROUGH)
    endif()
  endif()

  # Control Flow Integrity
  if(is_gcc
     OR is_clang
     OR (is_icpx AND MSVC))
    add_compile_option_ext("-fcf-protection=full" FCFPROTECTION)
  elseif(is_icpx)
    add_compile_option_ext("/Qcf-protection:full" FCFPROTECTION)
  elseif(is_msvc)
    add_link_option_ext("/LTCG" LTCG CMAKE_EXE_LINKER_FLAGS
                        CMAKE_MODULE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS)
    add_compile_option_ext("/sdl" SDL)
    add_compile_option_ext("/guard:cf" GUARDCF)
    add_link_option_ext("/CETCOMPAT" CETCOMPAT CMAKE_EXE_LINKER_FLAGS
                        CMAKE_MODULE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS)
  endif()

  # Format String Defense
  if(is_gcc
     OR is_clang
     OR (is_icpx AND MSVC))
    add_compile_option_ext("-Wformat" WFORMAT)
    add_compile_option_ext("-Wformat-security" WFORMATSECURITY)
  elseif(is_icpx)
    add_compile_option_ext("/Wformat" WFORMAT)
    add_compile_option_ext("/Wformat-security" WFORMATSECURITY)
  elseif(is_msvc)
    add_compile_option_ext("/analyze" ANALYZE)
  endif()

  if(CMAKE_BUILD_TYPE MATCHES "Release")
    if(is_gcc
       OR is_clang
       OR (is_icpx AND MSVC))
      add_compile_option_ext("-Werror=format-security" WERRORFORMATSECURITY)
    endif()
  endif()

  # Inexecutable Stack
  if(CMAKE_BUILD_TYPE MATCHES "Release")
    if(is_gcc
       OR is_clang
       OR (is_icpx AND MSVC))
      add_link_option_ext(
        "-Wl,-z,noexecstack" NOEXECSTACK CMAKE_EXE_LINKER_FLAGS
        CMAKE_MODULE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS)
    endif()
  endif()

  # Position Independent Code
  if(is_gcc
     OR is_clang
     OR (is_icpx AND MSVC))
    add_compile_option_ext("-fPIC" FPIC)
  elseif(is_msvc)
    add_compile_option_ext("/Gy" GY)
  endif()

  # Position Independent Execution
  # We rely on CMake to set the right -fPIE flags for us, but it must be
  # explicitly requested
  if (NOT CMAKE_POSITION_INDEPENDENT_CODE)
    message(FATAL_ERROR "To enable all necessary security flags, CMAKE_POSITION_INDEPENDENT_CODE must be set to ON")
  endif()

  if(is_msvc)
    add_link_option_ext("/DYNAMICBASE" DYNAMICBASE CMAKE_EXE_LINKER_FLAGS
                        CMAKE_MODULE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS)
  endif()

  if(CMAKE_BUILD_TYPE MATCHES "Release")
    if(is_msvc)
      add_link_option_ext("/NXCOMPAT" NXCOMPAT CMAKE_EXE_LINKER_FLAGS
                          CMAKE_MODULE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS)
    endif()
  endif()

  # Stack Protection
  if(is_msvc)
    add_compile_option_ext("/GS" GS)
  elseif(
    is_gcc
    OR is_clang
    OR (is_icpx AND MSVC))
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
      add_compile_option_ext("-fstack-protector" FSTACKPROTECTOR)
    elseif(CMAKE_BUILD_TYPE MATCHES "Release")
      add_compile_option_ext("-fstack-protector-strong" FSTACKPROTECTORSTRONG)
      add_compile_option_ext("-fstack-clash-protection" FSTACKCLASHPROTECTION)
    endif()
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

    # Full Relocation Read Only
    if(CMAKE_BUILD_TYPE MATCHES "Release")
      add_link_option_ext("-Wl,-z,relro" ZRELRO CMAKE_EXE_LINKER_FLAGS
                          CMAKE_MODULE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS)
    endif()

    # Immediate Binding (Bindnow)
    if(CMAKE_BUILD_TYPE MATCHES "Release")
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
