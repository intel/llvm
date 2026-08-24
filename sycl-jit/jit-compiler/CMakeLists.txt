

set(SYCL_JIT_RESOURCE_CPP "${CMAKE_CURRENT_BINARY_DIR}/resource.cpp")
set(SYCL_JIT_RESOURCE_OBJ "${CMAKE_CURRENT_BINARY_DIR}/resource.cpp.o")
set(SYCL_JIT_RESOURCE_SHARD_DIR "${CMAKE_CURRENT_BINARY_DIR}/resource-shards")
set(SYCL_JIT_RESOURCE_NUM_SHARDS 16)
# `generate.py` packs #embed'd files into fixed resource-shard-N.cpp units for
# parallel compilation. This count allows path declaration at configure time.
set(SYCL_JIT_RESOURCE_SHARD_CPPS)
math(EXPR SYCL_JIT_RESOURCE_LAST_SHARD "${SYCL_JIT_RESOURCE_NUM_SHARDS} - 1")
foreach(shard_index RANGE 0 ${SYCL_JIT_RESOURCE_LAST_SHARD})
  list(APPEND SYCL_JIT_RESOURCE_SHARD_CPPS "${SYCL_JIT_RESOURCE_SHARD_DIR}/resource-shard-${shard_index}.cpp")
endforeach()

if (WIN32)
set(SYCL_JIT_VIRTUAL_TOOLCHAIN_ROOT "c:/sycl-jit-toolchain/")
else()
set(SYCL_JIT_VIRTUAL_TOOLCHAIN_ROOT "/sycl-jit-toolchain/")
endif()


if (NOT "libdevice" IN_LIST LLVM_ENABLE_PROJECTS)
  message(FATAL_ERROR "libdevice is required for jit-compiler but not present in LLVM_ENABLE_PROJECTS")
endif()

set(SYCL_JIT_RESOURCE_INSTALL_COMPONENTS
  sycl-headers
  OpenCL-Headers
  clang-resource-headers
  libsycldevice)

set(libclc_enabled FALSE)
foreach(target IN LISTS LLVM_RUNTIME_TARGETS)
  if("libclc" IN_LIST RUNTIMES_${target}_LLVM_ENABLE_RUNTIMES)
    set(libclc_enabled TRUE)
    break()
  endif()
endforeach()

if (libclc_enabled)
  # If some targets required `libclc` then we should embed it for the
  # `sycl-jit`.
  list(APPEND SYCL_JIT_RESOURCE_INSTALL_COMPONENTS libspirv-builtins)
endif()

set(SYCL_JIT_RESOURCE_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/rtc-resources-install)

# Multi-version RTC toolchain. When enabled, the embedded toolchain holds one
# subtree per SYCL version -- the live tree under "latest/" plus a frozen
# subtree for each sycl-toolchains/sycl-v-* snapshot -- so a single
# libsycl-jit.so can compile against multiple SYCL versions (selected at runtime
# via JIT_MULTIVERSION_TOOLCHAIN in lib/rtc/DeviceCompilation.cpp). When disabled
# (default), the live tree is staged at the root exactly as before, and nothing
# about the produced library changes.
option(SYCL_JIT_EMBED_LEGACY_TOOLCHAINS
  "Embed frozen sycl-toolchains/sycl-v-* snapshots into libsycl-jit for multi-version RTC"
  OFF)
if(SYCL_JIT_EMBED_LEGACY_TOOLCHAINS)
  set(SYCL_JIT_LATEST_PREFIX "${SYCL_JIT_RESOURCE_INSTALL_DIR}/latest")
  file(GLOB SYCL_JIT_LEGACY_VERSION_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}/../sycl-toolchains/sycl-v-*")
else()
  set(SYCL_JIT_LATEST_PREFIX "${SYCL_JIT_RESOURCE_INSTALL_DIR}")
  set(SYCL_JIT_LEGACY_VERSION_DIRS)
endif()

set(SYCL_JIT_PREPARE_RESOURCE_COMMANDS)

function(sycl_jit_add_install_command component binary_dir prefix)
  list(APPEND SYCL_JIT_PREPARE_RESOURCE_COMMANDS
    COMMAND ${CMAKE_COMMAND} --install ${binary_dir} --prefix ${prefix} --component "${component}"
  )
  set(SYCL_JIT_PREPARE_RESOURCE_COMMANDS "${SYCL_JIT_PREPARE_RESOURCE_COMMANDS}" PARENT_SCOPE)
endfunction()

foreach(component IN LISTS SYCL_JIT_RESOURCE_INSTALL_COMPONENTS)
  if("${component}" STREQUAL "libspirv-builtins")
    list(APPEND SYCL_JIT_RESOURCE_DEPS libclc)
    foreach(tgt IN LISTS LLVM_RUNTIME_TARGETS)
      if("libclc" IN_LIST RUNTIMES_${tgt}_LLVM_ENABLE_RUNTIMES)
        sycl_jit_add_install_command("${component}" "${CMAKE_BINARY_DIR}/runtimes/runtimes-${tgt}-bins/libclc" "${SYCL_JIT_LATEST_PREFIX}")
      endif()
    endforeach()
  else()
    list(APPEND SYCL_JIT_RESOURCE_DEPS ${component})
    sycl_jit_add_install_command("${component}" "${CMAKE_BINARY_DIR}" "${SYCL_JIT_LATEST_PREFIX}")
  endif()
endforeach()

# Stage each frozen snapshot under its own "<install>/sycl-v-<label>/" subtree:
# the vendored headers (sycl/CL/std/unified-runtime + regenerated headers), the
# shared new-compiler resource + OpenCL headers, and that version's device
# libraries built from its OWN libdevice sources by the new clang. The device
# libraries keep their upstream names (libsycl-crt.bc, libsycl-imf.bc, ...), so
# the RTC loads them exactly as it loads the live ones -- no runtime change
# beyond the version-aware toolchain prefix. libdevice is thus pinned to the
# matching header version (sharing one libdevice across versions would break if
# a symbol's semantics changed or an implementation was removed between them).
# generate.py then embeds every subtree; the runtime selects among them by
# version. The live "latest" tree needs nothing extra here -- its libsycl-*.bc
# are already installed under latest/lib.
if(SYCL_JIT_EMBED_LEGACY_TOOLCHAINS)
  get_host_tool_path(clang CLANG SYCL_JIT_LD_CLANG _cjit_ld_ct)
  get_host_tool_path(llvm-link LLVM_LINK SYCL_JIT_LD_LINK _cjit_ld_lt)
  list(APPEND SYCL_JIT_RESOURCE_DEPS clang llvm-link)
  set(SYCL_JIT_VERSION_LIBDEV
    "${CMAKE_CURRENT_SOURCE_DIR}/../sycl-toolchains/build-version-libdevice.sh")

  foreach(vdir IN LISTS SYCL_JIT_LEGACY_VERSION_DIRS)
    get_filename_component(vname "${vdir}" NAME)
    set(vprefix "${SYCL_JIT_RESOURCE_INSTALL_DIR}/${vname}")
    list(APPEND SYCL_JIT_PREPARE_RESOURCE_COMMANDS
      COMMAND ${CMAKE_COMMAND} -E copy_directory "${vdir}/include" "${vprefix}/include")
    sycl_jit_add_install_command("clang-resource-headers" "${CMAKE_BINARY_DIR}" "${vprefix}")
    sycl_jit_add_install_command("OpenCL-Headers" "${CMAKE_BINARY_DIR}" "${vprefix}")
    # Build this version's device libraries (libsycl-*.bc) from its vendored
    # sources into <vprefix>/lib/dpcpp-<major>/sycl/. The script derives the
    # dpcpp-<major> subdir from the staged latest tree; all shell logic lives in
    # the script so the ninja rule carries no '$' to escape.
    list(APPEND SYCL_JIT_PREPARE_RESOURCE_COMMANDS
      COMMAND bash ${SYCL_JIT_VERSION_LIBDEV} ${SYCL_JIT_LD_CLANG} ${SYCL_JIT_LD_LINK} ${vprefix} ${SYCL_JIT_LATEST_PREFIX} ${vdir}/libdevice)
  endforeach()
endif()

  # OpenCL-Headers doesn't have a corresponding build target:
list(FILTER SYCL_JIT_RESOURCE_DEPS EXCLUDE REGEX "^OpenCL-Headers$")

# This is very hacky and I don't quite know what I'm doing, but it's necessary
# to have `resource.cpp` re-generated/re-built when some SYCL header changes.
#
# Inspired by the way `sycl-headers` target is created.
file(GLOB_RECURSE SYCL_JIT_RESOURCE_FILES CONFIGURE_DEPENDS "${SYCL_JIT_RESOURCE_INSTALL_DIR}/*" )

add_custom_target(rtc-prepare-resources
  DEPENDS ${SYCL_JIT_RESOURCE_DEPS}
  ${SYCL_JIT_PREPARE_RESOURCE_COMMANDS}
  BYPRODUCTS
  ${SYCL_JIT_RESOURCE_FILES}
)

add_custom_command(
  OUTPUT ${SYCL_JIT_RESOURCE_CPP} ${SYCL_JIT_RESOURCE_SHARD_CPPS}
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/utils/generate.py --toolchain-dir ${SYCL_JIT_RESOURCE_INSTALL_DIR} --output ${SYCL_JIT_RESOURCE_CPP} --prefix ${SYCL_JIT_VIRTUAL_TOOLCHAIN_ROOT} --shard-dir ${SYCL_JIT_RESOURCE_SHARD_DIR} --num-shards ${SYCL_JIT_RESOURCE_NUM_SHARDS}
  DEPENDS
  rtc-prepare-resources
  ${SYCL_JIT_RESOURCE_DEPS}
  ${SYCL_JIT_RESOURCE_FILES}
  ${CMAKE_CURRENT_SOURCE_DIR}/utils/generate.py
)

# We use C23/C++26's `#embed` to implement this resource creation, and "current"
# CMAKE_CXX_COMPILER might not have support for it. As such, use freshly built
# `clang++` instead.
if (WIN32)
  set(clang_exe ${CMAKE_BINARY_DIR}/bin/clang-cl.exe)
  set(SYCL_JIT_RESOURCE_CXX_FLAGS /O2 /std:c++17 /W0)

  # Determine if we should use static (/MT) or dynamic (/MD) runtime
  set(USE_STATIC_RUNTIME 0)
  if(CMAKE_MSVC_RUNTIME_LIBRARY AND NOT CMAKE_MSVC_RUNTIME_LIBRARY MATCHES "DLL")
    set(USE_STATIC_RUNTIME 1)
  endif()

  list(APPEND SYCL_JIT_RESOURCE_CXX_FLAGS /M$<IF:${USE_STATIC_RUNTIME},T,D>$<$<CONFIG:Debug>:d>)
else()
  get_host_tool_path( clang CLANG clang_exe clang_target )
  set(SYCL_JIT_RESOURCE_CXX_FLAGS -O2 -Wno-c23-extensions -std=c++17 -fPIC -fvisibility=hidden)

  if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
    # Hack based on the information at
    # https://cmake.org/cmake/help/latest/variable/CMAKE_OSX_SYSROOT.html
    execute_process(COMMAND xcrun --sdk macosx --show-sdk-path OUTPUT_VARIABLE SYCL_JIT_OSX_SYSROOT)
    list(APPEND SYCL_JIT_RESOURCE_CXX_FLAGS -isysroot ${SYCL_JIT_OSX_SYSROOT})
  endif()
endif()

# Compile each of the fixed set of resource-shard-N.cpp files (declared above).
set(SYCL_JIT_RESOURCE_SHARD_OBJS)
foreach(shard_cpp IN LISTS SYCL_JIT_RESOURCE_SHARD_CPPS)
  set(shard_obj "${shard_cpp}.o")
  add_custom_command(
    OUTPUT ${shard_obj}
    COMMAND
    ${clang_exe} --target=${LLVM_HOST_TRIPLE} ${shard_cpp} -I ${CMAKE_CURRENT_SOURCE_DIR}/include -c -o ${shard_obj} ${SYCL_JIT_RESOURCE_CXX_FLAGS}
    DEPENDS
    ${shard_cpp}
    ${SYCL_JIT_RESOURCE_CPP}
  )
  list(APPEND SYCL_JIT_RESOURCE_SHARD_OBJS ${shard_obj})
endforeach()

add_custom_command(
  OUTPUT ${SYCL_JIT_RESOURCE_OBJ}
  COMMAND
  # ${clang_exe}'s default target is not necessarily ${LLVM_HOST_TRIPLE}: when
  # cross compiling, it will be whatever the host tools were configured with,
  # and when building a cross compiler, it will be
  # ${LLVM_DEFAULT_TARGET_TRIPLE}. Rather than special casing these, just always
  # specify --target=${LLVM_HOST_TRIPLE}.
  ${clang_exe} --target=${LLVM_HOST_TRIPLE} ${SYCL_JIT_RESOURCE_CPP} -I ${CMAKE_CURRENT_SOURCE_DIR}/include -c -o ${SYCL_JIT_RESOURCE_OBJ} ${SYCL_JIT_RESOURCE_CXX_FLAGS}
  DEPENDS
  ${SYCL_JIT_RESOURCE_CPP}
  ${SYCL_JIT_RESOURCE_DEPS}
  ${SYCL_JIT_RESOURCE_FILES}
  ${CMAKE_CURRENT_SOURCE_DIR}/include/Resource.h
)

add_llvm_library(sycl-jit
   lib/translation/JITContext.cpp
   lib/translation/SPIRVLLVMTranslation.cpp
   lib/translation/Translation.cpp
   lib/materializer/MaterializerPipeline.cpp
   lib/materializer/Materializer.cpp
   lib/rtc/DeviceCompilation.cpp
   lib/rtc/ESIMD.cpp
   lib/rtc/RTC.cpp
   lib/helper/ConfigHelper.cpp
   lib/helper/ErrorHelper.cpp

   ${SYCL_JIT_RESOURCE_OBJ}
   ${SYCL_JIT_RESOURCE_SHARD_OBJS}

   SHARED

   DEPENDS
   intrinsics_gen

   LINK_COMPONENTS
   BitReader
   BitWriter
   Core
   Support
   Option
   Analysis
   IPO
   TransformUtils
   Passes
   IRReader
   Linker
   ScalarOpts
   InstCombine
   Target
   TargetParser
   MC
   SYCLLowerIR
   SYCLPostLink
   SPIRVLib
   ${LLVM_TARGETS_TO_BUILD}

# INTEL_CUSTOMIZATION
# Reusing PCH for Core would cause extra warnings in ADT
  DISABLE_PCH_REUSE
#end INTEL_CUSTOMIZATION

   LINK_LIBS
   clangBasic
   clangDriver
   clangOptions
   clangFrontend
   clangCodeGen
   clangTooling
   clangSerialization
)

if(WIN32)
  target_link_libraries(sycl-jit PRIVATE Shlwapi)
endif()

sycl_jit_apply_target_warning_flags(sycl-jit)

# Mark LLVM and SPIR-V headers as system headers to ignore warnigns in them.
# This classification remains intact even if the same paths are added as normal
# include paths in GCC and Clang.
target_include_directories(sycl-jit
  SYSTEM PRIVATE
  ${LLVM_MAIN_INCLUDE_DIR}
  ${LLVM_SPIRV_INCLUDE_DIRS}
  ${LLVM_EXTERNAL_CLANG_SOURCE_DIR}/include
  ${CMAKE_BINARY_DIR}/tools/clang/include
)
target_include_directories(sycl-jit
  PUBLIC
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/lib
)

find_package(Threads REQUIRED)

target_link_libraries(sycl-jit
  PRIVATE
  SYCLJITPasses
  ${CMAKE_THREAD_LIBS_INIT}
)

add_dependencies(sycl-jit sycl-headers)

if(SYCL_JIT_EMBED_LEGACY_TOOLCHAINS)
  # Enable the version-aware toolchain-root selection in DeviceCompilation.cpp.
  # Kept in lockstep with the staging layout above so the default build is
  # byte-for-byte unchanged.
  target_compile_definitions(sycl-jit PRIVATE JIT_MULTIVERSION_TOOLCHAIN)
endif()

if("NVPTX" IN_LIST LLVM_TARGETS_TO_BUILD)
  target_compile_definitions(sycl-jit PRIVATE JIT_SUPPORT_PTX)
endif()

if("AMDGPU" IN_LIST LLVM_TARGETS_TO_BUILD)
  target_compile_definitions(sycl-jit PRIVATE JIT_SUPPORT_AMDGCN)
endif()

if(NOT MSVC AND NOT APPLE)
  # Manage symbol visibility through the linker to make sure no LLVM symbols
  # are exported and confuse the drivers.
  set(linker_script "${CMAKE_CURRENT_SOURCE_DIR}/ld-version-script.txt")
  target_link_libraries(
    sycl-jit PRIVATE "-Wl,--version-script=${linker_script}")
  set_target_properties(sycl-jit PROPERTIES LINK_DEPENDS ${linker_script})
endif()
