

set(SYCL_JIT_RESOURCE_CPP "${CMAKE_CURRENT_BINARY_DIR}/resource.cpp")
set(SYCL_JIT_RESOURCE_OBJ "${CMAKE_CURRENT_BINARY_DIR}/resource.cpp.o")

if (WIN32)
set(SYCL_JIT_VIRTUAL_TOOLCHAIN_ROOT "c:/sycl-jit-toolchain/")
else()
set(SYCL_JIT_VIRTUAL_TOOLCHAIN_ROOT "/sycl-jit-toolchain/")
endif()

set(SYCL_JIT_RESOURCE_INSTALL_COMPONENTS sycl-headers OpenCL-Headers clang-resource-headers)

if ("libclc" IN_LIST LLVM_ENABLE_PROJECTS)
  list(APPEND SYCL_JIT_RESOURCE_INSTALL_COMPONENTS libspirv-builtins)
endif()

if ("libdevice" IN_LIST LLVM_ENABLE_PROJECTS)
  list(APPEND SYCL_JIT_RESOURCE_INSTALL_COMPONENTS libsycldevice)
endif()

set(SYCL_JIT_RESOURCE_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/rtc-resources-install)

set(SYCL_JIT_PREPARE_RESOURCE_COMMANDS)
foreach(component IN LISTS SYCL_JIT_RESOURCE_INSTALL_COMPONENTS)
  list(APPEND SYCL_JIT_PREPARE_RESOURCE_COMMANDS
    COMMAND ${CMAKE_COMMAND} --install ${CMAKE_BINARY_DIR} --prefix ${SYCL_JIT_RESOURCE_INSTALL_DIR} --component "${component}"
  )
endforeach()

set(SYCL_JIT_RESOURCE_DEPS ${SYCL_JIT_RESOURCE_INSTALL_COMPONENTS})
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
  OUTPUT ${SYCL_JIT_RESOURCE_CPP}
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/utils/generate.py --toolchain-dir ${SYCL_JIT_RESOURCE_INSTALL_DIR} --output ${SYCL_JIT_RESOURCE_CPP} --prefix ${SYCL_JIT_VIRTUAL_TOOLCHAIN_ROOT}
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

   LINK_LIBS
   clangBasic
   clangDriver
   clangFrontend
   clangCodeGen
   clangTooling
   clangSerialization
)

if(WIN32)
  target_link_libraries(sycl-jit PRIVATE Shlwapi)
endif()

target_compile_options(sycl-jit PRIVATE ${SYCL_JIT_WARNING_FLAGS})

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
