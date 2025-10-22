

set(SYCL_JIT_RESOURCE_CPP "${CMAKE_CURRENT_BINARY_DIR}/resource.cpp")
set(SYCL_JIT_RESOURCE_OBJ "${CMAKE_CURRENT_BINARY_DIR}/resource.cpp.o")

if (WIN32)
set(SYCL_JIT_VIRTUAL_TOOLCHAIN_ROOT "c:/sycl-jit-toolchain/")
else()
set(SYCL_JIT_VIRTUAL_TOOLCHAIN_ROOT "/sycl-jit-toolchain/")
endif()

set(SYCL_JIT_RESOURCE_DEPS
  sycl-headers                # include/sycl
  clang                       # lib/clang/N/include
  opencl-resource-headers     # fixes build race. not actually used.
  ${CMAKE_CURRENT_SOURCE_DIR}/utils/generate.py)

if ("libclc" IN_LIST LLVM_ENABLE_PROJECTS)
  # Somehow just "libclc" doesn't build "remangled-*" (and maybe whatever else).
  list(APPEND SYCL_JIT_RESOURCE_DEPS libclc libspirv-builtins) # lib/clc/*.bc
endif()

if ("libdevice" IN_LIST LLVM_ENABLE_PROJECTS)
  list(APPEND SYCL_JIT_RESOURCE_DEPS libsycldevice) # lib/*.bc
endif()

add_custom_command(
  OUTPUT ${SYCL_JIT_RESOURCE_CPP}
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/utils/generate.py --toolchain-dir ${CMAKE_BINARY_DIR} --output ${SYCL_JIT_RESOURCE_CPP} --prefix ${SYCL_JIT_VIRTUAL_TOOLCHAIN_ROOT}
  DEPENDS
  ${SYCL_JIT_RESOURCE_DEPS}
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
  ${CMAKE_CURRENT_SOURCE_DIR}/include/Resource.h
)

set(SYCL_JIT_SOURCES_LIST
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
)

add_llvm_library(sycl-jit
   ${SYCL_JIT_SOURCES_LIST}

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


# These variables for the install script
set(PYTHON_EXE_FOR_INSTALL "${Python3_EXECUTABLE}")
set(GENERATE_SCRIPT_PATH "${CMAKE_CURRENT_LIST_DIR}/utils/generate.py")
set(SCRIPT_SOURCE_DIR_FOR_INSTALL "${CMAKE_CURRENT_LIST_DIR}")
set(COMPILER_PATH_FOR_INSTALL "${CMAKE_INSTALL_PREFIX}/bin/clang++")
set(LLVM_INCLUDE_DIR_FOR_INSTALL ${LLVM_MAIN_INCLUDE_DIR})
set(SPIRV_INCLUDE_DIRS_FOR_INSTALL ${LLVM_SPIRV_INCLUDE_DIRS})
set(CLANG_SOURCE_INCLUDE_DIR_FOR_INSTALL "${LLVM_EXTERNAL_CLANG_SOURCE_DIR}/include")
set(CLANG_BUILD_INCLUDE_DIR_FOR_INSTALL "${CMAKE_BINARY_DIR}/tools/clang/include")
set(SYCL_JIT_PASSES_DIR_FOR_INSTALL "${CMAKE_CURRENT_LIST_DIR}/../passes")
set(SYCL_JIT_LOCAL_INCLUDE_DIR_FOR_INSTALL "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(SYCL_JIT_LOCAL_LIB_DIR_FOR_INSTALL "${CMAKE_CURRENT_SOURCE_DIR}/lib")
set(LLVM_BUILD_INCLUDE_DIR_FOR_INSTALL "${LLVM_BINARY_DIR}/include")
set(JIT_INSTALL_COMPILE_FLAGS "-std=c++17;-fPIC;-Wno-c23-extensions") 
set(JIT_INSTALL_LINK_FLAGS    "-shared") # Add other link flags

# Create a new list  for the install script
set(JIT_SOURCES_FOR_INSTALL "")
foreach(src_file ${SYCL_JIT_SOURCES_LIST})
  list(APPEND JIT_SOURCES_FOR_INSTALL "${CMAKE_CURRENT_LIST_DIR}/${src_file}")
endforeach()

# Create a list of local include paths based on the source list
set(JIT_INCLUDE_DIRS_FOR_INSTALL "")
foreach(src_file ${SYCL_JIT_SOURCES_LIST})
  get_filename_component(src_dir "${CMAKE_CURRENT_LIST_DIR}/${src_file}" DIRECTORY)
  list(APPEND JIT_INCLUDE_DIRS_FOR_INSTALL ${src_dir})
endforeach()

list(APPEND JIT_INCLUDE_DIRS_FOR_INSTALL "${CMAKE_CURRENT_LIST_DIR}/include")

# Remove duplicates and pass to the script
list(REMOVE_DUPLICATES JIT_INCLUDE_DIRS_FOR_INSTALL)
set(JIT_INCLUDE_DIRS_STRING_LIST "${JIT_INCLUDE_DIRS_FOR_INSTALL}")
set(JIT_SOURCES_STRING_LIST "${JIT_SOURCES_FOR_INSTALL}")

set(INSTALL_SCRIPT_TEMPLATE "${CMAKE_CURRENT_LIST_DIR}/BuildSyclJitAtInstall.cmake.in")
set(INSTALL_SCRIPT_GENERATED "${CMAKE_CURRENT_BINARY_DIR}/BuildSyclJitAtInstall.cmake")

# Use configure_file to pass the variables
configure_file(
  ${INSTALL_SCRIPT_TEMPLATE}
  ${INSTALL_SCRIPT_GENERATED}
  @ONLY
)

install(SCRIPT "${INSTALL_SCRIPT_GENERATED}" COMPONENT sycl-jit)
set(CPACK_COMPONENT_SYCL_JIT_DEPENDS
  "sycl-jit-resources"
  # From LINK_COMPONENTS
  "BitReader"
  "BitWriter"
  "Core"
  "Support"
  "Option"
  "Analysis"
  "IPO"
  "TransformUtils"
  "Passes"
  "IRReader"
  "Linker"
  "ScalarOpts"
  "InstCombine"
  "Target"
  "TargetParser"
  "MC"
  "SYCLLowerIR"
  "SYCLPostLink"
  "SPIRVLib"
  ${LLVM_TARGETS_TO_BUILD}
  # From LINK_LIBS
  "clangBasic"
  "clangDriver"
  "clangFrontend"
  "clangCodeGen"
  "clangTooling"
  "clangSerialization"
  # Also add the main header component
  "LLVMHeaders"
  "clangHeaders"
  )
