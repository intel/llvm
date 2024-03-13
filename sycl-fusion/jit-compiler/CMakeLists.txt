
add_llvm_library(sycl-fusion
   lib/KernelFusion.cpp
   lib/translation/KernelTranslation.cpp
   lib/translation/SPIRVLLVMTranslation.cpp
   lib/fusion/FusionPipeline.cpp
   lib/fusion/FusionHelper.cpp
   lib/fusion/JITContext.cpp
   lib/fusion/ModuleHelper.cpp
   lib/helper/ConfigHelper.cpp

   DEPENDS
   intrinsics_gen

   LINK_COMPONENTS
   BitReader
   Core
   Support
   Analysis
   IPO
   TransformUtils
   Passes
   Linker
   ScalarOpts
   InstCombine
   Target
   TargetParser
   MC
   ${LLVM_TARGETS_TO_BUILD}
)

target_compile_options(sycl-fusion PRIVATE ${SYCL_FUSION_WARNING_FLAGS})

# Mark LLVM and SPIR-V headers as system headers to ignore warnigns in them.
# This classification remains intact even if the same paths are added as normal
# include paths in GCC and Clang.
target_include_directories(sycl-fusion
  SYSTEM PRIVATE
  ${LLVM_MAIN_INCLUDE_DIR}
  ${LLVM_SPIRV_INCLUDE_DIRS}
)
target_include_directories(sycl-fusion
  PUBLIC
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${SYCL_JIT_BASE_DIR}/common/include>
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/lib
)

find_package(Threads REQUIRED)

target_link_libraries(sycl-fusion
  PRIVATE
  sycl-fusion-common
  LLVMSPIRVLib
  SYCLKernelFusionPasses
  ${CMAKE_THREAD_LIBS_INIT}
)

if("NVPTX" IN_LIST LLVM_TARGETS_TO_BUILD)
  target_compile_definitions(sycl-fusion PRIVATE FUSION_JIT_SUPPORT_PTX)
endif()

if("AMDGPU" IN_LIST LLVM_TARGETS_TO_BUILD)
  target_compile_definitions(sycl-fusion PRIVATE FUSION_JIT_SUPPORT_AMDGCN)
endif()

if (BUILD_SHARED_LIBS)
  if(NOT MSVC AND NOT APPLE)
    # Manage symbol visibility through the linker to make sure no LLVM symbols
    # are exported and confuse the drivers.
    set(linker_script "${CMAKE_CURRENT_SOURCE_DIR}/ld-version-script.txt")
    target_link_libraries(
      sycl-fusion PRIVATE "-Wl,--version-script=${linker_script}")
    set_target_properties(sycl-fusion PROPERTIES LINK_DEPENDS ${linker_script})
  endif()
endif()
