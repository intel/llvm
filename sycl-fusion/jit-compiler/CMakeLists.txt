
add_llvm_library(sycl-fusion
   lib/KernelFusion.cpp
   lib/JITContext.cpp
   lib/translation/SPIRVLLVMTranslation.cpp
   lib/fusion/FusionPipeline.cpp
   lib/fusion/FusionHelper.cpp
   lib/fusion/ModuleHelper.cpp
   lib/helper/ConfigHelper.cpp

  LINK_COMPONENTS
   Core
   Support
   Analysis
   TransformUtils
   Passes
   Linker
   ScalarOpts
   InstCombine
)

target_include_directories(sycl-fusion
  PUBLIC
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${SYCL_JIT_BASE_DIR}/common/include>
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/lib
  ${LLVM_SPIRV_INCLUDE_DIRS}
)

find_package(Threads REQUIRED)

target_link_libraries(sycl-fusion
  PRIVATE
  LLVMSPIRVLib
  SYCLKernelFusionPasses
  ${CMAKE_THREAD_LIBS_INIT}
)

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
