
add_llvm_library(sycl-fusion
  SHARED
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

install(TARGETS sycl-fusion
  LIBRARY DESTINATION "lib${LLVM_LIBDIR_SUFFIX}" COMPONENT sycl-fusion
  RUNTIME DESTINATION "bin" COMPONENT sycl-fusion)
