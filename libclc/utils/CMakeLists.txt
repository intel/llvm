set( LLVM_VERSION_DEFINE "-DHAVE_LLVM=0x${LLVM_VERSION_MAJOR}0${LLVM_VERSION_MINOR}" )

# Setup prepare_builtins tools
set(LLVM_LINK_COMPONENTS
  BitWriter
  BitReader
  Core
  IRReader
  Support
  Passes
  Analysis
  )

add_llvm_executable( prepare_builtins
  prepare-builtins.cpp )

target_compile_definitions( prepare_builtins PRIVATE ${LLVM_VERSION_DEFINE} )

# Setup remangler tool
add_llvm_subdirectory(LIBCLC TOOL libclc-remangler)
