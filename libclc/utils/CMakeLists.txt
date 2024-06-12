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

if( LIBCLC_STANDALONE_BUILD )
  add_llvm_executable( prepare_builtins prepare-builtins.cpp )
else()
  add_llvm_utility( prepare_builtins prepare-builtins.cpp )
endif()

target_compile_definitions( prepare_builtins PRIVATE ${LLVM_VERSION_DEFINE} )
# These were not properly reported in early LLVM and we don't need them
target_compile_options( prepare_builtins PRIVATE -fno-rtti -fno-exceptions )

# Setup remangler tool
add_llvm_subdirectory(LIBCLC TOOL libclc-remangler)
