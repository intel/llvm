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

# Trigger the generation of the SPIR-V binding header

if (TARGET ClangSPIRVBuiltinsJSON)
  # We can't access directly the output of the target
  # but can at least querry where it is stored.
  get_property(JSON_SPIRV_BUILTINS_DIR TARGET ClangSPIRVBuiltinsJSON PROPERTY BINARY_DIR)
  get_property(IS_JSON_SPIRV_BUILTINS_DIR_SET TARGET ClangSPIRVBuiltinsJSON PROPERTY BINARY_DIR SET)

  if (IS_JSON_SPIRV_BUILTINS_DIR_SET)
    set( LIBCLC_SPIRV_HEADER_TMP ${CMAKE_CURRENT_BINARY_DIR}/spirv_builtins.h )
    set( LIBCLC_SPIRV_HEADER ${LIBCLC_ROOT_DIR}/generic/include/spirv/spirv_builtins.h )

    # SPIRVBuiltins.json needs to be in sync with the name in clang/lib/Sema/CMakeLists.txt
    file( TO_CMAKE_PATH ${JSON_SPIRV_BUILTINS_DIR}/SPIRVBuiltins.json JSON_SPIRV_BUILTINS )
    file( TO_CMAKE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/generate-spirv-header.py SPIRV_HEADER_SCRIPT_LOC )

    set( USE_CLANG_FORMAT )
    if (LLVM_CLANG_FORMAT)
      set( USE_CLANG_FORMAT "-format" "${LLVM_CLANG_FORMAT}")
    endif ()

    add_custom_command(
	    OUTPUT ${LIBCLC_SPIRV_HEADER_TMP}
	    COMMAND ${PYTHON} ${SPIRV_HEADER_SCRIPT_LOC}
        ${USE_CLANG_FORMAT}
        ${JSON_SPIRV_BUILTINS}
        -o ${LIBCLC_SPIRV_HEADER_TMP}
	    DEPENDS ${SPIRV_HEADER_SCRIPT_LOC} ${JSON_SPIRV_BUILTINS} ${LLVM_CLANG_FORMAT})

    add_custom_command(OUTPUT ${LIBCLC_SPIRV_HEADER}
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${LIBCLC_SPIRV_HEADER_TMP} ${LIBCLC_SPIRV_HEADER}
      COMMENT "Copying SPIR-V header"
      DEPENDS ${LIBCLC_SPIRV_HEADER_TMP}
      )

    add_custom_target( gen_spirv_headers DEPENDS ${LIBCLC_SPIRV_HEADER})
  else ()
    add_custom_target( gen_spirv_headers )
  endif ()
else ()
  add_custom_target( gen_spirv_headers )
endif ()
