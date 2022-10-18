list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(LLVMCheckLinkerFlag)

# add_stripped_pdb(TARGET_NAME)
#
# Will add option for generating stripped PDB file and install the generated
# file as ${ARG_TARGET_NAME}.pdb in bin folder.
# NOTE: LLD does not currently support /PDBSTRIPPED so the PDB file is optional.
macro(add_stripped_pdb ARG_TARGET_NAME)
  llvm_check_linker_flag(CXX "/PDBSTRIPPED:${ARG_TARGET_NAME}.stripped.pdb"
                         LINKER_SUPPORTS_PDBSTRIPPED)
  if(LINKER_SUPPORTS_PDBSTRIPPED)
    target_link_options(${ARG_TARGET_NAME}
                        PRIVATE "/PDBSTRIPPED:${ARG_TARGET_NAME}.stripped.pdb")
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${ARG_TARGET_NAME}.stripped.pdb"
            DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
            RENAME "${ARG_TARGET_NAME}.pdb"
            COMPONENT ${ARG_TARGET_NAME}
            OPTIONAL)
  endif()
endmacro()
