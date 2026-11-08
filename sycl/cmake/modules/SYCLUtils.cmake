list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(CheckLinkerFlag)

# add_stripped_pdb(TARGET_NAME [COMPONENT component])
#
# Will add option for generating stripped PDB file and, if LLVM_ENABLE_PDB is
# set, install the generated file as ${ARG_TARGET_NAME}.pdb in bin folder, as
# part of the given install component (defaults to ${ARG_TARGET_NAME}).
# NOTE: LLD does not currently support /PDBSTRIPPED so the PDB file is optional.
macro(add_stripped_pdb ARG_TARGET_NAME)
  cmake_parse_arguments(ARG_PDB "" "COMPONENT" "" ${ARGN})
  if(NOT ARG_PDB_COMPONENT)
    set(ARG_PDB_COMPONENT ${ARG_TARGET_NAME})
  endif()
  check_linker_flag(CXX "LINKER:/PDBSTRIPPED:${ARG_TARGET_NAME}.stripped.pdb"
                         LINKER_SUPPORTS_PDBSTRIPPED)
  if(LINKER_SUPPORTS_PDBSTRIPPED)
    # A relative path would be resolved against the linker's working directory,
    # which is not CMAKE_CURRENT_BINARY_DIR for every generator.
    set(_stripped_pdb
        "${CMAKE_CURRENT_BINARY_DIR}/${ARG_TARGET_NAME}.stripped.pdb")
    target_link_options(${ARG_TARGET_NAME}
                        PRIVATE "LINKER:/PDBSTRIPPED:${_stripped_pdb}")
    if(LLVM_ENABLE_PDB)
      install(FILES "${_stripped_pdb}"
              DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
              RENAME "${ARG_TARGET_NAME}.pdb"
              COMPONENT ${ARG_PDB_COMPONENT}
              OPTIONAL)
    endif()
  endif()
endmacro()
