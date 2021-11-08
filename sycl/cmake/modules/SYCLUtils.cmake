# add_stripped_pdb(TARGET_NAME)
#
# Will add option for generating stripped PDB file and install the generated
# file as ${ARG_TARGET_NAME}.pdb in bin folder.
# NOTE: LLD does not currently support /PDBSTRIPPED so the PDB file is optional.
macro(add_stripped_pdb ARG_TARGET_NAME)
  target_link_options(${ARG_TARGET_NAME} PRIVATE "/PDBSTRIPPED:${ARG_TARGET_NAME}.stripped.pdb")
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${ARG_TARGET_NAME}.stripped.pdb"
          DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
          RENAME "${ARG_TARGET_NAME}.pdb"
          COMPONENT ${ARG_TARGET_NAME}
          OPTIONAL)
endmacro()
