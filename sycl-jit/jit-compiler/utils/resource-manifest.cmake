# Write the hash and path of every file under DIR to MANIFEST if they changed.
file(GLOB_RECURSE files "${DIR}/*")
foreach(file IN LISTS files)
  file(MD5 "${file}" hash)
  string(APPEND manifest "${hash} ${file}\n")
endforeach()
file(CONFIGURE OUTPUT "${MANIFEST}" CONTENT "${manifest}" @ONLY)
