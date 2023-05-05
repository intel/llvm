# Generate Documentation
add_custom_target(polygeist-doc)
function(add_polygeist_doc doc_filename output_file output_directory command)
  add_mlir_doc(${doc_filename} ${output_file} ${output_directory} ${command})
  add_dependencies(polygeist-doc ${output_file}DocGen)
endfunction()
