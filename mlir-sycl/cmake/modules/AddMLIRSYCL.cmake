# Generate Documentation
add_custom_target(mlir-sycl-doc)
function(add_mlir_sycl_doc doc_filename output_file output_directory command)
  add_mlir_doc(${doc_filename} ${output_file} ${output_directory} ${command})
  add_dependencies(mlir-sycl-doc ${output_file}DocGen)
endfunction()
