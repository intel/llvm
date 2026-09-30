# get_targets_building_llvm_runtime(runtime result) sets 'result' to the list of runtime targets
# (e.g. amdgcn-amd-amdhsa, nvptx64-nvidia-cuda) for which the specified runtime
# (e.g. libclc) will be built.
#
# Use this instead of if(TARGET <runtime>-<tgt>) because this runs before
# runtimes/CMakeLists.txt creates runtime targets, so a TARGET check would
# always be false. The enabled-runtimes lists aren't affected by that
# ordering.
function(get_targets_building_llvm_runtime runtime result)
  set(targets)
  foreach(runtime_target IN LISTS LLVM_RUNTIME_TARGETS)
    if("${runtime}" IN_LIST RUNTIMES_${runtime_target}_LLVM_ENABLE_RUNTIMES)
      list(APPEND targets "${runtime_target}")
    endif()
  endforeach()
  set(${result} "${targets}" PARENT_SCOPE)
endfunction()
