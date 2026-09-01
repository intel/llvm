# is_llvm_runtime_enabled(runtime result) sets 'result' to TRUE if 'runtime'
# (e.g. libclc) is enabled for any configured runtime target, FALSE otherwise.
#
# Use this instead of if(TARGET <runtime>) because this runs before
# runtimes/CMakeLists.txt creates runtime targets, so a TARGET check would
# always be false. The enabled-runtimes lists aren't affected by that
# ordering.
function(is_llvm_runtime_enabled runtime result)
  set(enabled FALSE)
  foreach(runtime_target IN LISTS LLVM_RUNTIME_TARGETS)
    if("${runtime}" IN_LIST RUNTIMES_${runtime_target}_LLVM_ENABLE_RUNTIMES)
      set(enabled TRUE)
      break()
    endif()
  endforeach()
  set(${result} ${enabled} PARENT_SCOPE)
endfunction()
