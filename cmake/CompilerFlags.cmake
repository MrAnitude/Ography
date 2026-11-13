# Shared warning/optimisation flags, applied per target via ography_set_flags().
# -Wno-unused-parameter is temporary while most functions are still stubs.

function(ography_set_flags target)
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wno-unused-parameter>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-O3 -march=native>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -Xcompiler=-Wall,-Wno-unused-parameter>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3 --use_fast_math>
    # Keep line info in every build so Nsight Compute can map SASS back to source.
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
  )
endfunction()
