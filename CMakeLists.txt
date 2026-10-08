cmake_minimum_required(VERSION 3.20)
project(vg_gguf_engine VERSION 0.1.0 LANGUAGES C CXX)

option(VG_ENABLE_ASAN "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(VG_ENABLE_HIP "Configure ROCm/HIP detection and build the trivial HIP probe target. OFF by default. No GPU matmuls are implied or present." OFF)

if(VG_ENABLE_HIP)
  find_program(VG_HIPCC hipcc
    PATHS "$ENV{HIP_PATH}/bin" "G:/ROCM10RT-gfx1201/bin" "G:/ROCM10RT-gfx1031/bin"
    NO_DEFAULT_PATH)
  if(NOT VG_HIPCC)
    find_program(VG_HIPCC hipcc)
  endif()
  if(NOT VG_HIPCC)
    message(FATAL_ERROR "VG_ENABLE_HIP=ON but hipcc was not found. Set HIP_PATH or install ROCm/HIP.")
  endif()
  message(STATUS "VG_ENABLE_HIP=ON: hipcc = ${VG_HIPCC}")
  if(NOT VG_ROCM_ROOT)
    get_filename_component(VG_ROCM_ROOT "${VG_HIPCC}/../.." ABSOLUTE)
  endif()
  message(STATUS "VG_ENABLE_HIP=ON: VG_ROCM_ROOT = ${VG_ROCM_ROOT}")
  set(VG_HIP_PROBE_EXE "${CMAKE_BINARY_DIR}/vg-hip-probe.exe")
  add_custom_command(OUTPUT "${VG_HIP_PROBE_EXE}"
    COMMAND "${VG_HIPCC}" --offload-arch=gfx1201 -O2 "${CMAKE_SOURCE_DIR}/tools/hip_probe.hip" -o "${VG_HIP_PROBE_EXE}" -I"${HIP_PATH}/include" --rocm-device-lib-path="${VG_ROCM_ROOT}/lib/llvm/amdgcn/bitcode" -lamdhip64
    DEPENDS "${CMAKE_SOURCE_DIR}/tools/hip_probe.hip"
    COMMENT "Building trivial HIP probe with hipcc (--offload-arch=gfx1201)")
  add_custom_target(vg-hip-probe ALL DEPENDS "${VG_HIP_PROBE_EXE}")
  set(VG_HIP_MATMUL_OBJ "${CMAKE_BINARY_DIR}/hip_matmul.hip.obj")
  set(VG_HIP_LIB "${CMAKE_BINARY_DIR}/libvghip.a")
  add_custom_command(OUTPUT "${VG_HIP_LIB}"
    COMMAND "${VG_HIPCC}" --offload-arch=gfx1201 -O2 -c -I"${CMAKE_SOURCE_DIR}/include" -I"${HIP_PATH}/include" --rocm-device-lib-path="${VG_ROCM_ROOT}/lib/llvm/amdgcn/bitcode" "${CMAKE_SOURCE_DIR}/tools/hip_matmul.hip" -o "${VG_HIP_MATMUL_OBJ}"
    COMMAND "${CMAKE_AR}" crs "${VG_HIP_LIB}" "${VG_HIP_MATMUL_OBJ}"
    DEPENDS "${CMAKE_SOURCE_DIR}/tools/hip_matmul.hip"
    COMMENT "Building F32/F16 HIP matmul kernels with hipcc (--offload-arch=gfx1201)")
add_custom_target(vghip ALL DEPENDS "${VG_HIP_LIB}")
add_library(vghip_lib STATIC IMPORTED GLOBAL)
set_target_properties(vghip_lib PROPERTIES IMPORTED_LOCATION "${VG_HIP_LIB}")
endif()

option(VG_BUILD_EXAMPLE_PLUGIN "Build the example runtime plugin" ON)
option(VG_BUILD_FULL_ENGINE "Build the complete model-generation engine" ON)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(vg_core STATIC src/gguf.c src/vg_prof.c src/tensor_source.c src/kv_cache.c src/plugin_loader.c src/trace.c src/quant.c src/tokenizer.c src/threadpool.c src/cpu_ops.c src/arch_spec.c src/model_graph.c src/layer_math.c)
if(VG_ENABLE_HIP)
  target_compile_definitions(vg_core PRIVATE VG_HAVE_HIP=1)
  target_link_libraries(vg_core PUBLIC vghip_lib "${VG_ROCM_ROOT}/bin/amdhip64_7.dll")
  add_dependencies(vg_core vghip)
  add_executable(vg-hip-matmul-test tools/hip_matmul_test.c)
  target_include_directories(vg-hip-matmul-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(vg-hip-matmul-test PRIVATE vg_core)
  add_dependencies(vg-hip-matmul-test vghip)
  add_test(NAME vg-hip-matmul COMMAND vg-hip-matmul-test "${CMAKE_SOURCE_DIR}/testdata/llama2.c-stories15M.f16.gguf")
endif()
target_include_directories(vg_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(vg_core PRIVATE _FILE_OFFSET_BITS=64)
if(NOT WIN32)
  find_package(Threads REQUIRED)
  target_link_libraries(vg_core PUBLIC Threads::Threads ${CMAKE_DL_LIBS})
endif()
if(MSVC)
  target_compile_options(vg_core PRIVATE /W4)
else()
  target_compile_options(vg_core PRIVATE -Wall -Wextra -Wconversion -Wshadow -Wpedantic)
endif()

if(VG_BUILD_FULL_ENGINE)
  add_library(vg_full STATIC src/native_engine.c src/lora.c src/speculative.c)
  target_include_directories(vg_full PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_executable(vg-run src/run_main.cpp)
  target_include_directories(vg-run PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(vg-run PRIVATE vg_full vg_core)
  message(STATUS "Full generation engine enabled (native inference)")
endif()

add_executable(vg-engine src/main.c)
target_include_directories(vg-engine PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(vg-engine PRIVATE vg_core)
if(VG_BUILD_FULL_ENGINE)
  target_compile_definitions(vg-engine PRIVATE VG_BUILD_FULL_ENGINE=1)
  target_link_libraries(vg-engine PRIVATE vg_full)
endif()
if(WIN32)
  target_link_libraries(vg-engine PRIVATE bcrypt)
endif()

if(VG_BUILD_EXAMPLE_PLUGIN)
  add_library(vg-example-plugin SHARED tools/example_plugin.c)
  target_include_directories(vg-example-plugin PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  set_target_properties(vg-example-plugin PROPERTIES PREFIX "")
endif()

enable_testing()
add_executable(vg-bench tools/bench.c)
 target_include_directories(vg-bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
 target_link_libraries(vg-bench PRIVATE vg_core)
add_executable(vg-gemv-bench tools/gemv_bench.c)
 target_include_directories(vg-gemv-bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
 target_link_libraries(vg-gemv-bench PRIVATE vg_core)
add_executable(vg-gemm-bench tools/gemm_bench.c)
  target_include_directories(vg-gemm-bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(vg-gemm-bench PRIVATE vg_core)

# IQ4_XS fused-GEMV correctness + throughput. Pass a model path to test real
# IQ4_XS bytes; falls back to synthetic blocks. The exactness assertions are the
# gate: the fused kernel must stay bit-identical to full-matrix F32 dequant.
# Fused-K-quant GEMV correctness + throughput (IQ4_XS type 23, Q5_K type 13).
# Pass a model path to test real bytes for both types - gemma4-12B-IQ4_XS.gguf
# carries both. Falls back to synthetic blocks. The exactness assertions are the
# gate: each fused kernel must stay bit-identical to full-matrix F32 dequant,
# and the batched GEMM must stay bit-identical to n separate GEMV calls.
add_executable(vg-fused-kernel-test tools/fused_quant_kernel_test.c)
  target_include_directories(vg-fused-kernel-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(vg-fused-kernel-test PRIVATE vg_core)
add_test(NAME vg-fused-kernel COMMAND vg-fused-kernel-test)
if(EXISTS "${CMAKE_SOURCE_DIR}/testdata/stories15M-IQ4_XS.gguf")
  # Q5_K and IQ4_XS only; stories15M is Q4_0, so the real-bytes path applies only
  # if a model of these types is present.
  add_test(NAME vg-fused-kernel-model COMMAND vg-fused-kernel-test
    "${CMAKE_SOURCE_DIR}/testdata/stories15M-IQ4_XS.gguf")
endif()

# Peak-working-set probe: proves a matmul on packed weights does not
# materialize the matrix as F32. Needs a model path, so it is a tool rather
# than a ctest; see AGENTS.md for the measured output.
add_executable(vg-quant-mem-probe tools/quant_mem_probe.c)
  target_include_directories(vg-quant-mem-probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(vg-quant-mem-probe PRIVATE vg_core)

add_executable(vg-tests tests/test_core.c)
target_include_directories(vg-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(vg-tests PRIVATE vg_core)
add_test(NAME vg-core COMMAND vg-tests)
add_executable(vg-quant-tests tests/test_quant.c)
target_include_directories(vg-quant-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(vg-quant-tests PRIVATE vg_core)
add_test(NAME vg-quant COMMAND vg-quant-tests)
add_executable(vg-tensor-tests tests/test_tensor.c)
target_include_directories(vg-tensor-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(vg-tensor-tests PRIVATE vg_core)
add_test(NAME vg-tensor COMMAND vg-tensor-tests)
if(EXISTS "${CMAKE_SOURCE_DIR}/testdata/stories15M-q4_0.gguf")
  add_test(NAME vg-tensor-model COMMAND vg-tensor-tests "${CMAKE_SOURCE_DIR}/testdata/stories15M-q4_0.gguf")
  message(STATUS "Tensor-source model tests registered (testdata/stories15M-q4_0.gguf found)")
endif()

add_executable(vg-tokenizer-tests tests/test_tokenizer.c)
target_include_directories(vg-tokenizer-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(vg-tokenizer-tests PRIVATE vg_core)
add_test(NAME vg-tokenizer COMMAND vg-tokenizer-tests)

add_executable(vg-sampler-tests tests/test_sampler.c)
target_include_directories(vg-sampler-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(vg-sampler-tests PRIVATE vg_full vg_core)
add_test(NAME vg-sampler COMMAND vg-sampler-tests)

if(TARGET vg-run AND EXISTS "${CMAKE_SOURCE_DIR}/testdata/stories15M-q4_0.gguf")
  add_executable(vg-prompt-tests tests/test_prompt.c)
  target_include_directories(vg-prompt-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_compile_definitions(vg-prompt-tests PRIVATE VG_TEST_MODEL="${CMAKE_SOURCE_DIR}/testdata/stories15M-q4_0.gguf")
  target_link_libraries(vg-prompt-tests PRIVATE vg_full vg_core)
  add_test(NAME vg-prompt COMMAND vg-prompt-tests)
  message(STATUS "Prompt-vs-oracle test registered (testdata/stories15M-q4_0.gguf found)")
elseif(TARGET vg-run AND NOT EXISTS "${CMAKE_SOURCE_DIR}/testdata/stories15M-q4_0.gguf")
  message(WARNING "testdata/stories15M-q4_0.gguf is missing; prompt-vs-oracle test (vg-prompt) was NOT registered.")
endif()

if(TARGET vg-run AND EXISTS "${CMAKE_SOURCE_DIR}/testdata/stories15M-q4_0.gguf")
  add_test(NAME vg-full-smoke COMMAND vg-run -m "${CMAKE_SOURCE_DIR}/testdata/stories15M-q4_0.gguf" -p "Once upon a time" -n 8 -c 128 --temp 0.0)
  message(STATUS "Smoke test registered (testdata/stories15M-q4_0.gguf found)")
elseif(TARGET vg-run AND NOT EXISTS "${CMAKE_SOURCE_DIR}/testdata/stories15M-q4_0.gguf")
  message(WARNING "vg-run target exists but testdata/stories15M-q4_0.gguf is missing; smoke test (vg-full-smoke) was NOT registered. CI will have no full-engine coverage.")
endif()

# Attention-convention gate: the engine's recorded top-8 must be reproducible
# by an INDEPENDENT forward pass (tools/oracle_gate.py -> tools/conv_matrix.py),
# and the matching convention must be the one the architecture specifies.
# This is the gate for the bug class that bit qwen2 (adjacent RoPE pairing +
# bias applied after the rotation): bit-stable, plausible-looking, and wrong.
# Exit 77 (missing python/gguf/model) is reported as SKIPPED, never as passed.
#
# The registered model is llama/adjacent+no-bias, which testdata covers for
# free. Point VG_ORACLE_MODEL at a qwen2/gemma file to widen coverage:
#   cmake -DVG_ORACLE_MODEL=/path/to/qwen2.gguf -DVG_ORACLE_TOKENS=1,2,3
#     -DVG_ORACLE_TOP8="123=1.0 ..." ..
if(TARGET vg-run)
  set(VG_ORACLE_MODEL "${CMAKE_SOURCE_DIR}/testdata/llama2.c-stories15M.f16.gguf"
      CACHE FILEPATH "Model used by the vg-conventions oracle gate")
  set(VG_ORACLE_TOKENS "1,9038,2501,263,931"
      CACHE STRING "Prefill token ids for the vg-conventions gate")
  set(VG_ORACLE_TOP8 "29892=18.3606 727=14.9800 297=11.3658 263=10.0178 1023=7.9929 2501=6.8542 373=6.5669 10600=6.2903"
      CACHE STRING "Engine top-8 the oracle must reproduce (vg-conventions)")
  find_program(VG_PYTHON_EXE NAMES python3 python)
  if(VG_PYTHON_EXE AND EXISTS "${VG_ORACLE_MODEL}")
    add_test(NAME vg-conventions
             COMMAND "${VG_PYTHON_EXE}" "${CMAKE_SOURCE_DIR}/tools/oracle_gate.py"
                     "${VG_ORACLE_MODEL}" "${VG_ORACLE_TOKENS}" "${VG_ORACLE_TOP8}")
    set_tests_properties(vg-conventions PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)
    message(STATUS "Convention oracle gate registered (${VG_ORACLE_MODEL})")
  elseif(NOT VG_PYTHON_EXE)
    message(WARNING "python not found; vg-conventions oracle gate was NOT registered.")
  else()
    message(WARNING "VG_ORACLE_MODEL not found; vg-conventions oracle gate was NOT registered.")
  endif()
endif()

if(VG_ENABLE_ASAN AND NOT MSVC)
  target_compile_options(vg_core PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(vg-engine PRIVATE -fsanitize=address,undefined)
  target_link_options(vg-bench PRIVATE -fsanitize=address,undefined)
  target_link_options(vg-tests PRIVATE -fsanitize=address,undefined)
  target_link_options(vg-quant-tests PRIVATE -fsanitize=address,undefined)
  if(TARGET vg-run)
    target_compile_options(vg-run PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(vg-run PRIVATE -fsanitize=address,undefined)
  endif()
endif()

install(TARGETS vg-engine RUNTIME DESTINATION bin)
if(TARGET vg-run)
  install(TARGETS vg-run vg_full RUNTIME DESTINATION bin LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
endif()
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY docs/ DESTINATION share/doc/vg-gguf-engine)
