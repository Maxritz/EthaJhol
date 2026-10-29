cmake_minimum_required(VERSION 3.20)
project(vg_gguf_engine VERSION 0.1.0 LANGUAGES C CXX)

option(VG_ENABLE_VULKAN "Build the Vulkan backend" ON)
option(VG_ENABLE_ASAN "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(VG_BUILD_EXAMPLE_PLUGIN "Build the example runtime plugin" ON)
option(VG_BUILD_FULL_ENGINE "Build the complete model-generation engine" ON)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
# Sanitizer flags are applied directory-wide. Attaching them to individual targets only worked
# for targets that already existed at that point, so vg_full (created further down) was linked
# without the sanitizer runtime and failed to build.
set(VG_SAN_FLAGS "")
if(VG_ENABLE_ASAN AND NOT MSVC)
  set(VG_SAN_FLAGS -fsanitize=address,undefined -fno-omit-frame-pointer)
  add_compile_options(${VG_SAN_FLAGS})
  add_link_options(${VG_SAN_FLAGS})
endif()
# Warning flags are directory-wide for the same reason the sanitizer flags are: attaching them
# per target left vg_full (the whole generation engine: native_engine.c, lora.c, speculative.c)
# and vg-engine/vg-run completely unchecked, which is how a dead static function and a fixed
# 256-entry draft buffer survived review.
if(MSVC)
  add_compile_options(/W4)
else()
  # -Wconversion belongs in this list, not only on vg_core: the engine itself (main.c,
  # native_engine.c, lora.c, speculative.c) narrows sizes and indices too, and it was the
  # last place a sign-changing conversion could hide.
  add_compile_options(-Wall -Wextra -Wconversion -Wshadow -Wpedantic)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(vg_core STATIC src/gguf.c src/tensor_source.c src/kv_cache.c src/plugin_loader.c src/trace.c src/quant.c src/tokenizer.c src/threadpool.c src/cpu_ops.c src/model_graph.c src/model_probe.c src/lora.c)
target_include_directories(vg_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(vg_core PRIVATE _FILE_OFFSET_BITS=64)
if(NOT WIN32)
  find_package(Threads REQUIRED)
  target_link_libraries(vg_core PUBLIC Threads::Threads ${CMAKE_DL_LIBS} m)
  find_library(VG_MATH_LIBRARY m)
  if(VG_MATH_LIBRARY)
    target_link_libraries(vg_core PUBLIC ${VG_MATH_LIBRARY})
  endif()
endif()
if(MSVC)
  target_compile_options(vg_core PRIVATE /W4)
else()
  target_compile_options(vg_core PRIVATE -Wall -Wextra -Wshadow -Wpedantic)
endif()

if(VG_ENABLE_VULKAN)
  find_package(Vulkan QUIET)
  if(Vulkan_FOUND)
    message(STATUS "Vulkan backend enabled (found ${Vulkan_VERSION})")
    find_program(GLSLANG_VALIDATOR glslangValidator REQUIRED)
    set(SPIRV_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
    file(MAKE_DIRECTORY ${SPIRV_DIR})
    set(VG_SHADER_SOURCES
      ${CMAKE_CURRENT_SOURCE_DIR}/shaders/matvec_i8.comp
      ${CMAKE_CURRENT_SOURCE_DIR}/shaders/matvec_f16.comp
      ${CMAKE_CURRENT_SOURCE_DIR}/shaders/matvec_f32.comp
      ${CMAKE_CURRENT_SOURCE_DIR}/shaders/matvec_q4_0.comp
${CMAKE_CURRENT_SOURCE_DIR}/shaders/matvec_q8_0.comp
${CMAKE_CURRENT_SOURCE_DIR}/shaders/matvec_q4_k.comp
${CMAKE_CURRENT_SOURCE_DIR}/shaders/matvec_q6_k.comp
${CMAKE_CURRENT_SOURCE_DIR}/shaders/activation_quant_q8.comp
${CMAKE_CURRENT_SOURCE_DIR}/shaders/attn_scores.comp
${CMAKE_CURRENT_SOURCE_DIR}/shaders/resid_add.comp
${CMAKE_CURRENT_SOURCE_DIR}/shaders/rope_adj.comp
      ${CMAKE_CURRENT_SOURCE_DIR}/shaders/rmsnorm.comp
      ${CMAKE_CURRENT_SOURCE_DIR}/shaders/rope.comp
      ${CMAKE_CURRENT_SOURCE_DIR}/shaders/softmax.comp
      ${CMAKE_CURRENT_SOURCE_DIR}/shaders/swiglu.comp
    )
    set(VG_SHADER_OUTPUTS "")
    foreach(SHADER ${VG_SHADER_SOURCES})
      get_filename_component(SHADER_NAME ${SHADER} NAME)
      add_custom_command(
        OUTPUT ${SPIRV_DIR}/${SHADER_NAME}.spv
        COMMAND ${GLSLANG_VALIDATOR} -V --target-env vulkan1.1 ${SHADER} -o ${SPIRV_DIR}/${SHADER_NAME}.spv
        DEPENDS ${SHADER}
        VERBATIM)
      list(APPEND VG_SHADER_OUTPUTS ${SPIRV_DIR}/${SHADER_NAME}.spv)
    endforeach()
    add_custom_target(vg_shaders ALL DEPENDS ${VG_SHADER_OUTPUTS})
    add_library(vg_vulkan STATIC src/vulkan_backend.cpp)
    add_dependencies(vg_vulkan vg_shaders)
    target_include_directories(vg_vulkan PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(vg_vulkan PUBLIC Vulkan::Vulkan)
    if(MSVC)
      target_compile_options(vg_vulkan PRIVATE /W4)
    else()
      # Vulkan structs are aggregate-initialized with only sType set; the remaining fields are
      # zero-initialized by value-initialization, which is exactly what the API requires.
      target_compile_options(vg_vulkan PRIVATE -Wno-missing-field-initializers)
    endif()
    target_compile_definitions(vg_vulkan PUBLIC VG_HAS_VULKAN=1)
    target_compile_definitions(vg_core PRIVATE VG_HAS_VULKAN=1)
  else()
    message(WARNING "VG_ENABLE_VULKAN is ON but no Vulkan SDK was found; building with vulkan_stub.c (CPU-only). Install the Vulkan SDK for Vulkan inference.")
  endif()
endif()

if(VG_BUILD_FULL_ENGINE)
  # lora.c lives in vg_core: the graph applies adapter deltas inside matmul, so every target
  # that links the graph needs it (and the tests exercise it without the full engine).
  add_library(vg_full STATIC src/native_engine.c src/speculative.c)
  target_include_directories(vg_full PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  # vg_full calls the tokenizer and graph evaluator, so it must carry vg_core itself; linking
  # them as siblings put vg_core before the objects that need it in static link order.
  target_link_libraries(vg_full PUBLIC vg_core)
  if(TARGET vg_vulkan)
    target_link_libraries(vg_full PUBLIC vg_vulkan)
  endif()
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
if(TARGET vg_vulkan)
  target_link_libraries(vg-engine PRIVATE vg_vulkan)
  add_custom_command(TARGET vg-engine POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_BINARY_DIR}/shaders $<TARGET_FILE_DIR:vg-engine>/shaders)
else()
  add_library(vg_vulkan STATIC src/vulkan_stub.c)
  target_include_directories(vg_vulkan PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_compile_definitions(vg-engine PRIVATE VG_NO_VULKAN=1)
  target_link_libraries(vg-engine PRIVATE vg_vulkan)
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

# Matvec throughput: the metric that predicts decode speed for a memory-bound engine.
add_executable(vg-mmbench tools/mmbench.c)
target_include_directories(vg-mmbench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(vg-mmbench PRIVATE vg_core)

add_executable(vg-tests tests/test_core.c)
target_include_directories(vg-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(vg-tests PRIVATE vg_core)
add_test(NAME vg-core COMMAND vg-tests)
add_executable(vg-quant-tests tests/test_quant.c)
target_include_directories(vg-quant-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(vg-quant-tests PRIVATE vg_core)
add_test(NAME vg-quant COMMAND vg-quant-tests)
# The tokenizer fixture carries a SentencePiece vocabulary, a merge chain and <0xNN> byte
# tokens. The dense/moe fixtures are ASCII-only with no merges, so they cannot catch an
# encoder that ignores SentencePiece markers or never applies merges.
add_executable(vg-tokenizer-tests tests/test_tokenizer.c tests/fixture.c)
target_include_directories(vg-tokenizer-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(vg-tokenizer-tests PRIVATE vg_core)
add_test(NAME vg-tokenizer COMMAND vg-tokenizer-tests "${CMAKE_CURRENT_BINARY_DIR}/fixture_spm.gguf")

add_executable(vg-tensor-tests tests/test_tensor.c)
target_include_directories(vg-tensor-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(TARGET vg_vulkan)
  target_link_libraries(vg-tensor-tests PRIVATE vg_core vg_vulkan)
else()
  target_link_libraries(vg-tensor-tests PRIVATE vg_core)
endif()
add_test(NAME vg-tensor COMMAND vg-tensor-tests "${CMAKE_CURRENT_BINARY_DIR}/fixture_dense.gguf")

# Sampling (temperature / top-k / top-p / repetition penalty) is pure logit math in the
# engine, so it is unit-tested without loading a model.
add_executable(vg-sampler-tests tests/test_sampler.c)
target_include_directories(vg-sampler-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(vg-sampler-tests PRIVATE vg_full vg_core)
add_test(NAME vg-sampler COMMAND vg-sampler-tests)

add_executable(vg-engine-tests tests/test_engine.c tests/fixture.c)
target_include_directories(vg-engine-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/tests)
if(TARGET vg_vulkan)
  target_link_libraries(vg-engine-tests PRIVATE vg_core vg_vulkan)
else()
  target_link_libraries(vg-engine-tests PRIVATE vg_core)
endif()
if(TARGET vg_full)
  target_link_libraries(vg-engine-tests PRIVATE vg_full)
  target_compile_definitions(vg-engine-tests PRIVATE VG_TEST_FULL_ENGINE=1)
endif()
# This test writes the synthetic models every other model-backed test consumes, so it is the
# fixture setup; depending tests run after it and are skipped if it fails.
add_test(NAME vg-engine COMMAND vg-engine-tests "${CMAKE_CURRENT_BINARY_DIR}")
set_tests_properties(vg-engine PROPERTIES FIXTURES_SETUP fixture_model)
# The tensor-source suite used to point at testdata/stories15M-q4_0.gguf, which does not exist
# in the repository, so every one of its assertions was skipped. Run it on the fixture instead.
set_tests_properties(vg-tensor PROPERTIES FIXTURES_REQUIRED fixture_model)
if(TARGET vg-run)
  # End-to-end generation must produce output and the probe must accept the same model.
  add_test(NAME vg-full-smoke COMMAND vg-run -m "${CMAKE_CURRENT_BINARY_DIR}/fixture_dense.gguf" -p "abc" -n 8 -c 32 --temp 0.0)
  # No PASS_REGULAR_EXPRESSION: CTest would then pass on any matching stdout even when the
  # process fails. The exit status is the contract, and a failure message is fatal.
  set_tests_properties(vg-full-smoke PROPERTIES FIXTURES_REQUIRED fixture_model
    FAIL_REGULAR_EXPRESSION "generation failed")
  add_test(NAME vg-full-probe COMMAND vg-engine probe "${CMAKE_CURRENT_BINARY_DIR}/fixture_dense.gguf")
  set_tests_properties(vg-full-probe PROPERTIES FIXTURES_REQUIRED fixture_model PASS_REGULAR_EXPRESSION "result: SUPPORTED")
endif()
# A real-GGUF smoke test used to be announced here whenever testdata/ happened to exist. It
# registered nothing, so it only ever printed a status line: drop it rather than imply coverage
# that is not there.


install(TARGETS vg-engine RUNTIME DESTINATION bin)
if(TARGET vg-run)
  install(TARGETS vg-run vg_full RUNTIME DESTINATION bin LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
endif()
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY docs/ DESTINATION share/doc/vg-gguf-engine)
