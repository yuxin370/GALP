# galp/external/CMakeLists.txt

# external/include
if (NOT TARGET galp_external_headers)
  add_library(galp_external_headers INTERFACE)
  target_include_directories(galp_external_headers INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  )
endif()

# ----------------------------
# ALP library
# ----------------------------
if (NOT TARGET galp_alp)
  add_library(galp_alp STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/alp/alp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/alp/encoder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/alp/falp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/alp/fastlanes_ffor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/alp/fastlanes_gen_ffor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/alp/fastlanes_gen_unffor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/alp/fastlanes_unffor.cpp
  )

  target_compile_features(galp_alp PUBLIC cxx_std_20)
  target_link_libraries(galp_alp PUBLIC galp_external_headers)

  option(ALP_ENABLE_ADAPTIVE_MODE "Enable adaptive mode" ON)
  if (ALP_ENABLE_ADAPTIVE_MODE)
    target_compile_definitions(galp_alp PUBLIC ALP_IS_ADAPTIVE_MODE_ENABLED=true)
  else()
    target_compile_definitions(galp_alp PUBLIC ALP_IS_ADAPTIVE_MODE_ENABLED=false)
  endif()

  if (COMMAND apply_warnings)
    apply_warnings(galp_alp)
  endif()
endif()

if (NOT TARGET ALP::alp)
  add_library(ALP::alp ALIAS galp_alp)
endif()

# ----------------------------
# FLS generated pack/unpack library 
# ----------------------------
if (NOT TARGET galp_fls_gen)
  add_library(galp_fls_gen STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/fls_gen/fallback_scalar_aav_1024_uf1_pack_src.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/fls_gen/fallback_scalar_aav_1024_uf1_unpack_src.cpp
  )

  target_compile_features(galp_fls_gen PUBLIC cxx_std_20)
  target_link_libraries(galp_fls_gen PUBLIC galp_external_headers)

  if (COMMAND apply_warnings)
    apply_warnings(galp_fls_gen)
  endif()
endif()

if (NOT TARGET FLS::gen)
  add_library(FLS::gen ALIAS galp_fls_gen)
endif()

