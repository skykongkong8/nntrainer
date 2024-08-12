hgemm_headers = [
  'my_igemm.h',
  'aligned-allocator.h',
  'compute_backend.h',
  'data_util.h',
  'igemm_init.h',
  'fxdiv.h',
  'igemm_kernel_sse.h',
  'igemm_packing.h',
  'scopy_avx.h',
]

hgemm_sources = [
  'my_igemm.cpp',
  'compute_backend.cpp',
  'igemm_init.cpp',
  'igemm_kernel_sse.cpp',
  'igemm_packing.cpp',
  'scopy_avx.cpp',
]

foreach s : hgemm_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : hgemm_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach

