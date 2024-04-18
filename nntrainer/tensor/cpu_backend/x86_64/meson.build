simd_interface_x86_64_headers = [
  'x86_64_compute_backend.h',
]
simd_interface_x86_64_sources = [
  'x86_64_compute_backend.cpp',
]

if get_option('enable-fp16')
    simd_interface_x86_64_headers += 'blas_avx.h'
    simd_interface_x86_64_sources += 'blas_avx.cpp'
endif

foreach s : simd_interface_x86_64_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : simd_interface_x86_64_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach
