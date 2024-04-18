simd_interface_aarch64_headers = [
    'aarch64_compute_backend.h',
    'neon_single.h',
    'neon_setting.h'
]
simd_interface_aarch64_sources = [
    'aarch64_compute_backend.cpp',
    'neon_single.cpp'
]

subdir('transpose')
nntrainer_inc += include_directories('transpose')
nntrainer_inc_abs += meson.current_source_dir() / 'transpose'

if get_option('enable-fp16')
    subdir('hgemm')
    nntrainer_inc += include_directories('hgemm')
    nntrainer_inc_abs += meson.current_source_dir() / 'hgemm'
    simd_interface_aarch64_headers += 'neon_half.h'
    simd_interface_aarch64_sources += 'neon_half.cpp'
endif

foreach s : simd_interface_aarch64_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : simd_interface_aarch64_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach
