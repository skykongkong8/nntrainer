simd_interface_arm_headers = [
    'arm_compute_backend.h',
    'neon_single.h',
    'neon_setting.h',
    'neon_mathfun.h',
    'neon_mathfun.hxx'
]
simd_interface_arm_sources = [
    'arm_compute_backend.cpp',
    'neon_single.cpp'
]

if get_option('enable-fp16')
    subdir('hgemm')
    nntrainer_inc += include_directories('hgemm')
    nntrainer_inc_abs += meson.current_source_dir() / 'hgemm'
    simd_interface_arm_headers += 'neon_half.h'
    simd_interface_arm_sources += 'neon_half.cpp'

    subdir('matrix_transpose_neon')
    nntrainer_inc += include_directories('matrix_transpose_neon')
    nntrainer_inc_abs += meson.current_source_dir() / 'matrix_transpose_neon'
endif

foreach s : simd_interface_arm_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : simd_interface_arm_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach
