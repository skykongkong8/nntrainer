fbgemm_headers = [

]

fbgemm_sources = [
    'Fbgemm.cc',
    'fbgemm_interface.cpp',
    'ExecuteKernel.cc',
    'FbgemmFP16.cc',
    'FbgemmFP16UKernelsAvx2.cc',
    'FbgemmFP16UKernelsAvx512.cc',
    'FbgemmFP16UKernelsAvx512_256.cc',
    'Utils.cc',
    'TransposeUtils.cc',
    'UtilsAvx2.cc',
]

foreach s : fbgemm_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : fbgemm_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach

