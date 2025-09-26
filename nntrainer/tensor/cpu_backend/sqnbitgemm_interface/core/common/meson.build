sqnbitgemm_core_common_headers = [
  'basic_types.h',
  'code_location.h',
  'common.h',
  'const_pointer_container.h',
  'cpuid_arch_definition.h',
  'cpuid_info.h',
  'cpuid_uarch.h',
  'denormal.h',
  'exceptions.h',
  'hash_combine.h',
  'make_string.h',
  'optional.h',
  'parse_string.h',
  'path_string.h',
  'profiler_common.h',
  'span_utils.h',
  'spin_pause.h',
  'status.h',
  'string_helper.h',
]

sqnbitgemm_core_common_sources = [
  'cpuid_info.cc',
]

subdir('logging')
nntrainer_inc += include_directories('logging')
nntrainer_inc_abs += meson.current_source_dir() / 'logging'

foreach h : sqnbitgemm_core_common_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach

foreach s : sqnbitgemm_core_common_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach
