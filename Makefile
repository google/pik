# Copyright 2019 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

SIMD_FLAGS := -march=haswell

ifeq ($(origin CXX),default)
  CXX = clang++
endif
ifeq ($(origin CC),default)
  CC = clang
endif

INC_FLAGS = -I. -I../ -Ithird_party/brotli/c/include/ $(shell pkg-config --cflags eigen3)

# Clang-specific flags, tested with 6.0.1
# -Xclang -mprefer-vector-width=128 would decrease likelihood of undesired and
# unsafe autovectorization, but it requires Clang6.
M_FLAGS = -mrelax-all -Xclang -mrelocation-model -Xclang pic -Xclang -pic-level -Xclang 2 -pthread -mthread-model posix -Xclang -mdisable-fp-elim -Xclang -mconstructor-aliases -mpie-copy-relocations -Xclang -munwind-tables

LANG_FLAGS = -x c++ -std=c++11 -disable-free -disable-llvm-verifier -discard-value-names -Xclang -relaxed-aliasing -fmath-errno

CPU_FLAGS = -mavx2 -mfma -Xclang -target-cpu -Xclang haswell -Xclang -target-feature -Xclang +avx2

F_FLAGS = -fmerge-all-constants -fno-builtin-fwrite -fno-builtin-fread -fno-signed-char -fsized-deallocation -fnew-alignment=8 -fno-cxx-exceptions -fno-exceptions -fno-slp-vectorize -fno-vectorize

DBG_FLAGS = -dwarf-column-info -debug-info-kind=line-tables-only -dwarf-version=4 -debugger-tuning=gdb

PP_FLAGS = -DNDEBUG -Dregister= -D"__DATE__=\"redacted\"" -D"__TIMESTAMP__=\"redacted\"" -D"__TIME__=\"redacted\""

MSG_FLAGS = -Xclang -ferror-limit -Xclang 19 -Xclang -fmessage-length -Xclang 0 -fdiagnostics-show-option -fcolor-diagnostics

WARN_FLAGS  = -Wall -Werror -Wformat-security -Wno-char-subscripts -Wno-error=deprecated-declarations -Wno-sign-compare -Wno-strict-overflow -Wno-unused-function -Wthread-safety-analysis -Wno-unknown-warning-option -Wno-unused-command-line-argument -Wno-ignored-optimization-argument -Wno-ambiguous-member-template -Wno-pointer-sign -Wno-address-of-packed-member -Wno-enum-compare-switch -Wno-expansion-to-defined -Wno-extern-c-compat -Wno-gnu-alignof-expression -Wno-gnu-designator -Wno-gnu-variable-sized-type-not-at-end -Wno-ignored-attributes -Wno-ignored-qualifiers -Wno-inconsistent-missing-override -Wno-invalid-source-encoding -Wno-mismatched-tags -Wno-potentially-evaluated-expression -Wno-return-std-move -Wno-self-assign-overloaded -Wno-tautological-constant-compare -Wno-tautological-constant-in-range-compare -Wno-tautological-type-limit-compare -Wno-tautological-undefined-compare -Wno-tautological-unsigned-zero-compare -Wno-tautological-unsigned-enum-zero-compare -Wno-undefined-func-template -Wno-unknown-pragmas -Wno-unused-const-variable -Wno-unused-lambda-capture -Wno-unused-local-typedef -Wno-unused-private-field -Wno-private-header -Wfloat-overflow-conversion -Wfloat-zero-conversion -Wfor-loop-analysis -Wgnu-redeclared-enum -Wimplicit-fallthrough -Winfinite-recursion -Wliteral-conversion -Wself-assign -Wstring-conversion -Wtautological-overlap-compare -Wunused-comparison -Wvla -Wno-reserved-user-defined-literal -Wno-return-type-c-linkage -Wno-deprecated -Wno-invalid-offsetof -Wno-literal-suffix -Woverloaded-virtual -Wnon-virtual-dtor -Wdeprecated-increment-bool -Wc++11-compat -Wno-c++11-compat-binary-literal -Wc++2a-extensions -Wno-register -Wno-dynamic-exception-spec -Wprivate-header -Wno-builtin-macro-redefined

ALL_FLAGS = $(DBG_FLAGS) $(PP_FLAGS) $(MSG_FLAGS) $(F_FLAGS) $(WARN_FLAGS) $(INC_FLAGS) $(M_FLAGS) $(LANG_FLAGS) $(CPU_FLAGS)

override CXXFLAGS += -O3 $(ALL_FLAGS)
# Static so we can run the binary on other systems. whole-archive ensures
# all objects in the archive are included - required for pthread weak symbols.
override LDFLAGS += -s -Wl,--whole-archive -lpthread -Wl,--no-whole-archive -static -static-libgcc -static-libstdc++

PIK_OBJS := $(addprefix obj/, \
	simd/targets.o \
	ac_predictions.o \
	ac_strategy.o \
	adaptive_quantization.o \
	adaptive_reconstruction.o \
	epf.o \
	alpha.o \
	ans_common.o \
	ans_decode.o \
	ans_encode.o \
	arch_specific.o \
	brotli.o \
	butteraugli/butteraugli.o \
	butteraugli_comparator.o \
	butteraugli_distance.o \
	codec_impl.o \
	codec_png.o \
	codec_pnm.o \
	color_correlation.o \
	color_encoding.o \
	color_management.o \
	compressed_image.o \
	context_map_encode.o \
	context_map_decode.o \
	data_parallel.o \
	dct.o \
	dct_util.o \
	dc_predictor.o \
	deconvolve.o \
	descriptive_statistics.o \
	external_image.o \
	gauss_blur.o \
	gaborish.o \
	gradient_map.o \
	headers.o \
	image.o \
	linalg.o \
	lossless16.o \
	lossless8.o \
	pik.o \
	pik_info.o \
	pik_pass.o \
	pik_multipass.o \
	huffman_decode.o \
	huffman_encode.o \
	image_io.o \
	lehmer_code.o \
	metadata.o \
	noise.o \
	entropy_coder.o \
	opsin_inverse.o \
	opsin_image.o \
	opsin_params.o \
	os_specific.o \
	padded_bytes.o \
	quantizer.o \
	quant_weights.o \
	single_image_handler.o \
	tile_flow.o \
	upscaler.o \
	yuv_convert.o \
	saliency_map.o \
)

all: $(addprefix bin/, cpik dpik butteraugli_main)

# print an error message with helpful instructions if the brotli git submodule
# is not checked out
ifeq (,$(wildcard third_party/brotli/c/include/brotli/decode.h))
$(error Submodules are required to make pik, run "git submodule update --init" to get them, then run make again)
endif

third_party/brotli/libbrotli.a:
	$(MAKE) -C third_party/brotli lib

third_party/lcms/src/.libs/liblcms2.a:
	cd third_party/lcms; ./configure; cd ..
	$(MAKE) -C third_party/lcms

THIRD_PARTY := \
	third_party/brotli/libbrotli.a \
	third_party/lcms/src/.libs/liblcms2.a \
	third_party/lodepng/lodepng.o

bin/cpik: obj/cpik.o $(PIK_OBJS) $(THIRD_PARTY)
bin/dpik: obj/dpik.o $(PIK_OBJS) $(THIRD_PARTY)
bin/butteraugli_main: obj/butteraugli_main.o $(PIK_OBJS) $(THIRD_PARTY)
bin/decode_and_encode: obj/decode_and_encode.o $(PIK_OBJS) $(THIRD_PARTY)

obj/%.o: %.cc
	@mkdir -p -- $(dir $@)
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) $(SIMD_FLAGS) $< -o $@

bin/%: obj/%.o
	@mkdir -p -- $(dir $@)
	$(CXX) $^ -o $@ $(LDFLAGS)

.DELETE_ON_ERROR:
deps.mk: $(wildcard *.cc) $(wildcard *.h) Makefile
	set -eu; for file in *.cc; do \
		target=obj/$${file##*/}; target=$${target%.*}.o; \
		$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) -MM -MT \
		"$$target" "$$file"; \
	done >$@
-include deps.mk

clean:
	[ ! -d obj ] || $(RM) -r -- obj/
	[ ! -d bin ] || $(RM) -r -- bin/
	[ ! -d lib ] || $(RM) -r -- lib/
	$(MAKE) -C third_party/brotli clean

.PHONY: clean all install third_party/brotli/libbrotli.a
