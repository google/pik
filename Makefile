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

INC_FLAGS = -I. -I../ -Ithird_party/brotli/c/include/ -Ithird_party/lcms/include/ -Ithird_party/

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

# Flags only used by tests
TEST_FLAGS = -Ithird_party/googletest/googletest/include/ -Ithird_party/googletest/googlemock/include/ -Ithird_party/benchmark/include/

override CXXFLAGS += -O3 $(ALL_FLAGS)
# Static so we can run the binary on other systems. whole-archive ensures
# all objects in the archive are included - required for pthread weak symbols.
override LDFLAGS += -s -Wl,--whole-archive -lpthread -Wl,--no-whole-archive -static -static-libgcc -static-libstdc++

PIK_OBJS := $(addprefix obj/pik/, \
	simd/targets.o \
	ac_predictions.o \
	ac_strategy.o \
	adaptive_quantization.o \
	adaptive_reconstruction.o \
	alpha.o \
	ans_common.o \
	ans_decode.o \
	ans_encode.o \
	ar_control_field.o \
	arch_specific.o \
	bilinear_transform.o \
	block_dictionary.o \
	brotli.o \
	butteraugli/butteraugli.o \
	butteraugli_comparator.o \
	butteraugli_distance.o \
	cache_aligned.o \
	codec_impl.o \
	codec_png.o \
	codec_pnm.o \
	color_correlation.o \
	color_encoding.o \
	color_management.o \
	compressed_dc.o \
	compressed_image.o \
	context_map_decode.o \
	context_map_encode.o \
	data_parallel.o \
	dc_predictor.o \
	dct.o \
	dct_util.o \
	deconvolve.o \
	descriptive_statistics.o \
	detect_dots.o \
	entropy_coder.o \
	epf.o \
	external_image.o \
	gaborish.o \
	gauss_blur.o \
	gradient_map.o \
	headers.o \
	huffman_decode.o \
	huffman_encode.o \
	image.o \
	lehmer_code.o \
	linalg.o \
	lossless16.o \
	lossless8.o \
	lossless_entropy.o \
	metadata.o \
	noise.o \
	opsin_image.o \
	opsin_inverse.o \
	opsin_params.o \
	os_specific.o \
	padded_bytes.o \
	pik.o \
	pik_frame.o \
	pik_info.o \
	quant_weights.o \
	quantizer.o \
	saliency_map.o \
	single_image_handler.o \
	status.o \
	upscaler.o \
	yuv_convert.o \
)

all: deps.mk $(addprefix bin/, cpik dpik butteraugli_main decode_and_encode)

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

third_party/FiniteStateEntropy/libfse.a: $(FSE_OBJS)
	$(MAKE) -C third_party fse

THIRD_PARTY := \
	third_party/brotli/libbrotli.a \
	third_party/lcms/src/.libs/liblcms2.a \
	third_party/lodepng/lodepng.o \
	third_party/FiniteStateEntropy/libfse.a

BIN_OBJS := $(addprefix obj/pik/, \
  butteraugli_main.o \
  cmdline.o \
  cpik.o \
  cpik_main.o \
  decode_and_encode.o \
  dpik.o \
  dpik_main.o \
)

bin/cpik: obj/pik/cpik_main.o obj/pik/cpik.o obj/pik/cmdline.o $(PIK_OBJS) $(THIRD_PARTY)
bin/dpik: obj/pik/dpik_main.o obj/pik/dpik.o obj/pik/cmdline.o $(PIK_OBJS) $(THIRD_PARTY)
bin/butteraugli_main: obj/pik/butteraugli_main.o $(PIK_OBJS) $(THIRD_PARTY)
bin/decode_and_encode: obj/pik/decode_and_encode.o $(PIK_OBJS) $(THIRD_PARTY)

obj/%.o: %.cc
	@mkdir -p -- $(dir $@)
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) $(SIMD_FLAGS) $< -o $@

bin/%: obj/pik/%.o
	@mkdir -p -- $(dir $@)
	$(CXX) $^ -o $@ $(LDFLAGS)

.DELETE_ON_ERROR:
obj/%.o.d: %.cc Makefile
	@mkdir -p $(@D)
	@# We set both the .o file and the .o.d file as targets so the dependencies
	@# file gets rebuilt as well when the code changes.
	@$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) $(TEST_FLAGS) -MM -MT "$(@:.d=), $@" "$<" \
	  >$@

ALL_DEPS := $(addsuffix .d,$(BIN_OBJS) $(PIK_OBJS))
deps.mk: $(ALL_DEPS) Makefile
	cat $(ALL_DEPS) > deps.mk
-include deps.mk

clean:
	[ ! -d obj ] || $(RM) -r -- obj/
	[ ! -d bin ] || $(RM) -r -- bin/
	[ ! -d lib ] || $(RM) -r -- lib/
	$(MAKE) -C third_party/brotli clean

.PHONY: clean all install third_party/brotli/libbrotli.a
