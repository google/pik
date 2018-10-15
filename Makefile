SIMD_FLAGS := -march=haswell
ROI_DETECTOR_FLAGS := -DROI_DETECTOR_OPENCV=1
OPENCVDIR = ${OPENCV_STATIC_INSTALL_DIR}

CXX ?= clang++
CC ?= clang

INC_FLAGS = -I. -I../ -Ithird_party/brotli/c/include/

# Clang-specific flags, tested with 6.0.1
M_FLAGS = -mrelax-all -Xclang -mrelocation-model -Xclang pic -Xclang -pic-level -Xclang 2 -pthread -mthread-model posix -Xclang -mdisable-fp-elim -Xclang -mconstructor-aliases -mpie-copy-relocations -Xclang -munwind-tables -Xclang -mprefer-vector-width=128

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
	guetzli/dct_double.o \
	guetzli/idct.o \
	guetzli/jpeg_data.o \
	guetzli/jpeg_data_decoder.o \
	guetzli/jpeg_data_reader.o \
	guetzli/jpeg_huffman_decode.o \
	guetzli/output_image.o \
	guetzli/quantize.o \
	third_party/lodepng/lodepng.o \
	adaptive_quantization.o \
	adaptive_reconstruction.o \
	af_edge_preserving_filter.o \
	af_stats.o \
	ans_common.o \
	ans_decode.o \
	ans_encode.o \
	arch_specific.o \
	brotli.o \
	brunsli_v2_decode.o \
	brunsli_v2_encode.o \
	butteraugli/butteraugli.o \
	butteraugli_comparator.o \
	butteraugli_distance.o \
	codec_impl.o \
	codec_png.o \
	codec_pnm.o \
	color_encoding.o \
	color_management.o \
	compressed_image.o \
	container.o \
	context.o \
	context_map_encode.o \
	context_map_decode.o \
	dct.o \
	dct_util.o \
	dc_predictor.o \
	deconvolve.o \
	external_image.o \
	gauss_blur.o \
	header.o \
  image.o \
	linalg.o \
	pik.o \
	pik_alpha.o \
	pik_info.o \
	huffman_decode.o \
	huffman_encode.o \
	image_io.o \
	jpeg_quant_tables.o \
	lehmer_code.o \
	noise.o \
	entropy_coder.o \
	opsin_inverse.o \
	opsin_image.o \
	opsin_params.o \
	os_specific.o \
	padded_bytes.o \
	quantizer.o \
	tile_flow.o \
	upscaler.o \
	yuv_convert.o \
)

all: $(addprefix bin/, cpik dpik cpik_roi butteraugli_main decode_and_encode)

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

THIRD_PARTY := third_party/brotli/libbrotli.a third_party/lcms/src/.libs/liblcms2.a

bin/cpik: obj/cpik.o $(PIK_OBJS) $(THIRD_PARTY)
bin/dpik: obj/dpik.o $(PIK_OBJS) $(THIRD_PARTY)
bin/butteraugli_main: obj/butteraugli_main.o $(PIK_OBJS) $(THIRD_PARTY)
bin/decode_and_encode: obj/decode_and_encode.o $(PIK_OBJS) $(THIRD_PARTY)

ifeq ($(OPENCVDIR),)
bin/cpik_roi:
	@echo "Not building cpik_roi, since OpenCV is not available."
	@echo "Please set ENV variable OPENCV_STATIC_INSTALL_DIR"
	@echo "to build the OpenCV-enabled encoder."
else
obj/libroi_detector_opencv.o:
	@mkdir -p obj
	# Using -std=c++14 for modern std::make_unique, but only for the opencv-enabled code.
	$(CXX) -o $@ -c $(CPPFLAGS) $(CXXFLAGS) -std=c++14 roi_detector_opencv.cc \
	-I $(OPENCVDIR)/include \
	-I $(OPENCVDIR)/include/opencv \
	-I $(OPENCVDIR)/include/opencv2 \

bin/cpik_roi: $(PIK_OBJS) obj/adaptive_quantization_roi.o obj/libroi_detector_opencv.o obj/cpik.o $(THIRD_PARTY)
	$(CXX) $^  -o $@ $(LDFLAGS) \
        $(OPENCVDIR)/lib/libopencv_imgcodecs.a \
	$(OPENCVDIR)/lib/libopencv_objdetect.a \
	$(OPENCVDIR)/lib/libopencv_imgproc.a \
	$(OPENCVDIR)/lib/libopencv_core.a \
	$(OPENCVDIR)/share/OpenCV/3rdparty/lib/libippiw.a \
	$(OPENCVDIR)/share/OpenCV/3rdparty/lib/libippicv.a \
	$(OPENCVDIR)/share/OpenCV/3rdparty/lib/libittnotify.a \
	-latomic -lz -lpthread -ldl
endif


obj/adaptive_quantization_roi.o: adaptive_quantization.cc
	@mkdir -p -- $(dir $@)
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) $(SIMD_FLAGS) $(ROI_DETECTOR_FLAGS) $< -o obj/adaptive_quantization_roi.o

obj/%.o: %.cc
	@mkdir -p -- $(dir $@)
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) $(SIMD_FLAGS) $< -o $@

bin/%: obj/%.o
	@mkdir -p -- $(dir $@)
	$(CXX) $^ -o $@ $(LDFLAGS)

.DELETE_ON_ERROR:
deps.mk: $(wildcard *.cc) $(wildcard *.h) Makefile
	set -eu; for file in *.cc; do \
		if [ "$$file" = roi_detector_opencv.cc -a -z "$(OPENCVDIR)" ]; then \
			continue \
		fi \
		target=obj/$${file##*/}; target=$${target%.*}.o; \
		$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) -MM -MT \
		"$$target" "$$file"; \
	done | sed -e ':b' -e 's-\.\./[^\./]*/--' -e 'tb' >$@
-include deps.mk

clean:
	[ ! -d obj ] || $(RM) -r -- obj/
	[ ! -d bin ] || $(RM) -r -- bin/
	[ ! -d lib ] || $(RM) -r -- lib/
	$(MAKE) -C third_party/brotli clean

.PHONY: clean all install third_party/brotli/libbrotli.a
