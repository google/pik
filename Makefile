PNG_FLAGS := $(shell pkg-config --cflags libpng)
PNG_LIBS := $(shell pkg-config --libs libpng)

SIMD_FLAGS := -DSIMD_ENABLE=4 -msse4.2 -maes
override CXXFLAGS += -std=c++11 -Wall -O3 -fPIC -I. -I../ -Ithird_party/brotli/c/include/ -Wno-sign-compare
override LDFLAGS += $(PNG_LIBS) -ljpeg -lpthread

PIK_OBJS := $(addprefix obj/, \
	simd/dispatch.o \
	adaptive_quantization.o \
	af_edge_preserving_filter.o \
	af_edge_preserving_filter_none.o \
	af_stats.o \
	alpha_blend.o \
	ans_decode.o \
	ans_encode.o \
	arch_specific.o \
	butteraugli/butteraugli.o \
	butteraugli_comparator.o \
	butteraugli_distance.o \
	compressed_image.o \
	context_map_encode.o \
	context_map_decode.o \
	dct.o \
	dct_util.o \
	dc_predictor.o \
	gamma_correct.o \
	gauss_blur.o \
	header.o \
	linalg.o \
	pik.o \
	pik_alpha.o \
	pik_info.o \
	huffman_decode.o \
	huffman_encode.o \
	histogram.o \
	histogram_decode.o \
	histogram_encode.o \
	image_io.o \
	lehmer_code.o \
	noise.o \
	opsin_codec.o \
	opsin_inverse.o \
	opsin_image.o \
	opsin_params.o \
	os_specific.o \
	padded_bytes.o \
	quantizer.o \
	tile_flow.o \
	yuv_convert.o \
	yuv_opsin_convert.o \
)

all: $(addprefix bin/, cpik dpik butteraugli_main png2y4m y4m2png)

# print an error message with helpful instructions if the brotli git submodule
# is not checked out
ifeq (,$(wildcard third_party/brotli/c/include/brotli/decode.h))
$(error Brotli is required to make pik, run "git submodule init && git submodule update" to get it, then run make again)
endif

third_party/brotli/libbrotli.a:
	make -C third_party/brotli lib

obj/af_edge_preserving_filter_none.o: af_edge_preserving_filter.cc
	@mkdir -p obj
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) af_edge_preserving_filter.cc -o obj/af_edge_preserving_filter_none.o


bin/cpik: $(PIK_OBJS) obj/cpik.o third_party/brotli/libbrotli.a
bin/dpik: $(PIK_OBJS) obj/dpik.o third_party/brotli/libbrotli.a
bin/butteraugli_main: $(PIK_OBJS) obj/butteraugli_main.o third_party/brotli/libbrotli.a
bin/png2y4m: $(PIK_OBJS) obj/png2y4m.o third_party/brotli/libbrotli.a
bin/y4m2png: $(PIK_OBJS) obj/y4m2png.o third_party/brotli/libbrotli.a

obj/%.o: %.cc
	@mkdir -p -- $(dir $@)
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) $(SIMD_FLAGS) $(PNG_FLAGS) $< -o $@

bin/%: obj/%.o
	@mkdir -p -- $(dir $@)
	$(CXX) $^ -o $@ $(LDFLAGS)

.DELETE_ON_ERROR:
deps.mk: $(wildcard *.cc) $(wildcard *.h) Makefile
	set -eu; for file in *.cc; do \
		target=obj/$${file##*/}; target=$${target%.*}.o; \
		$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) -MM -MT \
		"$$target" "$$file"; \
	done | sed -e ':b' -e 's-\.\./[^\./]*/--' -e 'tb' >$@
-include deps.mk

clean:
	[ ! -d obj ] || $(RM) -r -- obj/
	[ ! -d bin ] || $(RM) -r -- bin/
	[ ! -d lib ] || $(RM) -r -- lib/
	make -C third_party/brotli clean

.PHONY: clean all install third_party/brotli/libbrotli.a
