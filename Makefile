PNG_FLAGS := $(shell pkg-config --cflags libpng)
PNG_LIBS := $(shell pkg-config --libs libpng)

override CXXFLAGS += -std=c++11 -Wall -O3 -fPIC -mavx2 -mfma -mlzcnt -mbmi2 -I. -Wno-sign-compare
override LDFLAGS += $(PNG_LIBS) -ljpeg -lpthread

PIK_OBJS := $(addprefix obj/, \
	adaptive_quantization.o \
	ans_encode.o \
	arch_specific.o \
	butteraugli/butteraugli.o \
	butteraugli_comparator.o \
	cache_aligned.o \
	compressed_image.o \
	context_map_encode.o \
	context_map_decode.o \
	dct.o \
	dc_predictor.o \
	gamma_correct.o \
	header.o \
	pik.o \
	huffman_decode.o \
	huffman_encode.o \
	histogram_decode.o \
	histogram_encode.o \
	image_io.o \
	lehmer_code.o \
	opsin_codec.o \
	opsin_inverse.o \
	opsin_image.o \
	quantizer.o \
	yuv_convert.o \
)

all: $(addprefix bin/, cpik dpik)

bin/cpik: $(PIK_OBJS) obj/cpik.o
bin/dpik: $(PIK_OBJS) obj/dpik.o

obj/%.o: %.cc
	@mkdir -p -- $(dir $@)
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) $(PNG_FLAGS) $< -o $@

bin/%: obj/%.o
	@mkdir -p -- $(dir $@)
	$(CXX) $^ -o $@ $(LDFLAGS)

.DELETE_ON_ERROR:
deps.mk: $(wildcard *.cc) $(wildcard *.h) Makefile
	set -eu; for file in *.cc; do \
		target=obj/$${file##*/}; target=$${target%.*}.o; \
		$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) -MM -MT \
		"$$target" "$$file"; \
	done | sed -e ':b' -e 's-../[^./]*/--' -e 'tb' >$@
-include deps.mk

clean:
	[ ! -d obj ] || $(RM) -r -- obj/
	[ ! -d bin ] || $(RM) -r -- bin/
	[ ! -d lib ] || $(RM) -r -- lib/

.PHONY: clean all install
