override CPPFLAGS += -I. -I../ -Ithird_party/brotli/c/include//..
override CXXFLAGS += -std=c++11 -Wall -O3 -fPIC

TEST_OBJS := $(addprefix obj/, \
	dispatch.o \
	simd_test_target_sse4.o \
	simd_test_target_avx2.o \
	simd_test_target_none.o \
)

all: $(addprefix bin/, simd_test custom_dispatcher_test)

obj/%.o: %.cc
	@mkdir -p -- $(dir $@)
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) $< -o $@

bin/%: obj/%.o
	@mkdir -p -- $(dir $@)
	$(CXX) $(LDFLAGS) $^ -o $@

.DELETE_ON_ERROR:
deps.mk: $(wildcard *.cc) $(wildcard *.h) Makefile
	set -eu; for file in *.cc; do \
		target=obj/$${file##*/}; target=$${target%.*}.o; \
		$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) -DSIMD_DEPS -MM -MT \
		"$$target" "$$file"; \
	done | sed -e ':b' -e 's-../[^./]*/--' -e 'tb' >$@
-include deps.mk

bin/simd_test: $(TEST_OBJS)

# (Compiled from same source file with different compiler flags)
obj/simd_test_target_avx2.o: CXXFLAGS+=-msse4.2 -maes -mpclmul -mavx2 -mfma
obj/simd_test_target_sse4.o: CXXFLAGS+=-msse4.2 -maes -mpclmul

# No special compile flags needed.
bin/custom_dispatcher_test: obj/dispatch.o

clean:
	[ ! -d obj ] || $(RM) -r -- obj/
	[ ! -d bin ] || $(RM) -r -- bin/
	[ ! -d lib ] || $(RM) -r -- lib/

.PHONY: clean all
