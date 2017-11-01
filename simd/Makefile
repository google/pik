override CPPFLAGS += -I. -I../ -Ithird_party/brotli/c/include//..
override CXXFLAGS += -std=c++11 -Wall -O3 -fPIC

TEST_OBJS := $(addprefix obj/, \
	dispatch.o \
	simd_test_target_sse4.o \
	simd_test_target_avx2.o \
	simd_test_target_none.o \
)

all: $(addprefix bin/, simd_test attr_test)

obj/dispatch.o: dispatch.cc
	@mkdir -p obj
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) dispatch.cc -o obj/dispatch.o

obj/attr_test.o: attr_test.cc
	@mkdir -p obj
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) attr_test.cc -o obj/attr_test.o

obj/simd_test_target_none.o: simd_test_target.cc
	@mkdir -p obj
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) simd_test_target.cc -o obj/simd_test_target_none.o

obj/simd_test_target_sse4.o: simd_test_target.cc
	@mkdir -p obj
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) -DSIMD_ENABLE=4 -msse4.2 -maes -mpclmul simd_test_target.cc -o obj/simd_test_target_sse4.o

obj/simd_test_target_avx2.o: simd_test_target.cc
	@mkdir -p obj
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) -DSIMD_ENABLE=6 -msse4.2 -maes -mpclmul -mavx2 -mfma simd_test_target.cc -o obj/simd_test_target_avx2.o

bin/%: obj/%.o
	@mkdir -p bin
	$(CXX) $(LDFLAGS) $^ -o $@

.DELETE_ON_ERROR:
deps.mk: $(wildcard *.cc) $(wildcard *.h) Makefile
	set -eu; for file in *.cc; do \
		target=obj/$${file##*/}; target=$${target%.*}.o; \
		$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) -DSIMD_DEPS=1 -MM -MT \
		"$$target" "$$file"; \
	done | sed -e ':b' -e 's-../[^./]*/--' -e 'tb' >$@
-include deps.mk

bin/simd_test: $(TEST_OBJS)

bin/attr_test: obj/dispatch.o

clean:
	[ ! -d obj ] || $(RM) -r -- obj/
	[ ! -d bin ] || $(RM) -r -- bin/
	[ ! -d lib ] || $(RM) -r -- lib/

.PHONY: clean all
