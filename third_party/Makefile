# Copyright 2019 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

.PHONY: fse clean

FSE_OBJS := $(addprefix obj/, \
  FiniteStateEntropy/lib/entropy_common.o \
  FiniteStateEntropy/lib/fse_compress.o \
  FiniteStateEntropy/lib/fse_decompress.o \
)
FSE_LIB := FiniteStateEntropy/libfse.a

# Force include fse_wrapper.h in all the FSE objects.
override CFLAGS += -include fse_wrapper.h

override CXXFLAGS += -O3

fse: $(FSE_LIB)

$(FSE_LIB): $(FSE_OBJS)
	$(AR) cr $@ $^

obj/FiniteStateEntropy/lib/fse_decompress.o:
	@mkdir -p -- $(dir $@)
	$(CC) -c $(CFLAGS) $(CXXFLAGS) -include fse_error_wrapper.h $< -o $@

obj/%.o: %.c
	@mkdir -p -- $(dir $@)
	$(CC) -c $(CFLAGS) $(CXXFLAGS) $< -o $@

clean:
	rm -rf obj $(FSE_LIB)

.DELETE_ON_ERROR:
deps.mk: $(wildcard FiniteStateEntropy/lib/*.c) $(wildcard FiniteStateEntropy/lib/*.h) Makefile
	set -eu; for file in FiniteStateEntropy/lib/*.c; do \
	  target=obj/$${file}; target=$${target%.*}.o; \
	  $(CXX) -c $(CPPFLAGS) $(CXXFLAGS) -MM -MT \
	  "$$target" "$$file"; \
	done >$@
-include deps.mk
