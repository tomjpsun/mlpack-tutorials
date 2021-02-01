# This is a simple Makefile used to build the example source code.
# This example might requires some modifications in order to work correctly on
# your system.
# If you're not using the Armadillo wrapper, replace `armadillo` with linker commands
# for the BLAS and LAPACK libraries that you are using.

TARGET := demo

ODIR=obj
SRCS := feed_forward.cpp rnn_model.cpp
OBJS=$(patsubst %.cpp, $(ODIR)/%.o, $(SRCS))
DEPS=$(patsubst %.cpp, $(ODIR)/%.d, $(SRCS))

LIBS_NAME := armadillo mlpack
LIBS := $(addprefix -l,$(LIBS_NAME))

CXX := g++
CXXFLAGS = -I . -std=c++11 -Wall -Wextra -O3 -DNDEBUG -fopenmp
# Use these CXXFLAGS instead if you want to compile with debugging symbols and
# without optimizations.
# CXXFLAGS += -std=c++11 -Wall -Wextra -g -O0
LDFLAGS  += -fopenmp
LDFLAGS += -L . # /path/to/mlpack/library/  # if installed locally.

CLEAN_LIST := $(TARGET) $(OBJS) *.csv

.PHONY: all
all: 	$(TARGET)


# Objects generation
$(ODIR)/%.o:    %.cpp
	@$(call make-dirs)
	$(CXX) $(CXXFLAGS) $(INCFLAGS) -c -MMD $< -o $@

# Generating dependencies
$(ODIR)/%.d:    %.cpp
	@$(call make-dirs)
	$(CXX) -M $(CXXFLAGS) $(INCFLAGS) $< > $@

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $@ $(LDFLAGS) $(LIBS)

clean:
	@echo CLEAN $(CLEAN_LIST)
	@rm -f $(CLEAN_LIST)

# Create directories
makedirs:
	@$(call make-dirs)

ifneq ($(MAKECMDGOALS), clean)
-include $(DEPS)
endif

define make-dirs
	for dir in $(ODIR);\
	do\
		mkdir -p $$dir;\
	done
endef
