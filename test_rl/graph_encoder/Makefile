dir_guard = @mkdir -p $(@D)
FIND := find
CXX := g++

CXXFLAGS += -Wall -O3 -std=c++11
LDFLAGS += -lm

include_dirs = ./include

CXXFLAGS += $(addprefix -I,$(include_dirs)) -Wno-unused-local-typedef
CXXFLAGS += -fPIC
cpp_files = $(shell $(FIND) src/lib -name "*.cpp" -print | rev | cut -d"/" -f1 | rev)
cxx_obj_files = $(subst .cpp,.o,$(cpp_files))

objs = $(addprefix build/lib/,$(cxx_obj_files))
DEPS = $(objs:.o=.d)

target = build/dll/libs2v.so
target_dep = $(addsuffix .d,$(target))

.PRECIOUS: build/lib/%.o

all: $(target)

build/dll/libs2v.so : src/s2v_lib.cpp $(objs)
	$(dir_guard)
	$(CXX) -shared $(CXXFLAGS) -MMD -o $@ $(filter %.cpp %.o, $^) $(LDFLAGS)

DEPS += $(target_dep)

build/lib/%.o: src/lib/%.cpp
	$(dir_guard)
	$(CXX) $(CXXFLAGS) -MMD -c -o $@ $(filter %.cpp, $^)

clean:
	rm -rf build

-include $(DEPS)
