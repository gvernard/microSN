ACLOCAL_AMFLAGS=-I m4



BUILT_SOURCES = libmagmap.la libcommon.la libgpu.la
noinst_LTLIBRARIES = libmagmap.la libcommon.la libgpu.la


# MAP PATH
libmagmap_la_SOURCES = src/magnification_map.cpp
libmagmap_la_CXXFLAGS = -std=c++17 -fPIC -g -frounding-math -DMAP_PATH='"@MAP_PATH@"'
libmagmap_la_CPPFLAGS = -Iinclude



# Common functions
libcommon_la_SOURCES = \
	src/mpd.cpp \
	src/profile.cpp \
	src/image.cpp
libcommon_la_CXXFLAGS = -std=c++17 -fPIC -g -frounding-math
libcommon_la_CPPFLAGS = -Iinclude
libcommon_la_LIBADD = -lmagmap



# GPU functions
.cu.o:
	@NVCC@ @CUDA_CFLAGS@ -o $@ -c $<
.cu.lo:
	python $(top_srcdir)/cudalt.py $@ @NVCC@ \
	--compiler-options=\"$(CFLAGS) $(DEFAULT_INCLUDES) $(INCLUDES) $(AM_CPPFLAGS) $(CPPFLAGS) @CUDA_LIBS@\" -c $<

LINK = $(LIBTOOL) --mode=link $(CC) -o $@ @CUDA_LIBS@ $(CPPFLAGS)
libgpu_la_SOURCES = \
	src/variability_models.cu
libgpu_la_CPPFLAGS = -Iinclude
libgpu_la_LIBADD  = @CUDA_LIBS@ -lcommon




bin_PROGRAMS = microlc
microlc_SOURCES = src/driver.cpp
microlc_CXXFLAGS = -std=c++17 -fopenmp -g -frounding-math
microlc_CPPFLAGS = -Iinclude
microlc_LDADD = -lmagmap -lcommon -lgpu



install-exec-hook:
	rm $(bin_PROGRAMS) $(bin_SCRIPTS)




