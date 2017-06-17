
#  svrt is the ``Synthetic Visual Reasoning Test'', an image
#  generator for evaluating classification performance of machine
#  learning systems, humans and primates.
#
#  Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
#  Written by Francois Fleuret <francois.fleuret@idiap.ch>
#
#  This file is part of svrt.
#
#  svrt is free software: you can redistribute it and/or modify it
#  under the terms of the GNU General Public License version 3 as
#  published by the Free Software Foundation.
#
#  svrt is distributed in the hope that it will be useful, but
#  WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#  General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with svrt.  If not, see <http://www.gnu.org/licenses/>.

ifeq ($(DEBUG),yes)
 CXXFLAGS = -fPIC -Wall -g -DDEBUG
else
 CXXFLAGS = -fPIC -Wall -g -O3
endif

all: svrt TAGS

TAGS: *.cc *.h
	etags *.cc *.h

svrt:	libsvrt.so svrt.h svrt.c
	./build.py

libsvrt.so: \
	misc.o random.o \
	svrt_generator.o \
	shape.o vignette.o vignette_generator.o \
	$(patsubst %.cc,%.o,$(wildcard vision_problem_*.cc))
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -shared -fPIC -o $@ $^

Makefile.depend: *.h *.cc Makefile
	$(CC) $(CXXFLAGS) -M *.cc > Makefile.depend

clean:
	\rm -rf svrt *.o *.so Makefile.depend

-include Makefile.depend
