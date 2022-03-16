CC=g++
CFLAGS=-O2 -std=c++11
SRCS=glean.cpp
PROG=glean

OPENCV=`pkg-config opencv --cflags --libs` # if you use opencv 4.x, you need to change 'opencv' -> 'opencv4'
INCS=-I../include
LIBS=$(OPENCV) -L../lib -lpthread -lSoyNet 

all: $(PROG)

$(PROG):$(SRCS)
	$(CC) $(CFLAGS) -o $(PROG) $(SRCS) $(LIBS) $(INCS)
 
clean:
	@echo Cleaning...
	rm -f $(PROG)
