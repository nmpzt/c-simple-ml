CC = gcc
CFLAGS = -O

SRCS = $(wildcard *.c)

PROGS = $(SRCS:.c=)

all: $(PROGS)

%: %.c
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f $(PROGS) *.exe
	rm *.ppm
	rm ./models/*.bin
