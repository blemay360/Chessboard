MCU=atmega328p
CFLAGS=-g -Wall -mcall-prologues -mmcu=$(MCU) -Os
LDFLAGS=-Wl,-gc-sections -Wl,-relax
CC=avr-gcc
TARGET=ControlBoard
OBJECT_FILES=ControlBoard.o

all: $(TARGET).hex

clean:
	@echo Cleaning up...
	rm -f *.o *.hex *.obj *.hex

%.hex: %.obj
	avr-objcopy -R .eeprom -O ihex $< $@

%.obj: $(OBJECT_FILES)
	$(CC) $(CFLAGS) $(OBJECT_FILES) $(LDFLAGS) -o $@

program: $(TARGET).hex
	avrdude -c usbtiny -B 32 -p $(MCU) -U flash:w:$(TARGET).hex:i
# To get rid of 8x clock scale down: -U lfuse:w:0xe2:m
