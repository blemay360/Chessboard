/* ControlBoard.c, takes input from buttons and a pot, sends serial messages to pi and controls neopixel ring, LEDs, and 7-segment display */
#define F_CPU 8000000UL
#include <avr/io.h>
#include <avr/interrupt.h>
#include <util/delay.h>

#define BAUD 9600
#define MYUBRR F_CPU/16/BAUD-1

#define T0H 0.125
#define T0L 0.625
#define T1H 0.625
#define T1L 0.125

//---------------------------SPI FUNCTIONS---------------------------
/*
void SPI_MasterInit(void)
{
	DDRB =  (1 << DDB0) | (1 << DDB2) | (1 << DDB5); // Set MOSI, CS, and SCK output, all others input
	SPCR = (1<<SPE)|(1<<MSTR)|(1<<SPR0); // Enable SPI, Master, set clock rate fck/8
    SPSR = (1 << SPI2X);
}

void SPI_MasterTransmit(char cData)
{
    SPDR = cData; // Start transmission
    while(!(SPSR & (1<<SPIF)));  // Wait for transmission complete
}

void sendCommand(uint8_t addr, uint8_t value)
{
    PORTB &= ~(1 << PORTB2); // Set chip select low
    SPI_MasterTransmit(addr);
    SPI_MasterTransmit(value);
    _delay_ms(1);
    PORTB |= (1 << PORTB2); // Set chip select high
}
*/

//---------------------------GPIO FUNCTIONS---------------------------
void blink()
{
    int delay = 1000;
    
    for (uint8_t i = 0; i < 2; i++)
    {
        PORTC |= (1 << PORTC1);   // Turn the LED on
        _delay_ms(delay);
        PORTC &= ~(1 << PORTC1);  // Turn the LED off
        _delay_ms(delay);
    }
}

uint8_t read_button(uint8_t i)
{
    return PIND & (1<<(i + 2));
}

//---------------------------UART FUNCTIONS---------------------------
void USART_Init(unsigned int ubrr)
{
    // Set baud rate
    UBRR0H = (unsigned char)(ubrr>>8);
    UBRR0L = (unsigned char)ubrr;
    // Enable RX interrupt, receiver and transmitter pins
    UCSR0B = (1<<RXCIE0)|(1<<RXEN0)|(1<<TXEN0);
    // Set frame format: 8 data, 1 stop bit
    UCSR0C = (1<<UCSZ01)|(1<<UCSZ00);

    //Set Global Interrupt Enable
    sei(); //cli() to disable interrupts
}

void USART_Transmit(unsigned char data )
{
    // Wait for empty transmit buffer
    while ( !( UCSR0A & (1<<UDRE0)) )
    ;
    // Put data into buffer, sends the data
    UDR0 = data;
}

ISR(USART_RX_vect)
{
  char input = UDR0;
  USART_Transmit(input);
}

//---------------------------NEOPIXEL FUNCTIONS---------------------------
void neopixel_high(void)
{
    //To send a 1, signal should be:
    //high for 0.8 us
    //low for 0.45 us
    PORTD |= (1 << PORTD7);   // Set PD7 high
    _delay_us(0.5);
    PORTD &= ~(1 << PORTD7);  // Set PD7 low
    _delay_us(0.25);
}
    
void neopixel_low(void)
{
    //To send a 0, signal should be:
    //high for 0.4 us
    //low for 0.85 us
    PORTD |= (1 << PORTD7);   // Set PD7 high
    _delay_us(0.125);
    PORTD &= ~(1 << PORTD7);  // Set PD7 low
    _delay_us(0.625);
}

void neopixel_high_byte(void) {
    //All ones
    PORTD |= (1 << PORTD7);   // Set PD7 high
    _delay_us(T1H);
    PORTD &= ~(1 << PORTD7);  // Set PD7 low
    _delay_us(T1L);
    PORTD |= (1 << PORTD7);   // Set PD7 high
    _delay_us(T1H);
    PORTD &= ~(1 << PORTD7);  // Set PD7 low
    _delay_us(T1L);
    PORTD |= (1 << PORTD7);   // Set PD7 high
    _delay_us(T1H);
    PORTD &= ~(1 << PORTD7);  // Set PD7 low
    _delay_us(T1L);
    PORTD |= (1 << PORTD7);   // Set PD7 high
    _delay_us(T1H);
    PORTD &= ~(1 << PORTD7);  // Set PD7 low
    _delay_us(T1L);
    PORTD |= (1 << PORTD7);   // Set PD7 high
    _delay_us(T1H);
    PORTD &= ~(1 << PORTD7);  // Set PD7 low
    _delay_us(T1L);
    PORTD |= (1 << PORTD7);   // Set PD7 high
    _delay_us(T1H);
    PORTD &= ~(1 << PORTD7);  // Set PD7 low
    _delay_us(T1L);
    PORTD |= (1 << PORTD7);   // Set PD7 high
    _delay_us(T1H);
    PORTD &= ~(1 << PORTD7);  // Set PD7 low
    _delay_us(T1L);
    PORTD |= (1 << PORTD7);   // Set PD7 high
    _delay_us(T1H);
    PORTD &= ~(1 << PORTD7);  // Set PD7 low
    _delay_us(T1L);
}

void neopixel_low_byte(void)
{
    //All zeros
    PORTD |= (1 << PORTD7);   // Set PD7 high
    _delay_us(T0H);
    PORTD &= ~(1 << PORTD7);  // Set PD7 low
    _delay_us(T0L);
    PORTD |= (1 << PORTD7);   // Set PD7 high
    _delay_us(T0H);
    PORTD &= ~(1 << PORTD7);  // Set PD7 low
    _delay_us(T0L);
    PORTD |= (1 << PORTD7);   // Set PD7 high
    _delay_us(T0H);
    PORTD &= ~(1 << PORTD7);  // Set PD7 low
    _delay_us(T0L);
    PORTD |= (1 << PORTD7);   // Set PD7 high
    _delay_us(T0H);
    PORTD &= ~(1 << PORTD7);  // Set PD7 low
    _delay_us(T0L);
    PORTD |= (1 << PORTD7);   // Set PD7 high
    _delay_us(T0H);
    PORTD &= ~(1 << PORTD7);  // Set PD7 low
    _delay_us(T0L);
    PORTD |= (1 << PORTD7);   // Set PD7 high
    _delay_us(T0H);
    PORTD &= ~(1 << PORTD7);  // Set PD7 low
    _delay_us(T0L);
    PORTD |= (1 << PORTD7);   // Set PD7 high
    _delay_us(T0H);
    PORTD &= ~(1 << PORTD7);  // Set PD7 low
    _delay_us(T0L);
    PORTD |= (1 << PORTD7);   // Set PD7 high
    _delay_us(T0H);
    PORTD &= ~(1 << PORTD7);  // Set PD7 low
    _delay_us(T0L);
}

void clear_neopixels(void)
{
    for (uint8_t i = 0; i < 12; i++) {
        neopixel_low_byte();
        neopixel_low_byte();
        neopixel_low_byte();
        neopixel_low_byte();
    }
    _delay_ms(1000);
}

void set_neopixels(void)
{
    //GRBW
    for (uint8_t pixels_to_light = 1; pixels_to_light < 12; pixels_to_light++)
    {
        for (uint8_t i = 0; i <= pixels_to_light; i++) {
            neopixel_low_byte();
            neopixel_high_byte();
            neopixel_low_byte();
            neopixel_high_byte();
        }
        _delay_ms(100);
    }
}

//---------------------------MAIN FUNCTION---------------------------
int main()
{
    //Set direction registers
    DDRC |= (1 << DDC1) | (1 << DDC3);    // Make pin PC1 and PC3 be outputs
    DDRD |= (1 << DDD1) | (1 << DDD7);    // Make pin PD1 and PD7 be an output
    
    PORTC |= (1 << PORTC3);   // Set PC3 high
    
    clear_neopixels();
    
    _delay_ms(1000);
    
    //7-segment SPI control
    /*
    SPI_MasterInit();
    PORTB &= ~(1 << PORTB2); // Set chip select low
    
    sendCommand (12,1); //normal mode (default is shutdown mode)
    sendCommand (15,0); //Display test off
    sendCommand (10,15); //set medium intensity (range is 0-15)
    sendCommand (11, 3); //7219 digit scan limit command
    sendCommand (9, 0); //decode command, no decode
    
    PORTB |= (1 << PORTB2); // Set chip select high
    
    sendCommand (2, 2); //send commmand to digit two to display a single link
    */
    
    //Pi UART communication
    //USART_Init(MYUBRR);
    
    //unsigned char data = 0x75;
    
    
    while(1)
    {
        //Neopixel code
        
        set_neopixels();
        /*
        for (uint8_t pixel = 0; pixel < 12*24; pixel++){
            neopixel_high();
        }
        _delay_us(500);
        */
        
        //UART transmit code
        
//         USART_Transmit(data);
//         _delay_ms(1000);
        
        
        /*
        //Button press connection
        if (PIND & 0b01111100)
        {
            USART_Transmit(PIND);
            PORTC |= (1 << PORTC1);   // Turn the LED on
            _delay_ms(500);
        }
        else
        {
            PORTC &= ~(1 << PORTC1);  // Turn the LED off
        }
        */
    }
}
