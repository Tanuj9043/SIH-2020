import sys, termios, atexit
from select import select

# save the terminal settings
fd = sys.stdin.fileno()
new_term = termios.tcgetattr(fd)
old_term = termios.tcgetattr(fd)

# new terminal setting unbuffered
new_term[3] = (new_term[3] & ~termios.ICANON & ~termios.ECHO)

# switch to normal terminal
def set_normal_term():
    termios.tcsetattr(fd, termios.TCSAFLUSH, old_term)

# switch to unbuffered terminal
def set_curses_term():
    termios.tcsetattr(fd, termios.TCSAFLUSH, new_term)

def putch(ch):
    sys.stdout.write(ch)

def getch():
    return sys.stdin.read(1)

def getche():
    ch = getch()
    putch(ch)
    return ch

def kbhit():
    dr,dw,de = select([sys.stdin], [], [], 0)
    return dr

def keyPressStart():
    atexit.register(set_normal_term)
    set_curses_term()

def keyPressEnd():
    atexit.register(set_curses_term)
    set_normal_term()

if __name__ == '__main__':
    keyPressStart()

    while 1:
        if kbhit():
            ch = getch()
            print(chr(ord(ch)+1))
            if ch=='q':
                break

    print('done')

    keyPressEnd()
    #necessary to call this function to allow input to be taken the same way as before
    print('here')
    x=input()
    print(x)
    print('there')

