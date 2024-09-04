import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import argparse
import sounddevice as sd
import numpy as np
import pygame
import colors
from spectrum import draw_spectrum
from scope import draw_scope

# pygame setup
pygame.init()
screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()
running = True
fftsize = 512
data = None
spectrum = None

def callback(indata, frames, time, status):
    if any(indata):
        global data
        data = indata[:, 0]
        F = np.fft.rfft(indata[0:fftsize, 0] * np.hanning(fftsize))
        global spectrum
        spectrum = (F*F.conjugate()).real / fftsize ** 2
        print(frames)
    else:
        print('no input')

def main():
    device_info = sd.query_devices(args.device, 'input')
    samplerate = int(device_info['default_samplerate'])

    print('samplerate:', samplerate)

    with sd.InputStream(device=args.device, channels=1, callback=callback,
                        blocksize=512, samplerate=samplerate):

        global running
        while running:
            # poll for events
            # pygame.QUIT event means the user clicked X to close your window
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            w, h = pygame.display.get_surface().get_size()

            # fill the screen with a color to wipe away anything from last frame
            screen.fill(colors.GRAY_900)

            draw_spectrum(screen, spectrum, pygame.Rect(24, h/2 + 4, w - 48, h/2 - 28))
            draw_scope(screen, data, pygame.Rect(24, 24, w - 48, h/2 - 28))

            # flip() the display to put your work on screen
            pygame.display.flip()

            clock.tick(60)  # limits FPS to 60

        pygame.quit()


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

usage_line = ' press <enter> to quit'

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    '-l', '--list-devices', action='store_true',
    help='show list of audio devices and exit')
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)
parser = argparse.ArgumentParser(
    description='\n\nSupported keys:' + usage_line,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser])
parser.add_argument(
    '-b', '--block-duration', type=float, metavar='DURATION', default=50,
    help='block size (default %(default)s milliseconds)')
parser.add_argument(
    '-d', '--device', type=int_or_str,
    help='input device (numeric ID or substring)')
parser.add_argument(
    '-r', '--range', type=float, nargs=2,
    metavar=('LOW', 'HIGH'), default=[100, 2000],
    help='frequency range (default %(default)s Hz)')
args = parser.parse_args(remaining)
low, high = args.range
if high <= low:
    parser.error('HIGH must be greater than LOW')

if __name__ == '__main__':
    main()
