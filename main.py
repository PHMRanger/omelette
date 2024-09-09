from collections import deque
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import argparse
import sounddevice as sd
import numpy as np
import pygame
import colors
from spectrum import draw_spectrum
from scope import draw_scope, draw_traces
from buffer import Buffer

# pygame setup
pygame.init()
screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()
running = True
blocksize = 1024
fftsize = 512
succ_samples = 512
data = deque()
spectrum = deque()
use_trigger = True

triggered_index = None
was_one = False

buffer = Buffer()

def callback(indata, frames, time, status):
    if not use_trigger:
        i = 0
        while i + fftsize < indata.shape[0]:
            f = np.fft.rfft(indata[i:fftsize+i, 0] * np.hanning(fftsize)) 
            power = (f*f.conjugate()).real / fftsize ** 2
            spectrum.append(power)
            i += fftsize
        data.append(indata.copy())
    else:
        if indata.shape[1] < 2:
            return

        global last_sample
        global triggered_index
        global previous
        global was_one
        global background
        
        i = buffer.end + 1
        start = buffer.end + 1
        buffer.extend(indata[:, 0])
        end = buffer.end

        while i < end:
            if triggered_index is None:
                if indata[i - start, 1] > 0.1:
                    was_one = True

                if was_one and indata[i - start, 1] < -0.1:
                    triggered_index = i + 100 
                    was_one = False

            if triggered_index is not None:
                if buffer.end - triggered_index >= succ_samples:
                    zeros = np.zeros(fftsize)
                    slice = buffer.pop_slice(triggered_index, triggered_index + succ_samples)
                    zeros[:succ_samples] = slice * np.hanning(succ_samples)
                    f = np.fft.rfft(zeros) 
                    power = (f*f.conjugate()).real / fftsize ** 2
                    spectrum.append(power)
                    data.append(np.expand_dims(zeros, axis=1))
                    triggered_index = None
                    i += succ_samples - 1
                else:
                    break

            i += 1

def main():
    device_info = sd.query_devices(args.device, 'input')
    samplerate = int(device_info['default_samplerate'])
    channels = int(device_info['max_input_channels'])

    bg_index = None
    bg_spectrum = None
    spectrum_index = 0
    gui_spectrum = np.zeros((fftsize // 2 + 1, 16))
    gui_scope = None

    font = pygame.font.SysFont('Arial', 16)

    print('samplerate:', samplerate)

    with sd.InputStream(device=args.device, channels=2, callback=callback,
                        blocksize=blocksize, samplerate=samplerate):

        global running
        global use_trigger
        global background
        while running:
            # poll for events
            # pygame.QUIT event means the user clicked X to close your window
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYUP and event.key == pygame.K_t:
                    use_trigger = not use_trigger
                elif event.type == pygame.KEYUP and event.key == pygame.K_b:
                    if bg_spectrum is None:
                        bg_index = 0
                        bg_spectrum = np.zeros((fftsize // 2 + 1, 64))
                    else:
                        bg_spectrum = None
                        bg_index = None

            w, h = pygame.display.get_surface().get_size()

            # fill the screen with a color to wipe away anything from last frame
            screen.fill(colors.GRAY_900)

            if len(spectrum) > 0:
                x = spectrum.pop()
                gui_spectrum[:, spectrum_index] = x
                spectrum_index = (spectrum_index + 1) % gui_spectrum.shape[1]
                if bg_index is not None and bg_spectrum is not None:
                    bg_spectrum[:, bg_index] = x
                    bg_index = bg_index + 1
                    if bg_index == bg_spectrum.shape[1]:
                        bg_index = None
            if len(data) > 0:
                gui_scope = data.pop()

            mean_spectrum = np.mean(gui_spectrum, axis=1)
            if bg_spectrum is not None and bg_index is None:
                mean_spectrum = mean_spectrum - np.mean(bg_spectrum, axis=1)
                mean_spectrum = np.abs(mean_spectrum)

            draw_spectrum(screen, mean_spectrum, pygame.Rect(24, h/2 + 4, w - 48, h/2 - 28))

            scope_rect = pygame.Rect(24, 24, w - 48, h/2 - 28)
            draw_scope(screen, scope_rect)
            draw_traces(screen, gui_scope, scope_rect)

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
