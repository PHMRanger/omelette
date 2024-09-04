import pygame
import numpy as np
import colors

def draw_spectrum(screen, spectrum, rect: pygame.Rect, db_range=(0, -120)):
    inner_rect = pygame.Rect(rect.x, rect.y, rect.width, rect.height)
    pygame.draw.rect(screen, colors.GRAY_800, inner_rect)
    if spectrum is not None:
        dbs = 10. * np.log10(spectrum + 1e-30)
        l = spectrum.shape[0]
        for i in range(spectrum.shape[0]):
            tick_width = (inner_rect.width - 16) / l
            tick_height = (dbs[i] - db_range[1]) / (db_range[0] - db_range[1]) * inner_rect.height
            tick_color = (100, 100, min(100 + i, 255))
            tick_x = inner_rect.x + i * tick_width + 8
            tick_y = inner_rect.y - tick_height + rect.height

            pygame.draw.rect(screen, tick_color, (tick_x, tick_y, tick_width + 1, tick_height))
