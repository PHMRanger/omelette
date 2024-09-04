import pygame
import colors

def draw_scope(screen, data, rect: pygame.Rect):
    pygame.draw.rect(screen, colors.GRAY_800, rect)
    if data is not None:
        points = [(
            float(i) / data.shape[0] * rect.width + rect.x,
            float(data[i] * rect.height / 2) + rect.y + rect.height / 2
        ) for i in range(0, data.shape[0])]
        pygame.draw.lines(screen, colors.PURPLE_800, False, points, width=2)
        
