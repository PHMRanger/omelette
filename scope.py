import pygame
import colors

def draw_scope(screen, rect: pygame.Rect):
    pygame.draw.rect(screen, colors.GRAY_800, rect)
        
def draw_trace(screen, data, color, rect):
    if data is None:
        return

    points = [(
        float(i) / data.shape[0] * rect.width + rect.x,
        float(data[i] * rect.height / 2) + rect.y + rect.height / 2
    ) for i in range(0, data.shape[0])]
    pygame.draw.lines(screen, color, False, points, width=2)

TRACE_COLORS = [colors.RED_200, colors.GREEN_200, colors.BLUE_200]

def draw_traces(screen, data, rect):
    if data is None:
        return

    for i in range(0, data.shape[1]):
        draw_trace(screen, data[:, i], TRACE_COLORS[i % len(TRACE_COLORS)], rect)
