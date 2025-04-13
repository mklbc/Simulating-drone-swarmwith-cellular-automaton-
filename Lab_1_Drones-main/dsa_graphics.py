from PIL import Image, ImageDraw
import pygame
from dsa_automaton import GRID_H, GRID_W, PAD
from dsa_automaton import DR_HERE, DR_NEAR, DR_NONE, DR_VECT
from dsa_automaton import V_VISITED, V_UNVISITED
from dsa_automaton import L_OBST, L_DRON, L_VIS

# Visualization parameters
CELL_SIZE = 6   # Cell size in pixels
FPS = 20        # Screen update frequency limit

COLOR_OBST = (0, 0, 0)          # Black: color for obstacle
COLOR_NO_OBST = (255, 255, 255) # White: color for no obstacle; for initialization

COLOR_DRONE = (0, 0, 205)       # Blue: drone is in this cell
COLOR_DR_PR1 = (100,149,237)    # Dimmer blue: drone was last turn
COLOR_DR_PR2 = (135,206,250)    # Even dimmer blue: drone was here 2 turns ago
COLOR_DR_NEAR = (255, 0, 0)     # Red: drone can move here
COLOR_DR_VECT = (64, 224, 208)  # Bright blue: drone is going there

COLOR_VISIT = (50,205,50)        # This cell has been visited
COLOR_NO_VISIT = (255,165,0)     # This cell hasn't been visited
COLOR_UNREACHABLE = (255,99,71)  # This cell is unreachable


# Initialize Pygame
def init_pygame(grid_w, grid_h) -> pygame.Surface:
    pygame.init()
    screen = pygame.display.set_mode((grid_w * CELL_SIZE, grid_h * CELL_SIZE))
    screen.fill(COLOR_NO_OBST)
    pygame.display.set_caption("CA-Based Drone Swarm Simulation")
    return screen


# Helper function
def draw_cell(screen, x, y, color):
    pygame.draw.rect(screen, color, pygame.Rect(
                x * CELL_SIZE, 
                y * CELL_SIZE, 
                CELL_SIZE, 
                CELL_SIZE
            ))


# Visualization
def draw_grid(screen, new_grid, old_grid=None):
    for x in range(GRID_W):
        for y in range(GRID_H):
            obst, drone, visit, _, _, _ = new_grid[PAD+y, PAD+x, :]
            if old_grid is not None:
                _, drone_old, visit_old, _, _, _ = old_grid[PAD+y, PAD+x, :]
                # If nothing changed, skip this cell
                if drone == drone_old and visit == visit_old:
                    continue
            # 1. If cell contains obstacle - no need to draw anything else
            if obst:
                draw_cell(screen, x, y, COLOR_OBST)
                continue
            # 2. If cell contains drone or drone-related information:
            if drone != DR_NONE:
                if drone == DR_HERE:
                    # Drone is in this cell
                    draw_cell(screen, x, y, COLOR_DRONE)
                elif drone == DR_NEAR:
                    # This cell is near a drone
                    draw_cell(screen, x, y, COLOR_DR_NEAR)
                elif drone == DR_VECT:
                    # This cell is drone's vector point
                    draw_cell(screen, x, y, COLOR_DR_VECT)
                else:
                    # Unexprected value
                    print(f'Unexpected value in drone layer: {drone} at {x}, {y}')
                    return None
                continue
            # 3. In other cases, draw the visit value of the cell
            if visit == V_VISITED:
                draw_cell(screen, x, y, COLOR_VISIT)
            elif visit == V_UNVISITED:
                draw_cell(screen, x, y, COLOR_NO_VISIT)
            else:
                draw_cell(screen, x, y, COLOR_UNREACHABLE)
    # Update screen
    pygame.display.flip()
    return 1


# Report progress via window title
def observer(new_grid, old_grid, iter_count):
    np_visit = new_grid[:, :, L_VIS]
    unvisited_count = (np_visit == V_UNVISITED).sum()
    visited_count = (np_visit == V_VISITED).sum()
    visited_share = visited_count / (visited_count + unvisited_count)
    pygame.display.set_caption(f'CA-Based Drone Swarm Simulation: {visited_share * 100.0:.2f}% (iter. {iter_count})')

