import numpy as np
from PIL import Image
import random

# Grid parameters
GRID_W = 128
GRID_H = 128

# Drone vision and recon parameters
BIG_ZONE_R = 2   # How far, in cells, drone sees in general
DRONE_VIS_R = 1  # How far, in cells, drone makes cells "visited"

# Grid layers indices
L_OBST = 0  # Layer of obstacles: 0 - empty, 1 - occupied by an obstacle
L_DRON = 1  # Layer of drones
L_VIS = 2   # Layer of (un)visited cells
L_GRAD = 3  # Additional layer
L_COLL1 = 4 # Collision avoidance layer, a boolean plane of (x%2 == y%2)
L_COLL2 = 5 # Collision avoidance layer, a boolean plane of (x%2)

# Drone layer values
DR_HERE = 10 # Drone is in the cell now
DR_NONE = 0  # Cell contains no drone nor other drone-related information
DR_NEAR = 5  # Cell next to a drone; it can potentially move here
DR_VECT = 11 # Drone is heading in this direction

# Vision layer values
V_VISITED = 1       # A drone visited this cell
V_UNVISITED = 0     # No drone has visited this cell yet
V_UNREACHABLE = -1  # This cell is unreachable

# Grid init parameters
BITMAP_OBSTACLES = 'bitmap_obstacles_128.bmp'
BITMAP_DRONES = 'bitmap_drones_128.bmp'
CELL_TYPE = np.int8  # Daha küçük integer tipi, negatif değerleri destekler
PAD = 2  # Padding eklenmesi

# Grid initialization as a multi-layered Numpy array
def init_grid():
    # Layer 0: obstacles:
    img_obstacles = Image.open(BITMAP_OBSTACLES).convert('L')  # Monochrome
    img_w, img_h = img_obstacles.size
    if img_w < GRID_W or img_h < GRID_H:
        print(f'Obstacle layer bitmap ({BITMAP_OBSTACLES}) not big enough: '
              f'{img_w}x{img_h} instead of required minimum of {GRID_W}x{GRID_H}')
        exit(0)
    np_obstacles = (np.array(img_obstacles) < 128).astype(CELL_TYPE)
    np_obstacles = np_obstacles[:GRID_W, :GRID_H]
    obst_mask = np_obstacles == 1
    np_obstacles_padded = np.pad(np_obstacles, pad_width=PAD, mode='constant', constant_values=1)

    # Layer 1: drones
    img_drones = Image.open(BITMAP_DRONES).convert('L')  # Monochrome
    img_w, img_h = img_drones.size
    if img_w < GRID_W or img_h < GRID_H:
        print(f'Drone layer bitmap ({BITMAP_OBSTACLES}) not big enough: '
              f'{img_w}x{img_h} instead of required minimum of {GRID_W}x{GRID_H}')
        exit(0)
    np_drones = (np.array(img_drones) < 128).astype(CELL_TYPE)
    np_drones = np_drones[:GRID_W, :GRID_H]
    np_drones[np_drones == 1] = DR_HERE
    np_drones[obst_mask] = DR_NONE
    np_drones_padded = np.pad(np_drones, pad_width=PAD, mode='constant', constant_values=DR_NONE)

    # Layer 2: visits
    np_visits = np.zeros_like(np_obstacles).astype(CELL_TYPE)
    np_visits[obst_mask] = V_UNREACHABLE
    np_visits_padded = np.pad(np_visits, pad_width=PAD, mode='constant', constant_values=V_UNREACHABLE)

   
#########################################################


    # Layer 3: additional
    np_grads_padded = np.zeros_like(np_visits_padded).astype(CELL_TYPE)

    # Layer 4: collision avoidance 1
    np_coll1_padded = np.zeros_like(np_visits_padded).astype(CELL_TYPE)
    for x in range(np_coll1_padded.shape[0]):
        for y in range(np_coll1_padded.shape[1]):
            if x%2 == y%2:
                np_coll1_padded[x, y] = 1

    # Layer 5: collision avoidance 2
    np_coll2_padded = np.zeros_like(np_visits_padded).astype(CELL_TYPE)
    for x in range(np_coll2_padded.shape[0]):
        for y in range(np_coll2_padded.shape[1]):
            if x%2:
                np_coll2_padded[x, y] = 1

    # Full multi-layered grid
    np_grid = np.stack((np_obstacles_padded, 
                        np_drones_padded, 
                        np_visits_padded, 
                        np_grads_padded,
                        np_coll1_padded,
                        np_coll2_padded), axis=-1)
    print(f'Grid initialized as Numpy array of shape {np_grid.shape}, {(np_drones == DR_HERE).sum()} drones')
    return np_grid


# Upoating of the grid: drones make decisions, move and change their surroundings
def update_grid(grid, iter_count=0):
    new_grid = grid.copy()

    for x in range(GRID_W):
        for y in range(GRID_H):
            # What kind of content does this cell have?
            cell_drone_val = int(grid[PAD+x, PAD+y, L_DRON])
            # If cell has no drone-related information or 
            # if it's just vector point - nothing happens
            if cell_drone_val == DR_NONE:
                continue
            
            # If cell is marked as nearest to some drone or a vector point - check if the drone is still nearby
            if cell_drone_val in (DR_NEAR, DR_VECT):
                drone_near = False
                for dx in range(-1, 1+1):
                    if drone_near:
                        break
                    for dy in range(-1, 1+1):
                        if int(grid[PAD+x+dx, PAD+y+dy, L_VIS]) == V_UNREACHABLE:
                            continue
                        if int(grid[PAD+x+dx, PAD+y+dy, L_DRON]) == DR_HERE:
                            drone_near = True
                            break
                # drone_near_2 = np.any(grid[PAD+x-1:PAD+x+2, PAD+y-1:PAD+y+2, L_DRON] == DR_HERE)
                # if drone_near != drone_near_2:
                #     print('ALARM! ERROR IN NUMPY!')
                if not drone_near:
                    # No drone nearby, this is a DR_NONE
                    new_grid[PAD+x, PAD+y, L_DRON] = DR_NONE
                continue

            # If cell doesn't contain a drone at this point - something is
            # wrong
            if cell_drone_val != DR_HERE:
                print(f'Problem with cell drone value: {cell_drone_val} at {x}, {y}')
                return None
            
            # Otherwise - this cell contains a drone, and it must make a decision
            # "Big zone" - 5x5 square with the drone in the center,
            # i.e. drone cell and its surrounding 2 cells away
            big_zone_index = (
                    slice(PAD+x-BIG_ZONE_R, PAD+x+BIG_ZONE_R+1), 
                    slice(PAD+y-BIG_ZONE_R, PAD+y+BIG_ZONE_R+1), 
                    slice(None)
                )
            np_big_zone = grid[big_zone_index].copy()

            # Separate layers
            np_obst, np_drones, np_visit, np_grad, np_coll1, np_coll2 = np.dsplit(np_big_zone, grid.shape[-1])
            # Now make (5, 5, 1) arrays into (5, 5)
            np_obst = np.squeeze(np_obst)
            np_drones = np.squeeze(np_drones)
            np_visit = np.squeeze(np_visit)
            np_grad = np.squeeze(np_grad)
            np_coll1 = np.squeeze(np_coll1)
            np_coll2 = np.squeeze(np_coll2)

            # Center of the 5x5 "big zone" - the drone itself
            drone_x, drone_y = BIG_ZONE_R, BIG_ZONE_R

            # What was the previous cell? It's the one with DR_NONE
            prev_coords = []
            prev_x = 1
            prev_y = 1
            for dx in range(-1, 1+1):
                for dy in range(-1, 1+1):
                    if np_drones[drone_x+dx, drone_y+dy] == DR_NONE:
                        prev_coords = (dx, dy)
                        prev_x, prev_y = prev_coords
                        break
                if len(prev_coords):
                    break

            # Drone visits cells within its visit radius, including its own cell 
            for vx in range(-DRONE_VIS_R, DRONE_VIS_R+1):
                for vy in range(-DRONE_VIS_R, DRONE_VIS_R+1):
                    np_visit[drone_x+vx, drone_y+vy] = V_VISITED

            # All possible moves as changes to current drone coordinates:
            # -1 <= dx <= 1,  -1 <= dy <= 1,
            # as pairs (dx, dy), excluding staying in the same cell (0, 0)
            # because it's handled differently
            possible_moves = []
            for dx in range(-1, 1+1):
                for dy in range(-1, 1+1):
                    if dx == dy == 0:
                        # Standing still is not an option
                        continue
                    if drone_x+dx >= GRID_W or 0 >= drone_x+dx:
                        continue
                    if drone_y+dy >= GRID_H or 0 >= drone_y+dy:
                        continue
                    possible_moves.append((dx, dy))

            #    I. Decision making
            # 1. Exclude definitely impossible moves - obstacles and cells occupied by other drones
            for dx in range(-1, 1+1):
                for dy in range(-1, 1+1):
                    obstacle = np_obst[drone_x+dx, drone_y+dy]
                    drone = np_drones[drone_x+dx, drone_y+dy]
                    if (obstacle or (drone == DR_HERE)) and (dx, dy) in possible_moves:
                        possible_moves.remove((dx, dy))
            if not len(possible_moves):
                continue

            # 2. Exclude moves that could get the drone too close 
            # to another drone
            remove_this_drone = False
            thrust_away = False

            for dx in range(-BIG_ZONE_R, BIG_ZONE_R+1):
                if remove_this_drone or thrust_away:
                    break
                for dy in range(-BIG_ZONE_R, BIG_ZONE_R+1):
                    if remove_this_drone or thrust_away:
                        break
                    collision_val = int(np_drones[drone_x+dx, drone_y+dy])
                    # Let's check our nearest surrounding
                    if -1 <= dx <= 1 and -1 <= dy <= 1:
                        # Drone ignores itself
                        if dx == dy == 0:
                            continue
                        if collision_val == DR_HERE:
                            # Collision imminent, thrust away
                            possible_moves = [(-dx, -dy)]
                            thrust_away = True
                            # Experimental: since, in theory, such close encounter with another
                            # drone is supposed to be impossible, and this might be a "drone spawning"
                            # error, we should remove some of these "neighboring" drones
                            if abs(dx) != abs(dy):
                                # Use diagonal collision matrix
                                removal_flag = np_coll1[drone_x, drone_y]
                            else:
                                # Use straight collision matrix
                                removal_flag = np_coll2[drone_x, drone_y]
                            if removal_flag:
                                possible_moves = [(0, 0)]
                                remove_this_drone = True
                            continue
                    elif collision_val == DR_NEAR:
                        # Exclude all potential moves that could lead 
                        # to collisions
                        if dx == BIG_ZONE_R:
                            # Another drone to the east - remove all east-containing moves
                            possible_moves = [pm for pm  in possible_moves if pm[0] != 1]
                        if dx == -BIG_ZONE_R:
                            # Another drone to the west - remove all west-containing moves
                            possible_moves = [pm for pm  in possible_moves if pm[0] != -1]
                        if dy == BIG_ZONE_R:
                            # Another drone to the south - remove all south-containing moves
                            possible_moves = [pm for pm  in possible_moves if pm[1] != 1]
                        if dy == -BIG_ZONE_R:
                            # Another drone to the north - remove all north-containing moves
                            possible_moves = [pm for pm  in possible_moves if pm[1] != -1]
                    elif collision_val in (DR_VECT, DR_HERE) and dx != -prev_x and dy != -prev_y:
                        # Collision imminent, thrust away
                        if abs(dx) > 1:
                            dx //= 2
                        if abs(dy) > 1:
                            dy //= 2
                        possible_moves = [(-dx, -dy)]
                        thrust_away = True

            if not len(possible_moves):
                possible_moves = [(0, 0)]

            # If drone is already stuck inside an obstacle - remove it
            if np_obst[drone_x, drone_y]:
                print(f'{iter_count}. Drone at {x},{y} is inside an obstacle - removing it')
                remove_this_drone = True
            
            # If drone is thrusting into an obstacle - just stop
            if len(possible_moves) == 1:
                pm_x, pm_y = possible_moves[0]
                obstacle = np_obst[drone_x+pm_x, drone_y+pm_y]
                if obstacle:
                    possible_moves = [(0, 0)]

            # 3. Unvisited cells take more priority via 
            # how many unvisited cells will be available from new position
            pm_with_priorities = []
            if len(possible_moves) != 1:
                for pm_x, pm_y in possible_moves:
                    # Drone would move to these coordinates within "big zone"
                    temp_x = drone_x + pm_x
                    temp_y = drone_y + pm_y
                    unvisited_cells_count = (np_visit[temp_x-1:temp_x+2, temp_y-1:temp_y+2] == V_UNVISITED).sum()
                    if unvisited_cells_count != 0:
                        pm_with_priorities.append(((pm_x, pm_y), unvisited_cells_count))

            #    II. Drone movement
            if len(pm_with_priorities):
                # Pick any move that leads to the best result
                max_cells_count = max([pm[1] for pm in pm_with_priorities])
                best_moves = [pm for pm in pm_with_priorities if pm[1] == max_cells_count]
                mov_dx, mov_dy = random.choice(best_moves)[0]
            else:
                # If no move leads to any gain - continue moving in the same direction
                if len(prev_coords):  
                    mov_dx = -prev_x
                    mov_dy = -prev_y
                    if (mov_dx, mov_dy) not in possible_moves:
                        mov_dx, mov_dy = random.choice(possible_moves)
                else:
                    mov_dx, mov_dy = random.choice(possible_moves)

            # Final cleanup of possible moves
            old_possible_moves = possible_moves[:]
            possible_moves = []
            for pm in old_possible_moves:
                dx, dy = pm
                if dx == dy == 0:
                    continue
                obst = int(np_obst[drone_x+dx, drone_y+dy])
                drone = int(np_drones[drone_x+dx, drone_y+dy])
                if obst or drone in (DR_HERE, DR_VECT):
                    continue
                possible_moves.append(pm)
            if not len(possible_moves):
                possible_moves = [(0, 0)]

            if not remove_this_drone:  
                # Drone moves and changes drone-related values around it
                # 1. Drone's nearest cells (DR_NEAR) aren't these anymore
                for dx in range(-1, 1+1):
                    for dy in range(-1, 1+1):
                        if dx == dy == 0:
                            continue
                        if possible_moves == [(0, 0)]:
                            ovewriteables = (DR_NEAR, DR_HERE, DR_VECT)
                        else:
                            ovewriteables = (DR_NEAR, DR_VECT)
                        if int(np_drones[drone_x+dx, drone_y+dy]) in ovewriteables:
                            np_drones[drone_x+dx, drone_y+dy] = DR_NONE

                # Bit of a hack: if stuck, remove all foreign vector points
                if possible_moves == [(0, 0)]:
                    for dx in range(-BIG_ZONE_R, BIG_ZONE_R+1):
                        for dy in range(-BIG_ZONE_R, BIG_ZONE_R+1):
                            if -1 <= dx <= 1 and -1 <= dy <= 1:
                                continue
                            if int(np_drones[drone_x+dx, drone_y+dy]) in (DR_VECT, DR_NEAR):
                                np_drones[drone_x+dx, drone_y+dy] = DR_NONE

                # 2. Drone is now in a different cell
                new_x = drone_x + mov_dx
                new_y = drone_y + mov_dy
                if (new_x, new_y) != (drone_x, drone_y) and np_drones[new_x, new_y] == DR_HERE:
                    # Already a drone there
                    new_x = drone_x
                    new_y = drone_y
                np_drones[new_x, new_y] = DR_HERE

                # 3. Drone's nearest cells are now these ones
                for dx in range(-1, 1+1):
                    for dy in range(-1, 1+1):
                        if int(np_drones[new_x+dx, new_y+dy]) in (DR_NONE, DR_VECT):
                            np_drones[new_x+dx, new_y+dy] = DR_NEAR

                # 4. Drone isn't in its old cell anymore
                if (new_x, new_y) != (drone_x, drone_y):
                    np_drones[drone_x, drone_y] = DR_NONE

                # 5. Drone's vector point is new
                vect_x = new_x+mov_dx
                vect_y = new_y+mov_dy
                if int(np_drones[vect_x, vect_y]) in (DR_NONE, DR_NEAR):
                    # DO NOT rewrite another drone with vector
                    np_drones[new_x+mov_dx, new_y+mov_dy] = DR_VECT
            else:
                # Remove this drone and its additional information
                # 1. Drone's nearest cells (DR_NEAR) aren't these anymore
                for dx in range(-1, 1+1):
                    for dy in range(-1, 1+1):
                        if dx == dy == 0:
                            continue
                        if int(np_drones[drone_x+dx, drone_y+dy]) in (DR_NEAR, DR_VECT):
                            np_drones[drone_x+dx, drone_y+dy] = DR_NONE

                # 2. Drone itself is removed
                np_drones[drone_x, drone_y] = DR_NONE

            #    III. Update grid
            np_big_zone = np.stack((np_obst,
                                    np_drones,
                                    np_visit,
                                    np_grad,
                                    np_coll1,
                                    np_coll2), axis=-1)
            new_grid[big_zone_index] = np_big_zone
    return new_grid