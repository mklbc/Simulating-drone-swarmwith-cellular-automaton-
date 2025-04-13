import pygame
from dsa_graphics import init_pygame, draw_grid, observer, FPS
from dsa_automaton import init_grid, update_grid, GRID_W, GRID_H


def calculate_progress(grid):
    """Calculates the percentage of the operational area that has been visited."""
    visit_layer = grid[:, :, 2]  # Visit layer (L_VIS)
    visited_cells = (visit_layer == 1).sum()  # Number of visited cells
    total_cells = (visit_layer != -1).sum()  # Total number of cells excluding obstacles
    return (visited_cells / total_cells) * 100


def run_simulation_until_threshold(threshold=75):
    """
    Runs the simulation until the specified progress threshold (75%) is reached.
    Returns the number of iterations to achieve the progress.
    """
    grid = init_grid()
    iter_count = 0

    while calculate_progress(grid) < threshold:
        grid = update_grid(grid, iter_count)
        iter_count += 1

    return iter_count


def run_multiple_simulations(simulation_count=10, threshold=75):
    """
    Runs multiple simulations and calculates statistics.
    Measures the number of iterations in each simulation and prints the results to the terminal.
    """
    results = []

    for _ in range(simulation_count):
        steps = run_simulation_until_threshold(threshold)
        results.append(steps)

    print("Minimum Number of Iterations:", min(results))
    print("Maximum Number of Iterations:", max(results))
    print("Average Number of Iterations:", sum(results) / len(results))


if __name__ == '__main__':
    run_multiple_simulations()