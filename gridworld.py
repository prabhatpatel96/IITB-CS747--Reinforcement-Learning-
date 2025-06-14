import random
import argparse

class Gridworld:
    def __init__(self, size=15):
        self.unicode_elements = {'W': u'■', 's': u'☆', 'k': u'⚷', 'd': u'⌥', 'g': u'★', '_': u' ', '>': u'→', '<': u'←', '^': u'↑', 'v': u'↓'}
        # Empty Gridworld
        gridworld = [['_' for i in range(size)] for j in range(size)]
        for i in range(size):
            gridworld[0][i] = 'W'
            gridworld[size-1][i] = 'W'
            gridworld[i][0] = 'W'
            gridworld[i][size-1] = 'W'
        
        self.size = size
        self.start = None
        self.goal = None
        self.key = None
        self.door = None
        self.gridworld = gridworld
    
    def print_gridworld(self):
        for i in range(self.size):
            for j in range(self.size):
                print(self.unicode_elements[self.gridworld[i][j]], end=' ')
            print()

    def get_accessible_squares(self):
        accessible_squares = []
        for i in range(1, self.size-1):
            for j in range(1, self.size-1):
                if self.gridworld[i][j] != 'W':
                    accessible_squares.append((i, j))
        return accessible_squares

    def pick_up_key(self):
        self.gridworld[self.key[0]][self.key[1]] = '_'
        self.key = None

    def generate_random_3x3_block(self, alpha=0.8):
        block = [['_' for i in range(3)] for j in range(3)]
        edges = [(0, 1), (1, 0), (1, 2), (2, 1)]
        corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
        # randomly permute edges and corners
        edges = random.sample(edges, len(edges))
        corners = random.sample(corners, len(corners))

        for corner in corners:
            if random.random() < alpha:
                block[corner[0]][corner[1]] = 'W'
            
        count = 0
        for edge in edges:
            if random.random() < alpha:
                if count < 2:
                    block[edge[0]][edge[1]] = 'W'
                    count += 1

        return block

    def generate_random_door_column(self):
        door_column = ['W' for i in range(self.size)]
        door_column[random.randint(1, self.size-2)] = 'd'
        return door_column

    def generate_random_gridworld(self, alpha=0.8):
        # Generate door wall column
        door_column = self.generate_random_door_column()
        door_column_index = self.size - 5
        for i in range(self.size):
            self.gridworld[i][door_column_index] = door_column[i]
            if door_column[i] == 'd':
                self.door = (i, door_column_index)

        # calculate number of blocks based on the size of the grid --> 15x15 grid has 6 blocks as you can fit 2 blocks in 9 columns and 3 blocks in 13 rows
        size_x = door_column_index - 1
        size_y = self.size - 2
        blocks_x = (size_x - 1) // 4
        blocks_y = (size_y - 1) // 4
        num_blocks = blocks_x * blocks_y
        key_block = random.randint(0, num_blocks-1)

        for b_y in range(blocks_y):
            for b_x in range(blocks_x):
                block = self.generate_random_3x3_block(alpha)
                if b_x * blocks_y + b_y == key_block:
                    block[1][1] = 'k'
                    self.key = (b_y*4 + 3, b_x*4 + 3)

                top_left = (b_y*4 + 2, b_x*4 + 2)
                for i in range(3):
                    for j in range(3):
                        self.gridworld[top_left[0]+i][top_left[1]+j] = block[i][j]

        # Generate start and goal
        self.start = (random.randint(1, self.size-2), random.randint(1, door_column_index-1))
        self.goal = (random.randint(1, self.size-2), random.randint(door_column_index+1, self.size-2))
        while self.gridworld[self.start[0]][self.start[1]] != '_':
            self.start = (random.randint(1, self.size-2), random.randint(1, door_column_index-1))
        while self.gridworld[self.goal[0]][self.goal[1]] != '_':
            self.goal = (random.randint(1, self.size-2), random.randint(door_column_index+1, self.size-2))

        self.gridworld[self.start[0]][self.start[1]] = 's'
        self.gridworld[self.goal[0]][self.goal[1]] = 'g'

    def save_gridworld(self, filename):
        with open(filename, 'w') as f:
            for i in range(self.size):
                for j in range(self.size):
                    f.write(self.gridworld[i][j] + ' ')
                f.write('\n')

    def load_gridworld(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
            gridworld = []
            row_index = 0
            for line in lines:
                gridworld.append([])
                squares = line.split(' ')
                for square in squares:
                    if square == 's':
                        self.start = (row_index, len(gridworld[row_index]))
                    elif square == 'g':
                        self.goal = (row_index, len(gridworld[row_index]))
                    elif square == 'k':
                        self.key = (row_index, len(gridworld[row_index]))
                    elif square == 'd':
                        self.door = (row_index, len(gridworld[row_index]))

                    if square != '\n':
                        gridworld[row_index].append(square)
                        
                row_index += 1

            self.size = len(gridworld)
            self.gridworld = gridworld


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MDP Solver")
    parser.add_argument("--size", help="Size of the gridworld", dest="size", required=True)
    parser.add_argument("--sparsity", help="Density of walls, 0: no walls, 1: maximum walls", dest="alpha", required=True)
    parser.add_argument("--output", help="Output file", dest="output", required=True)
    args = parser.parse_args()

    size = int(args.size)
    alpha = float(args.alpha)
    
    G = Gridworld(size)
    G.generate_random_gridworld(alpha)
    G.print_gridworld()
    G.save_gridworld(args.output)

