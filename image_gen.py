from PIL import Image, ImageDraw, ImageFont
from gridworld import Gridworld

def draw_grid_world(grid, cell_size=100, line_color=(0, 0, 0), output_path="images/gridworld.png"):
    """
    Draws a grid world with different cell types and saves it as a PNG image.
    
    Parameters:
        grid (list of lists): 2D array representing the grid world with characters ['W', '_', 'k', 'd', 's', 'g', '>', '<', '^', 'v'].
        cell_size (int): Size of each cell in pixels.
        line_color (tuple): RGB color of the grid lines.
        output_path (str): Path to save the PNG file.
    """
    rows, cols = len(grid), len(grid[0])
    width, height = cols * cell_size, rows * cell_size
    
    # Load a font that supports Unicode characters
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", cell_size // 2)
    except IOError:
        font = ImageFont.load_default()
    
    # Define colors
    colors = {
        'W': (100, 100, 100),  # Grey for walls
        '_': (180, 200, 255),  # Light blue for ice
        'k': (180, 200, 255),  # Ice background for key
        'd': (180, 200, 255),  # Ice background for door
        's': (180, 200, 255),  # Ice background for start
        'g': (180, 200, 255),  # Ice background for goal
        '>': (180, 200, 255),  # Ice background for arrows
        '<': (180, 200, 255),
        '^': (180, 200, 255),
        'v': (180, 200, 255)
    }
    
    symbols = {
        's': u'☆', 'k': u'⚷', 'd': u'⌥', 'g': u'★',
        '>': u'→', '<': u'←', '^': u'↑', 'v': u'↓'
    }
    
    # Create a blank image
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    
    # Draw grid cells
    for row in range(rows):
        for col in range(cols):
            x0, y0 = col * cell_size, row * cell_size
            x1, y1 = x0 + cell_size, y0 + cell_size
            draw.rectangle([x0, y0, x1, y1], fill=colors.get(grid[row][col], (255, 255, 255)))
    
    # Draw special elements (keys, doors, start, goal, arrows)
    for row in range(rows):
        for col in range(cols):
            x0, y0 = col * cell_size, row * cell_size
            if grid[row][col] in symbols:
                text = symbols[grid[row][col]]
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                text_x = x0 + (cell_size - text_width) // 2
                text_y = y0 + (cell_size - text_height) // 2
                draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)
    
    # Draw the grid lines
    for row in range(rows + 1):
        y = row * cell_size
        draw.line([(0, y), (width, y)], fill=line_color, width=2)
    
    for col in range(cols + 1):
        x = col * cell_size
        draw.line([(x, 0), (x, height)], fill=line_color, width=2)
    
    # Save the image
    image.save(output_path)
    print(f"Grid world saved to {output_path}")

G = Gridworld()
G.load_gridworld('data/gridworld/gridworld_03.txt')
draw_grid_world(G.gridworld, cell_size=100, output_path="images/gridworld_03.png")
