import os
import pandas as pd
from PIL import Image, ImageDraw

def plot_dot_coordinates(csv_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for csv_file in os.listdir(csv_dir):
        if csv_file.endswith('.csv'):
            csv_path = os.path.join(csv_dir, csv_file)
            data = pd.read_csv(csv_path)

            img = Image.new('RGB', (800, 600), 'black')
            draw = ImageDraw.Draw(img)

            for _, row in data.iterrows():
                x, y = int(row['X']), int(row['Y'])
                draw.point((x, y), fill='red')

            output_path = os.path.join(output_dir, f'{os.path.splitext(csv_file)[0]}.png')
            img.save(output_path)

if __name__ == "__main__":
    csv_directory = './icdia/ground_truth'
    output_directory = './icdia/dots'
    plot_dot_coordinates(csv_directory, output_directory)
