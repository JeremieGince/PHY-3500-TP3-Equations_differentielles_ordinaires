from PIL import Image
from PIL import GifImagePlugin
import os


def gif_to_png(filename: str, directory: str = ""):
    imageObject = Image.open(f"{directory}/{filename}")
    filename = filename.replace(".gif", "")

    if os.path.exists(f"{directory}/{filename}"):
        for f in os.listdir(f"{directory}/{filename}"):
            os.remove(f"{directory}/{filename}/{f}")
    else:
        os.makedirs(f"{directory}/{filename}")

    for frame in range(0, imageObject.n_frames):
        imageObject.seek(frame)
        imageObject.save(f"{directory}/{filename}/frame_{frame}.png")


if __name__ == '__main__':
    directory = "Simulations"

    for filename in os.listdir(directory):
        if filename.endswith(".gif"):
            gif_to_png(filename, directory)
        else:
            continue
