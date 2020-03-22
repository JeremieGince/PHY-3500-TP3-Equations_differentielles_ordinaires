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


def gif_to_mp4(filename: str, directory: str = ""):
    import moviepy.editor as mp
    if filename.endswith(".gif"):
        clip = mp.VideoFileClip(f"{directory}/{filename}")
        clip.write_videofile(f"{directory}/{filename.replace('.gif', '.mp4')}")
    else:
        raise ValueError("filename must endswith .gif")


if __name__ == '__main__':
    directory = "Simulations"

    for filename in os.listdir(directory):
        if filename.endswith(".gif"):
            # gif_to_png(filename, directory)
            gif_to_mp4(filename, directory)
        else:
            continue
