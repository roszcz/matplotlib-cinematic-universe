import subprocess
from pathlib import Path


def ffmpeg_movie(movie_dir: Path, output_path: Path) -> None:
    image_pattern = movie_dir / '*png'

    # It's a lengthy command, but there's no need to grok it
    command = f"""
        ffmpeg -y -framerate 30 -f image2 -pattern_type glob \
        -i '{image_pattern}' -c:v libx264 -r 30 -profile:v high -crf 20 \
        -pix_fmt yuv420p {output_path}
    """
    subprocess.call(command, shell=True)


def movie_to_gif(movie_path, gif_path):
    movie_path = str(movie_path)
    gif_path = str(gif_path)

    command = f'ffmpeg -y -i {movie_path} -f gif {gif_path}'
    subprocess.call(command, shell=True)


def image_directory_to_gif(image_dir: Path, output_path: Path) -> None:
    tmp_movie_path = 'tmp/tmp.mp4'
    ffmpeg_movie(image_dir, tmp_movie_path)
    movie_to_gif(tmp_movie_path, output_path)
    print('gif saved to:', output_path)
