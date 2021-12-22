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
