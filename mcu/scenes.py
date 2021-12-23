from pathlib import Path
import random
import string
import os


class Scene:
    SCENE_ID = ''

    def __init__(self):
        self.axes = []
        self.scene_id = f'{self.SCENE_ID}-' + ''.join(random.choices(string.ascii_uppercase, k=16))
        self.content_dir = Path('tmp/scenes') / self.scene_id
        self.content_dir.mkdir(parents=True, exist_ok=True)

        self.frame_paths = []

    def save_frame(self, savepath='tmp/tmp.png'):
        self.figure.tight_layout()

        self.figure.savefig(savepath)

        self.clean_figure()

    def clean_figure(self):
        for ax in self.axes:
            ax.clear()

    def cleanup_frames(self):
        for f_path in self.frame_paths:
            os.remove(f_path)
        self.frame_paths = []
