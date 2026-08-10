#!/usr/bin/env python3
"""Render and save the three VLA observation streams from the stack scene."""

from pathlib import Path
import mujoco
from PIL import Image, ImageDraw

from stack_scene import build_model, reset_to_stand, CAMERA_NAMES

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results" / "camera_observations"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model = build_model()
    data = mujoco.MjData(model)
    reset_to_stand(model, data)
    # Let free cubes settle on the table while the robot holds its stand pose.
    for _ in range(250):
        mujoco.mj_step(model, data)

    renderer = mujoco.Renderer(model, height=480, width=640)
    frames = []
    try:
        for camera_name in CAMERA_NAMES:
            renderer.update_scene(data, camera=camera_name)
            frame = renderer.render().copy()
            image = Image.fromarray(frame)
            path = OUTPUT / f"{camera_name}.png"
            image.save(path)
            frames.append((camera_name, image))
            print(path)
    finally:
        renderer.close()

    sheet = Image.new("RGB", (640, 3 * 390), (7, 11, 20))
    draw = ImageDraw.Draw(sheet)
    for index, (name, image) in enumerate(frames):
        resized = image.resize((480, 360))
        y = index * 390
        sheet.paste(resized, (160, y + 30))
        draw.text((18, y + 42), name.replace("cam_", "").replace("_", " ").upper(), fill=(34, 211, 238))
        draw.text((18, y + 68), "640 × 480 RGB", fill=(148, 163, 184))
    contact_sheet = OUTPUT / "camera_contact_sheet.png"
    sheet.save(contact_sheet)
    print(contact_sheet)


if __name__ == "__main__":
    main()
