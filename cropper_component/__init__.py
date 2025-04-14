import streamlit.components.v1 as components
import os

RELEASE = True
if not RELEASE:
    _component_func = components.declare_component(
        "cropper_component", url="http://localhost:5173/"
    )
else:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend/dist")
    _component_func = components.declare_component("cropper_component", path=build_dir)


def cropper_component(image):
    cropped_image = _component_func(image=image)
    return cropped_image
