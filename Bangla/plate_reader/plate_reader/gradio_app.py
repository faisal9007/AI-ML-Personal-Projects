"""Quick Gradio demo. Run with:  python gradio_app.py"""
import cv2
import gradio as gr

from plate_reader import PlateReader

reader = PlateReader()


def app_fn(image, plate_conf, char_conf):
    if image is None:
        return None, "Upload a vehicle photo."
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    res = reader.read(bgr, plate_conf, char_conf)
    ann = cv2.cvtColor(reader.annotate(bgr, res), cv2.COLOR_BGR2RGB)
    if not res:
        return ann, "No plate found. Try lowering the plate confidence."
    lines = []
    for i, r in enumerate(res, 1):
        lines.append(f"Plate {i}: {r['text_bn'] or r['text']}")
        lines.append(f"   English: {r['text']}")
        lines.append(f"   plate confidence {r['plate_conf']:.2f} | "
                     f"avg character confidence {r['char_conf']:.2f}")
    return ann, "\n".join(lines)


demo = gr.Interface(
    fn=app_fn,
    inputs=[gr.Image(type="numpy", label="Vehicle photo"),
            gr.Slider(0.1, 0.9, value=0.4, step=0.05, label="Plate detection confidence"),
            gr.Slider(0.05, 0.9, value=reader.char_conf_default, step=0.05,
                      label="Character confidence")],
    outputs=[gr.Image(label="Result"), gr.Textbox(label="Plate text", lines=6)],
    title="Bangla License Plate Reader",
    description="Upload a vehicle photo. The app finds the plate and reads the characters.")

if __name__ == "__main__":
    demo.launch()
