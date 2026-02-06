import gradio as gr
from ultralytics import YOLO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from PIL import Image
import tempfile

model = YOLO('/path/')
names = model.names

def generate_pdf(image, detections_text, count):
    tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    c = canvas.Canvas(tmp_pdf.name, pagesize=A4)

    c.setFont("Helvetica-Bold", 14)
    c.drawString(2 * cm, 10 * cm, "Detection Report")
    c.setFont("Helvetica", 11)
    c.drawString(2 * cm, 26.5 * cm, f"Detected objects: {count}")
    c.drawString(2 * cm, 25.8 * cm, "Details:")

    y = 25 * cm
    for line in detections_text.split("\n"):
        c.drawString(2.5 * cm, y, f"- {line}")
        y -= 0.6 * cm

    img_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    image.save(img_path)
    c.drawImage(img_path, 2 * cm, 5 * cm, width=17 * cm, preserveAspectRatio=True)

    c.showPage()
    c.save()
    return tmp_pdf.name

def classify_image(img):
    results = model(img)
    boxes = results[0].boxes

    output = []
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        output.append(f"{names[cls_id]} ({conf*100:.2f}%)")

    result_img = results[0].plot()
    text = "\n".join(output)
    summary = f"Detected {len(output)} objects"

    pdf_path = generate_pdf(
        Image.fromarray(result_img),
        text,
        len(output)
    )

    return result_img, text, pdf_path

gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Image(label="Result"),
        gr.Textbox(label="Detection summary"),
        gr.File(label="Download PDF report")
    ],
    title="Jet Classifier (YOLO11)",
    description="Download a satellite image of the aircraft. The model will determine the type.",
    flagging_mode="manual",
    flagging_dir="flagged"
).launch(share=True)
