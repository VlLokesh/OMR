import os
import uuid
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from werkzeug.utils import secure_filename

UPLOAD_DIR = "uploads"
PDF_DIR = "generated_pdfs"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
DEFAULT_QUESTIONS = 150
DEFAULT_OPTIONS = ("A", "B", "C", "D")

@dataclass
class Bubble:
    contour: np.ndarray
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    fill_ratio: float

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.config["PDF_FOLDER"] = PDF_DIR
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)

ocr_engine: Optional[Any] = None

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def get_ocr_engine():
    global ocr_engine
    if ocr_engine is None:
        from paddleocr import PaddleOCR

        ocr_engine = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    return ocr_engine

def order_points(points: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = points.sum(axis=1)
    diff = np.diff(points, axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    return rect


def four_point_transform(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    rect = order_points(points)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)

    max_width = max(int(width_a), int(width_b))
    max_height = max(int(height_a), int(height_b))

    destination = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(rect, destination)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


def preprocess_image(image: np.ndarray) -> Dict[str, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresholded = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]
    edges = cv2.Canny(blurred, 75, 200)
    return {
        "gray": gray,
        "blurred": blurred,
        "thresholded": thresholded,
        "edges": edges,
    }


def detect_sheet(image: np.ndarray, edges: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approximation) == 4:
            return four_point_transform(image, approximation.reshape(4, 2))

    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        if w > image.shape[1] * 0.6 and h > image.shape[0] * 0.6:
            return image[y : y + h, x : x + w]

    raise ValueError("Unable to detect answer sheet contour.")


def _bubble_from_box(thresh: np.ndarray, box: Tuple[int, int, int, int]) -> Bubble:
    x, y, w, h = box
    roi = thresh[y : y + h, x : x + w]
    area = max(w * h, 1)
    fill_ratio = float(cv2.countNonZero(roi)) / float(area)
    return Bubble(
        contour=np.array(
            [[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]], dtype=np.int32
        ),
        bbox=box,
        center=(x + w // 2, y + h // 2),
        fill_ratio=fill_ratio,
    )


def _bubble_from_circle(thresh: np.ndarray, center: Tuple[int, int], radius: int) -> Bubble:
    x, y = center
    mask = np.zeros(thresh.shape, dtype=np.uint8)
    # Score only the inner bubble area so the printed outline does not look filled.
    cv2.circle(mask, (x, y), max(radius, 2), 255, -1)
    roi = cv2.bitwise_and(thresh, thresh, mask=mask)
    masked_area = max(cv2.countNonZero(mask), 1)
    fill_ratio = float(cv2.countNonZero(roi)) / float(masked_area)
    diameter = radius * 2
    return Bubble(
        contour=np.array([[[x, y]]], dtype=np.int32),
        bbox=(x - radius, y - radius, diameter, diameter),
        center=center,
        fill_ratio=fill_ratio,
    )


def _extract_bubbles_predefined(
    thresholded: np.ndarray, bubble_boxes: Sequence[Sequence[int]]
) -> List[Bubble]:
    return [_bubble_from_box(thresholded, tuple(map(int, box))) for box in bubble_boxes]


def _cluster_axis(values: np.ndarray, max_gap: int, min_cluster_size: int) -> List[int]:
    if values.size == 0:
        return []

    sorted_values = np.sort(values.astype(int))
    clusters: List[List[int]] = [[int(sorted_values[0])]]

    for value in sorted_values[1:]:
        if abs(int(value) - clusters[-1][-1]) <= max_gap:
            clusters[-1].append(int(value))
        else:
            clusters.append([int(value)])

    centers = [
        int(round(sum(cluster) / len(cluster)))
        for cluster in clusters
        if len(cluster) >= min_cluster_size
    ]
    return centers


def _extract_bubbles_hough_grid(
    gray_image: np.ndarray, thresholded: np.ndarray, options: Sequence[str]
) -> Optional[List[List[Bubble]]]:
    circles = cv2.HoughCircles(
        gray_image,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=18,
        param1=60,
        param2=14,
        minRadius=7,
        maxRadius=10,
    )
    if circles is None:
        return None

    detected = np.round(circles[0]).astype(int)
    x_centers = _cluster_axis(detected[:, 0], max_gap=8, min_cluster_size=20)
    y_centers = _cluster_axis(detected[:, 1], max_gap=8, min_cluster_size=18)

    if len(x_centers) < len(options) or len(y_centers) < 5:
        return None
    if len(x_centers) % len(options) != 0:
        return None

    radius = max(3, int(round(float(np.median(detected[:, 2])) * 0.7)))
    grouped: List[List[Bubble]] = []
    for block_start in range(0, len(x_centers), len(options)):
        option_x_positions = x_centers[block_start : block_start + len(options)]
        for y in y_centers:
            row = [
                _bubble_from_circle(thresholded, (x, y), radius)
                for x in option_x_positions
            ]
            grouped.append(row)

    return grouped


def _extract_bubbles_contours(thresholded: np.ndarray) -> List[Bubble]:
    contours, _ = cv2.findContours(
        thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    bubbles: List[Bubble] = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h) if h else 0
        area = cv2.contourArea(contour)

        if area < 150 or w < 12 or h < 12:
            continue
        if not 0.75 <= aspect_ratio <= 1.25:
            continue

        roi = thresholded[y : y + h, x : x + w]
        if roi.size == 0:
            continue

        fill_ratio = float(cv2.countNonZero(roi)) / float(w * h)
        bubbles.append(
            Bubble(
                contour=contour,
                bbox=(x, y, w, h),
                center=(x + w // 2, y + h // 2),
                fill_ratio=fill_ratio,
            )
        )

    if not bubbles:
        raise ValueError("No bubble candidates detected.")

    bubbles = sorted(bubbles, key=lambda bubble: (bubble.center[1], bubble.center[0]))
    return bubbles


def extract_bubbles(
    gray_image: np.ndarray,
    thresholded: np.ndarray,
    options: Sequence[str],
    bubble_boxes: Optional[Sequence[Sequence[int]]] = None,
) -> List[List[Bubble]]:
    if bubble_boxes:
        bubbles = _extract_bubbles_predefined(thresholded, bubble_boxes)
    else:
        hough_grouped = _extract_bubbles_hough_grid(gray_image, thresholded, options)
        if hough_grouped:
            return hough_grouped
        bubbles = _extract_bubbles_contours(thresholded)

    row_tolerance = max(12, int(np.median([b.bbox[3] for b in bubbles]) * 0.8))
    rows: List[List[Bubble]] = []

    for bubble in bubbles:
        if not rows:
            rows.append([bubble])
            continue

        last_row_y = int(np.mean([candidate.center[1] for candidate in rows[-1]]))
        if abs(bubble.center[1] - last_row_y) <= row_tolerance:
            rows[-1].append(bubble)
        else:
            rows.append([bubble])

    grouped: List[List[Bubble]] = []
    for row in rows:
        grouped.append(sorted(row, key=lambda bubble: bubble.center[0]))

    return grouped


def evaluate_answers(
    grouped_bubbles: Sequence[Sequence[Bubble]],
    options: Sequence[str],
    mark_threshold: float = 0.20,
    ambiguity_margin: float = 0.06,
    question_count: Optional[int] = None,
) -> Dict[str, str]:
    answers: Dict[str, str] = {}
    total_questions = question_count or len(grouped_bubbles)

    for index in range(total_questions):
        label = f"Q{index + 1}"
        if index >= len(grouped_bubbles):
            answers[label] = "BLANK"
            continue

        row = list(grouped_bubbles[index])[: len(options)]
        if len(row) < len(options):
            answers[label] = "BLANK"
            continue

        scores = [bubble.fill_ratio for bubble in row]
        max_score = max(scores)
        marked_indices = [i for i, score in enumerate(scores) if score >= mark_threshold]

        if not marked_indices:
            answers[label] = "BLANK"
            continue

        strong_indices = [
            i for i, score in enumerate(scores) if max_score - score <= ambiguity_margin
        ]
        if len(strong_indices) > 1 and max_score >= mark_threshold:
            answers[label] = "INVALID"
            continue

        best_index = int(np.argmax(scores))
        answers[label] = options[best_index]

    return answers


def extract_text_fields(
    gray_image: np.ndarray, regions: Optional[Dict[str, Sequence[int]]] = None
) -> Dict[str, str]:
    if not regions:
        return {}

    engine = get_ocr_engine()
    extracted: Dict[str, str] = {}
    for field_name, box in regions.items():
        x, y, w, h = map(int, box)
        roi = gray_image[y : y + h, x : x + w]
        if roi.size == 0:
            extracted[field_name] = ""
            continue
        try:
            rgb_roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
            result = engine.ocr(rgb_roi, cls=True)
            lines: List[str] = []
            for page in result or []:
                for item in page or []:
                    if item and len(item) > 1 and item[1]:
                        lines.append(item[1][0])
            text = " ".join(lines).strip()
        except Exception:
            text = ""
        extracted[field_name] = " ".join(text.split())

    return extracted


def generate_pdf(answers: Dict[str, str], output_path: str, metadata: Optional[Dict[str, str]] = None) -> None:
    pdf = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4

    pdf.setTitle("OMR Result")
    left_margin = 36
    right_margin = 36
    top_margin = 42
    bottom_margin = 40
    gutter = 16
    columns_per_page = 3
    usable_width = width - left_margin - right_margin
    column_width = (usable_width - gutter * (columns_per_page - 1)) / columns_per_page
    line_height = 14

    answer_items = list(answers.items())
    answered_count = sum(1 for _, value in answer_items if value not in ("BLANK", "INVALID"))
    blank_count = sum(1 for _, value in answer_items if value == "BLANK")
    invalid_count = sum(1 for _, value in answer_items if value == "INVALID")

    meta_rows = len(metadata or {})
    header_height = 86 + (meta_rows * 14)
    available_height = height - top_margin - bottom_margin - header_height
    rows_per_column = max(1, int(available_height // line_height))
    items_per_page = rows_per_column * columns_per_page

    def draw_page_header(page_number: int) -> None:
        pdf.setFont("Helvetica-Bold", 18)
        pdf.drawString(left_margin, height - top_margin, "OMR Result")

        pdf.setFont("Helvetica", 10)
        pdf.drawRightString(
            width - right_margin, height - top_margin + 2, f"Page {page_number}"
        )
        pdf.drawString(
            left_margin,
            height - top_margin - 18,
            f"Total Questions: {len(answer_items)}    Answered: {answered_count}    Blank: {blank_count}    Invalid: {invalid_count}",
        )

        current_y = height - top_margin - 36
        if metadata:
            for key, value in metadata.items():
                pdf.drawString(left_margin, current_y, f"{key}: {value or 'N/A'}")
                current_y -= 14

        divider_y = height - top_margin - header_height + 22
        pdf.setStrokeColor(colors.HexColor("#9fb4cc"))
        pdf.setLineWidth(0.5)
        pdf.line(left_margin, divider_y, width - right_margin, divider_y)

        column_header_y = divider_y - 14
        pdf.setFont("Helvetica-Bold", 10)
        for column_index in range(columns_per_page):
            x = left_margin + column_index * (column_width + gutter)
            header_top = column_header_y + 6
            header_bottom = column_header_y - 10
            answer_split_x = x + (column_width * 0.62)

            pdf.setFillColor(colors.HexColor("#dfeaf5"))
            pdf.rect(x, header_bottom, column_width, header_top - header_bottom, fill=1, stroke=0)
            pdf.setStrokeColor(colors.HexColor("#9fb4cc"))
            pdf.rect(x, header_bottom, column_width, header_top - header_bottom, fill=0, stroke=1)
            pdf.line(answer_split_x, header_bottom, answer_split_x, header_top)

            pdf.setFillColor(colors.black)
            pdf.drawString(x + 6, column_header_y - 2, "Question")
            pdf.drawString(answer_split_x + 6, column_header_y - 2, "Answer")

        return column_header_y - 14

    start_index = 0
    page_number = 1
    while start_index < len(answer_items):
        current_y_top = draw_page_header(page_number)
        page_items = answer_items[start_index : start_index + items_per_page]

        pdf.setFont("Helvetica", 10)
        for item_index, (question, answer) in enumerate(page_items):
            column_index = item_index // rows_per_column
            row_index = item_index % rows_per_column
            x = left_margin + column_index * (column_width + gutter)
            y = current_y_top - row_index * line_height
            answer_split_x = x + (column_width * 0.62)
            row_bottom = y - 4

            if row_index % 2 == 0:
                pdf.setFillColor(colors.HexColor("#f5f8fc"))
                pdf.rect(x, row_bottom, column_width, line_height, fill=1, stroke=0)

            pdf.setStrokeColor(colors.HexColor("#c8d6e5"))
            pdf.rect(x, row_bottom, column_width, line_height, fill=0, stroke=1)
            pdf.line(answer_split_x, row_bottom, answer_split_x, row_bottom + line_height)

            pdf.setFillColor(colors.black)
            pdf.drawString(x + 6, y, question)
            pdf.drawString(answer_split_x + 6, y, answer)

        start_index += items_per_page
        if start_index < len(answer_items):
            pdf.showPage()
            page_number += 1

    pdf.save()


def _coerce_json_value(value):
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return value
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return value
    return value


def load_processing_config(payload: dict) -> Tuple[int, Tuple[str, ...], Optional[List[List[int]]], Optional[Dict[str, List[int]]]]:
    if "config" in payload:
        config_value = _coerce_json_value(payload["config"])
        if isinstance(config_value, dict):
            merged = dict(payload)
            merged.update(config_value)
            payload = merged

    question_count = int(payload.get("question_count", DEFAULT_QUESTIONS))
    options_value = _coerce_json_value(payload.get("options", DEFAULT_OPTIONS))
    if isinstance(options_value, str):
        options = tuple(part.strip() for part in options_value.split(",") if part.strip())
    else:
        options = tuple(options_value)
    if not options:
        raise ValueError("At least one option label is required.")

    bubble_boxes = _coerce_json_value(payload.get("bubble_boxes"))
    ocr_regions = _coerce_json_value(payload.get("ocr_regions"))
    return question_count, options, bubble_boxes, ocr_regions


def process_omr(image_path: str, config: dict) -> Dict[str, object]:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Unable to read uploaded image.")

    raw = preprocess_image(image)
    aligned = detect_sheet(image, raw["edges"])
    aligned_preprocessed = preprocess_image(aligned)

    question_count, options, bubble_boxes, ocr_regions = load_processing_config(config)
    grouped_bubbles = extract_bubbles(
        aligned_preprocessed["gray"], aligned_preprocessed["thresholded"], options, bubble_boxes
    )
    answers = evaluate_answers(grouped_bubbles, options, question_count=question_count)
    extracted_fields = extract_text_fields(aligned_preprocessed["gray"], ocr_regions)

    return {
        "answers": answers,
        "fields": extracted_fields,
        "aligned_shape": {
            "width": int(aligned.shape[1]),
            "height": int(aligned.shape[0]),
        },
    }


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part in request."}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "No selected file."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type."}), 400

    filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(image_path)

    try:
        config = request.form.to_dict(flat=True)
        if request.is_json:
            config.update(request.get_json(silent=True) or {})

        result = process_omr(image_path, config)
        pdf_name = f"{uuid.uuid4().hex}_omr_result.pdf"
        pdf_path = os.path.join(app.config["PDF_FOLDER"], pdf_name)
        generate_pdf(result["answers"], pdf_path, result["fields"])

        response = {
            "answers": result["answers"],
            "fields": result["fields"],
            "download_link": f"/download/{pdf_name}",
            "aligned_shape": result["aligned_shape"],
        }
        return jsonify(response), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover
        return jsonify({"error": f"Processing failed: {exc}"}), 500


@app.route("/download/<path:filename>", methods=["GET"])
def download_file(filename: str):
    return send_from_directory(app.config["PDF_FOLDER"], filename, as_attachment=True)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def healthcheck():
    return jsonify({"status": "ok", "message": "OMR service is running."})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
