from flask import Flask, request, jsonify
import os
import numpy as np
import requests
import torch
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pdf2image import convert_from_path
import fitz  # PyMuPDF
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

app = Flask(__name__)

# Global configurations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize models
@app.before_first_request
def initialize_models():
    global trocr_processor, trocr_model, detector, text_splitter, model, emb_dict
    
    # Initialize TrOCR
    print("Loading TrOCR models...")
    trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
    trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten").to(device)
    
    # Initialize DocTR
    print("Loading DocTR models...")
    detector = ocr_predictor(pretrained=True)
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=0
    )
    
    # Initialize embeddings model
    print("Loading embedding model...")
    model = HuggingFaceEmbeddings(
        model_name="intfloat/e5-large-v2",
    )
    
    # Load embeddings from Solr
    print("Loading embeddings from Solr...")
    emb_dict = load_embeddings_from_solr()
    print(f"Loaded {len(emb_dict)} embeddings from Solr")

def load_embeddings_from_solr():
    CORE = "emb_core"
    BASE_URL = f"http://localhost:8983/solr/{CORE}/select"
    
    try:
        resp0 = requests.get(BASE_URL, params={"q": "*:*", "wt": "json", "rows": 0})
        resp0.raise_for_status()
        total = resp0.json()["response"]["numFound"]
        
        resp1 = requests.get(
            BASE_URL,
            params={"q": "*:*", "wt": "json", "rows": total}
        )
        resp1.raise_for_status()
        docs = resp1.json()["response"]["docs"]
        
        emb_dict = {}
        for d in docs:
            path = d["id"]
            vec = d.get("total_embedding", [])
            emb_dict[path] = np.array(vec, dtype=float)
            
        # Remove specific document if it exists
        if "/home/pa2/ml/Document-Classification/CLASSIFICATION FILES/11.Bill of Entry.txt" in emb_dict:
            emb_dict.pop("/home/pa2/ml/Document-Classification/CLASSIFICATION FILES/11.Bill of Entry.txt")
            
        return emb_dict
    except Exception as e:
        print(f"Error loading embeddings from Solr: {str(e)}")
        return {}

def classify_and_extract_pdf(pdf_path, text_threshold=20):
    doc = fitz.open(pdf_path)
    image_pages = 0
    text_pages = 0
    full_text = ""

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        images = page.get_images(full=True)

        if len(text.strip()) < text_threshold and len(images) > 0:
            image_pages += 1
        else:
            text_pages += 1
            full_text += f"\n--- Page {page_num + 1} ---\n{text}"

    doc.close()

    if image_pages > text_pages:
        return 0
    else:
        return full_text.strip()

def preprocess_image(pil_image):
    """Convert a PIL image to grayscale and apply adaptive thresholding."""
    img = np.array(pil_image)  
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
    return Image.fromarray(binary).convert("RGB")

def cluster_lines(boxes, y_threshold=15):
    """Groups bounding boxes into lines and sorts them correctly."""
    boxes = sorted(boxes, key=lambda b: b[1])
    lines = []
    current_line = []
    for box in boxes:
        if not current_line:
            current_line.append(box)
        else:
            prev_box = current_line[-1]
            if abs(box[1] - prev_box[1]) < y_threshold:  
                current_line.append(box)
            else:
                lines.append(sorted(current_line, key=lambda b: b[0]))
                current_line = [box]

    if current_line:
        lines.append(sorted(current_line, key=lambda b: b[0]))
    return lines

def process_pdf(pdf_path):
    """Extract text from a PDF, perform OCR on image-based PDFs."""
    output_folder = "extracted_data"
    os.makedirs(output_folder, exist_ok=True)

    pdf_name = os.path.basename(pdf_path)  
    extracted_text = []

    images = convert_from_path(pdf_path, dpi=300)  

    for page_num, img in enumerate(images, start=1):
        print(f"Processing Page {page_num}...")

        temp_image_path = f"temp_page_{page_num}.png"
        img.save(temp_image_path)

        single_page_doc = DocumentFile.from_images(temp_image_path)
        result = detector(single_page_doc)

        img_width, img_height = img.size  

        bounding_boxes = []
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    (x_min, y_min), (x_max, y_max) = line.geometry
                    left = int(x_min * img_width)
                    top = int(y_min * img_height)
                    right = int(x_max * img_width)
                    bottom = int(y_max * img_height)
                    bounding_boxes.append((left, top, right, bottom))

        sorted_lines = cluster_lines(bounding_boxes)

        cropped_lines, line_positions = [], []
        for line_boxes in sorted_lines:
            left = min(box[0] for box in line_boxes)
            top = min(box[1] for box in line_boxes)
            right = max(box[2] for box in line_boxes)
            bottom = max(box[3] for box in line_boxes)

            crop = img.crop((left, top, right, bottom))
            processed_crop = preprocess_image(crop)
            cropped_lines.append(processed_crop)
            line_positions.append((left, top, right, bottom))

        if cropped_lines:
            batch_pixel_values = trocr_processor(images=cropped_lines, return_tensors="pt").pixel_values.to(device)
            with torch.no_grad():
                generated_ids = trocr_model.generate(batch_pixel_values)
                predictions = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)
        else:
            predictions = []

        extracted_text.append(f"Page {page_num}:\n" + "\n".join(predictions) + "\n\n")
        os.remove(temp_image_path)
    
    txt = ""
    for i in extracted_text:
        txt += i
        txt += '\n'
    return txt

def classification(text, name):
    """Classify document based on embeddings."""
    docs1 = text_splitter.split_text(text)
    for i in range(len(docs1)):
        docs1[i] += " " + name
    
    docs1_embeddings = model.embed_documents(docs1)
    doc1 = np.zeros(len(docs1_embeddings[0]))
    for k in docs1_embeddings:
        doc1 += k
    
    file = ""
    csin1 = 0
    for i in emb_dict:
        name = i.split('/')[-1].split('.')[1]
        csin = cosine_similarity([emb_dict[i]], [doc1])
        if csin > csin1:
            csin1 = csin
            file = name
    
    return file, docs1_embeddings, doc1

def send_result_to_solr(result):
    """Send document result to Solr."""
    SOLR_URL = "http://localhost:8983/solr/mycore/update/json/docs?commit=true"
    HEADERS = {"Content-Type": "application/json"}
    
    for field in ("section_embedding", "total_embedding"):
        val = result.get(field)
        if hasattr(val, "tolist"):
            result[field] = val.tolist()

    try:
        resp = requests.post(SOLR_URL, headers=HEADERS, json=result)
        if resp.status_code == 200:
            print(f"[OK] Indexed id={result.get('id')}")
            return True
        else:
            print(f"[FAIL] id={result.get('id')} → {resp.status_code}\n{resp.text}")
            return False
    except Exception as e:
        print(f"Error sending to Solr: {str(e)}")
        return False

def embedding_conversion(li, embedding_dim=1024):
    """Convert flat embedding to 2D array."""
    flat_embedding = li
    if len(flat_embedding) % embedding_dim != 0:
        raise ValueError("Flat embedding length is not divisible by embedding dimension.")
    
    num_vectors = len(flat_embedding) // embedding_dim
    section_embedding_2d = [
        flat_embedding[i * embedding_dim: (i + 1) * embedding_dim]
        for i in range(num_vectors)
    ]
    return section_embedding_2d

def load_docs_from_solr():
    """Load all documents from Solr."""
    core_name = "mycore"
    solr_url = f"http://localhost:8983/solr/{core_name}/select"
    
    try:
        initial_params = {
            "q": "*:*",
            "wt": "json",
            "rows": 0
        }
        response = requests.get(solr_url, params=initial_params)
        total_docs = response.json()["response"]["numFound"]
        
        params = {
            "q": "*:*",
            "wt": "json",
            "rows": total_docs
        }
        response = requests.get(solr_url, params=params)
        docs = response.json()["response"]["docs"]
        
        dic = {}
        for i in range(len(docs)):
            li = [
                docs[i]['class'][0],
                docs[i]['text'][0],
                embedding_conversion(docs[i]["section_embedding"]),
                docs[i]['total_embedding']
            ]
            dic[i] = li
        
        return dic
    except Exception as e:
        print(f"Error loading docs from Solr: {str(e)}")
        return {}

def context_provider(dic, query):
    """Provide context for a query from indexed documents."""
    import faiss
    
    idx = ''
    query_embedding = model.embed_query(query)
    query_emb = np.array([query_embedding], dtype='float32')
    sim = 0
    
    for i in dic:
        tot_emb = np.array([dic[i][-1]])
        cos_sim = cosine_similarity(query_emb, tot_emb)
        print(f"Document {i} similarity: {cos_sim[0][0]}")
        
        if cos_sim > sim:
            sim = cos_sim
            idx = i
    
    if idx == '':
        return "No relevant documents found."
    
    text, sec_emb = dic[idx][1], np.array(dic[idx][2], dtype='float32')
    faiss.normalize_L2(sec_emb)
    db = faiss.IndexFlatIP(sec_emb.shape[1])
    db.add(sec_emb)
    D, I = db.search(query_emb, k=8)
    
    context = ""
    text_split = text_splitter.split_text(text)
    
    for i in I[0]:
        if i < len(text_split):
            context += text_split[i]
            context += '\n'
    
    return {
        "context": context,
        "document_class": dic[idx][0],
        "similarity_score": float(sim[0][0]) if hasattr(sim, "shape") else float(sim)
    }

@app.route('/process_pdf', methods=['POST'])
def api_process_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "File must be a PDF"}), 400
    
    # Save the uploaded file temporarily
    temp_path = f"temp_{file.filename}"
    file.save(temp_path)
    
    try:
        name = os.path.splitext(file.filename)[0]
        
        # Check if PDF is text-based or image-based
        out = classify_and_extract_pdf(temp_path)
        
        if out == 0:
            print(f"{name} Document seems to be a pdf of image")
            text = process_pdf(temp_path)
        else:
            print(f"{name} Document is pdf of text")
            text = out
        
        # Classify the document
        doc_class, docs_embeddings, doc_embedding = classification(text, name)
        print("Document best classified as", doc_class)
        
        # Store in Solr
        result = {
            "id": name,
            "class": doc_class,
            "text": text, 
            "section_embedding": docs_embeddings,
            "total_embedding": doc_embedding.tolist() if hasattr(doc_embedding, "tolist") else doc_embedding 
        }
        solr_result = send_result_to_solr(result)
        
        return jsonify({
            "id": name,
            "class": doc_class,
            "text": text,
            "solr_indexed": solr_result
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/query', methods=['POST'])
def api_query():
    data = request.json
    
    if not data or 'query' not in data:
        return jsonify({"error": "No query provided"}), 400
    
    query = data['query']
    
    try:
        # Load documents from Solr
        docs = load_docs_from_solr()
        
        if not docs:
            return jsonify({"error": "No documents found in the database"}), 404
        
        # Get context for query
        context_result = context_provider(docs, query)
        
        return jsonify(context_result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Make sure output directories exist
    os.makedirs("extracted_data", exist_ok=True)
    
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
