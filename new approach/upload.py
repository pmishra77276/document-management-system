from flask import Flask, request, jsonify
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from pdf2image import convert_from_path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
import fitz
from pdf2image import convert_from_bytes
from PIL import Image
import os
import cv2
import torch
import requests
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import numpy as np
from bs4 import BeautifulSoup
text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=0
    )
app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading TrOCR models...")
trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten").to(device)

# Initialize DocTR
print("Loading DocTR models...")
detector = ocr_predictor(pretrained=True)

# print("Loading embedding model...")
# model = HuggingFaceEmbeddings(
# model_name="intfloat/e5-large-v2",
# )
def load_embeddings_from_solr():
    CORE = "emb_core"
    BASE_URL = f"http://10.111.65.43:8983/solr/{CORE}/select"
    
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
print("Loading embeddings from Solr...")
emb_dict = load_embeddings_from_solr()
print(f"Loaded {len(emb_dict)} embeddings from Solr")
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
def classification(text,emb_dict,name):
    docs1=text_splitter.split_text(text)
    for i in range(len(docs1)):
        docs1[i]+=" "+name
    docs1=model.embed_documents(docs1)
    doc1=np.zeros(len(docs1[0]))
    for k in docs1:
        doc1+=k
    file=""
    csin1=0
    for i in emb_dict:
        name=i.split('/')[-1].split('.')[1]
        # print(name)
        csin=cosine_similarity([emb_dict[i]],[doc1])
        # print(name)
        # print(csin)
        if csin>csin1:
            csin1=csin
            file=name
    return file,docs1,doc1
def list_files_from_url(base_url):
    """Get list of PDF file URLs from a directory-style HTTP page."""
    resp = requests.get(base_url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    files = []
    for link in soup.find_all("a"):
        href = link.get("href")
        if href and href.lower().endswith(".pdf"):
            files.append(base_url.rstrip("/") + "/" + href.split('/')[-1])
        # print()
    return files

def send_result_to_solr(result):
    """Send document result to Solr."""
    SOLR_URL = "http://10.111.65.43:8983/solr/mycore/update/json/docs?commit=true"
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
CORE = "mycore"
BASE_URL = f"http://10.111.65.43:8983/solr/{CORE}/select"
resp0 = requests.get(BASE_URL, params={"q": "*:*", "wt": "json", "rows": 0})
resp0.raise_for_status()
total = resp0.json()["response"]["numFound"]

resp1 = requests.get(
    BASE_URL,
    params={"q": "*:*", "wt": "json", "rows": total}
)
resp1.raise_for_status()
docs = resp1.json()["response"]["docs"]

dict1 = {}
files=[]
for d in docs:
    files.append(d['id'])
@app.route('/upload', methods=['POST'])
def main():
    default="/mnt/filereader/"
    folder_path=request.json.get('folder_path',default)
    paths=os.listdir(folder_path)
    for path in paths:
        if not path.lower().endswith(".pdf"):
            continue 
        name=path.split("\\")[-1].split('.')[0]
        if name not in files:
            os.makedirs("/home/pa/ml/tf_gpu/OCR/D:/BPL/BLS/Result/", exist_ok=True)
            output_json=f"/home/pa/ml/tf_gpu/OCR/D:/BPL/BLS/Result/{name}.json"
            ppath=os.path.join(folder_path,path)
            # print(classify_and_extract_pdf(ppath))
            text1=""
            results=[]
            
            text1+=name
            text1+=" "
            out=classify_and_extract_pdf(ppath)
            # print(out)
            if(out==0):
                print(f"{name} Document seems to be a pdf of image")
                text1=process_pdf(ppath,trocr_model,detector)
            else:
                print(f"{name} Document is pdf of text")
                text1=out
            class1,docs1,doc1=classification(text1,emb_dict,name)
            print("Document best classified as",class1)
            # print(text1)
            # break
            result = {
                "id":name,
                "class": class1,
                "text": text1, 
                "section_embedding": docs1, 
                "total_embedding": doc1.tolist() if hasattr(doc1, "tolist") else doc1 
            }
            send_result_to_solr(result)
            print()
            print()
        else:
            print("data already present in solr")
            continue
    return jsonify({"message": "File processing completed."})

if __name__ == '__main__':
    # Make sure output directories exist
    os.makedirs("extracted_data", exist_ok=True)
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
# main("/home/pa/Downloads/")
    
    