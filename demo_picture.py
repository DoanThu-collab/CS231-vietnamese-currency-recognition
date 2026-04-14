import cv2
import numpy as np
import pickle
import json
from PIL import Image, ImageDraw, ImageFont
import textwrap

# ================= CONFIG =================
MODEL_PATH = r"D:\UIT_NamHai\NHAP_MON_CV\CS231-vietnamese-currency-recognition\traditional\orb_model.pkl"
MAPPING_PATH = r"D:\UIT_NamHai\NHAP_MON_CV\CS231-vietnamese-currency-recognition\app\mapping.json"
TEST_IMAGE = r"D:\UIT_NamHai\NHAP_MON_CV\CS231-vietnamese-currency-recognition\data\test\005000\1633073869592.jpg"

# ================= INIT =================
ORB = cv2.ORB_create(nfeatures=500)
BF  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# ================= LOAD =================
with open(MODEL_PATH, "rb") as f:
    templates = pickle.load(f)

with open(MAPPING_PATH, "r", encoding="utf-8") as f:
    mapping = json.load(f)

# ================= PREDICT =================
def predict_one(image_path, templates):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return "unknown"

    img = cv2.resize(img, (224, 224))
    _, des = ORB.detectAndCompute(img, None)

    if des is None:
        return "unknown"

    scores = {}
    for class_name, tmpl_matrix in templates.items():
        try:
            matches = BF.match(des, tmpl_matrix)
            good = [m for m in matches if m.distance < 50]
            scores[class_name] = len(good)
        except:
            scores[class_name] = 0

    return max(scores, key=scores.get)

# ================= HELPER CHỮ TIẾNG VIỆT =================
def put_text_vietnamese(img_pil, text, position, font_size, color_rgb):
    draw = ImageDraw.Draw(img_pil)
    try:
        # Sử dụng font Arial mặc định của Windows (hỗ trợ tốt tiếng Việt)
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    
    draw.text(position, text, font=font, fill=color_rgb)

# ================= DRAW UI =================
def draw_ui(image_path, pred_class):
    img = cv2.imread(image_path)
    if img is None:
        print("Không đọc được ảnh")
        return

    # Resize ảnh cho đẹp (W=500, H=350)
    img = cv2.resize(img, (500, 350))

    # Tạo panel bên phải rộng hơn để chứa chữ (W=450, H=350)
    panel = np.ones((350, 450, 3), dtype=np.uint8) * 40  

    # Ghép ảnh + panel bằng OpenCV trước
    combined_cv2 = np.hstack((img, panel))
    
    # Border đẹp hơn
    cv2.rectangle(combined_cv2, (0, 0), 
                  (combined_cv2.shape[1]-1, combined_cv2.shape[0]-1), 
                  (0, 255, 0), 2)

    # Chuyển OpenCV (BGR) sang PIL (RGB) để vẽ chữ
    combined_cv2_rgb = cv2.cvtColor(combined_cv2, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(combined_cv2_rgb)

    # Toạ độ X bắt đầu cho text trên panel (Ảnh rộng 500 -> panel bắt đầu từ x=520)
    start_x = 520

    if pred_class in mapping:
        info = mapping[pred_class]

        # Title (Lưu ý: Hệ màu của PIL là RGB)
        put_text_vietnamese(pil_img, "RESULT", (680, 20), 30, (0, 255, 0)) # Green

        # Info
        put_text_vietnamese(pil_img, f"Money: {info['denomination']} VND", (start_x, 70), 22, (255, 255, 255))
        put_text_vietnamese(pil_img, f"Type: {info.get('type', '')}", (start_x, 110), 20, (0, 255, 255)) # Cyan
        
        put_text_vietnamese(pil_img, "Place:", (start_x, 150), 20, (255, 255, 0)) # Yellow
        put_text_vietnamese(pil_img, info.get('place', ''), (start_x, 180), 18, (255, 255, 255))

        put_text_vietnamese(pil_img, "Description:", (start_x, 220), 20, (255, 255, 0)) # Yellow
        
        # Xử lý tự động xuống dòng cho description (wrap text)
        description_text = info.get('description', '')
        # Chia text thành các dòng, mỗi dòng max 45 ký tự
        wrapped_text = textwrap.wrap(description_text, width=45) 
        
        y_offset = 250
        for line in wrapped_text:
            put_text_vietnamese(pil_img, line, (start_x, y_offset), 16, (255, 255, 255))
            y_offset += 25 # Khoảng cách giữa các dòng
            
    else:
        put_text_vietnamese(pil_img, "UNKNOWN", (650, 150), 35, (255, 0, 0)) # Red

    # Chuyển ngược lại từ PIL (RGB) về OpenCV (BGR) để show ảnh
    final_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    cv2.imshow("Vietnam Currency Recognition", final_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ================= MAIN =================
if __name__ == "__main__":
    pred = predict_one(TEST_IMAGE, templates)
    print(f"Predicted class: {pred}")
    # ... (giữ nguyên phần print text terminal của bạn)
    draw_ui(TEST_IMAGE, pred)