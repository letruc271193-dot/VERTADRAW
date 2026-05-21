import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import gradio as gr
from PIL import Image

# Ép hệ thống chạy bằng CPU để tiết kiệm RAM trên Render
device = torch.device("cpu")
MODEL_PATH = "fashion_ann_quickdraw.pth"

CATEGORIES = [
    "t-shirt", "pants", "sweater", "underwear", "jacket",
    "shoe", "sock", "hat", "shorts","purse","backpack","necklace","eyeglasses","bowtie"
]

VIETNAMESE_NAMES = {
    "t-shirt": "Áo thun (T-shirt)",
    "pants": "Quần dài (Pants)",
    "sweater": "Áo len (Sweater)",
    "underwear": "Đồ lót (Underwear)",
    "jacket": "Áo khoác (Jacket)",
    "shoe": "Giày (Shoe)",
    "sock": "Vớ/Tất (Sock)",
    "hat": "Nón/Mũ (Hat)",
    "shorts": "Quần đùi (Shorts)",
    "purse": "Túi xách (Purse)",
    "eyeglasses": "Kính mắt (Eyeglasses)",
    "backpack": "Balo (Backpack)",
    "necklace": "Dây chuyền/Vòng cổ (Necklace)",
    "bowtie": "Nơ búm (Bowtie)"
}

class FashionANN(nn.Module):
    def __init__(self):
        super(FashionANN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, len(CATEGORIES))
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)

def load_model():
    print("[HỆ THỐNG] Khởi tạo mô hình ANN...")
    model = FashionANN().to(device)

    # Nếu không có file, báo lỗi ngay lập tức để lập trình viên biết
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"🚨 LỖI TRÍ MẠNG: Không tìm thấy file '{MODEL_PATH}'! Chắc chắn bạn chưa upload file này lên GitHub hoặc sai tên file (chú ý chữ hoa/thường).")

    print(f"[HỆ THỐNG] Đã tìm thấy '{MODEL_PATH}', đang nạp trọng số...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("[HỆ THỐNG] Nạp mô hình thành công! Sẵn sàng nhận diện.")
    return model

# Gọi hàm load model
model = load_model()

def preprocess_and_predict(image_data):
    if image_data is None or 'composite' not in image_data:
        raise gr.Error("⚠️ Bảng vẽ trống! Hãy vẽ một món đồ thời trang!")
    try:
        rgba_img = image_data['composite']
        gray_img = cv2.cvtColor(rgba_img[:, :, :3], cv2.COLOR_RGB2GRAY)
        processed_img = cv2.bitwise_not(gray_img)
        pixel_range = int(np.max(processed_img)) - int(np.min(processed_img))
        if pixel_range < 15:
            raise gr.Error("⚠️ Bảng vẽ trống! Hãy vẽ một món đồ thời trang!")
        kernel = np.ones((4, 4), np.uint8)
        processed_img = cv2.dilate(processed_img, kernel, iterations=1)
        resized_img = cv2.resize(processed_img, (28, 28), interpolation=cv2.INTER_AREA)
        img_tensor = torch.tensor(resized_img, dtype=torch.float32) / 255.0
        img_flat = img_tensor.view(-1).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(img_flat)
            probabilities = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        top3_indices = np.argsort(probabilities)[::-1][:3]

        html_output = "<div class='result-container'>"
        colors = ["#4F46E5", "#10B981", "#F59E0B"]

        for rank, idx in enumerate(top3_indices, 1):
            prob = probabilities[idx]
            percent = prob * 100

            if percent >= 80.0:
                fuzzy_lbl = "Rất giống"
            elif percent >= 50.0:
                fuzzy_lbl = "Khá giống"
            elif percent >= 20.0:
                fuzzy_lbl = "Hơi giống"
            else:
                fuzzy_lbl = "Ít tương đồng"

            bar_color = colors[rank - 1]
            category_key = CATEGORIES[idx]

            html_output += f"""
            <div class="result-card" style="border-left: 5px solid {bar_color};">
                <div class="result-meta">
                    <span class="rank-badge" style="background-color: {bar_color};">Hạng {rank}</span>
                    <span class="class-name">{VIETNAMESE_NAMES[category_key]}</span>
                </div>
                <div class="fuzzy-info">
                    <span class="fuzzy-text" style="color: {bar_color}; font-weight: 700;">{fuzzy_lbl}</span>
                    <span class="prob-percentage">{percent:.1f}%</span>
                </div>
                <div class="progress-track">
                    <div class="progress-bar" style="width: {percent}%; background-color: {bar_color};"></div>
                </div>
            </div>
            """
        html_output += "</div>"
        return gr.update(visible=False), gr.update(visible=True), html_output

    except gr.Error as ge:
        raise ge
    except Exception as e:
        raise gr.Error(f"❌ Có lỗi xảy ra trong quá trình nhận diện: {str(e)}")

def navigate_back():
    return gr.update(visible=True), gr.update(visible=False), gr.update(value=blank_canvas)

custom_stylesheet = """
body { background-color: #F8FAFC !important; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important; }
.gradio-container { max-width: 480px !important; margin: 0 auto !important; padding: 16px !important; border-radius: 20px !important; background-color: #FFFFFF !important; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.04) !important; border: 1px solid #E2E8F0 !important; }
.app-header { text-align: center; margin-bottom: 12px; }
.app-title { font-size: 22px !important; font-weight: 800 !important; background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%) !important; -webkit-background-clip: text !important; -webkit-text-fill-color: transparent !important; margin-bottom: 6px !important; }
.app-subtitle { font-size: 13px !important; color: #64748B !important; }
.action-button { background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%) !important; color: #FFFFFF !important; font-weight: 700 !important; font-size: 16px !important; border-radius: 12px !important; padding: 12px !important; border: none !important; cursor: pointer; box-shadow: 0 4px 10px rgba(79, 70, 229, 0.25) !important; transition: all 0.2s ease !important; }
.action-button:active { transform: scale(0.97) !important; }
.back-button { background-color: #F1F5F9 !important; color: #475569 !important; border: 1px solid #E2E8F0 !important; font-weight: 600 !important; font-size: 13px !important; border-radius: 8px !important; padding: 6px 14px !important; }
.result-card { background: #F8FAFC; padding: 14px; border-radius: 12px; margin-bottom: 12px; border: 1px solid #E2E8F0; }
.result-meta { display: flex; align-items: center; margin-bottom: 6px; }
.rank-badge { color: #FFFFFF; font-size: 10px; font-weight: 800; padding: 2px 6px; border-radius: 999px; margin-right: 8px; }
.class-name { font-size: 14px; font-weight: 700; color: #0F172A; }
.fuzzy-info { display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 4px; }
.progress-track { background-color: #E2E8F0; height: 8px; border-radius: 999px; width: 100%; overflow: hidden; }
.progress-bar { height: 100%; border-radius: 999px; transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1); }
"""

blank_canvas = Image.new("RGBA", (450, 450), (255, 255, 255, 255))

with gr.Blocks(css=custom_stylesheet, title="VERTADRAW - Nhận diện nét vẽ") as demo:
    with gr.Column(visible=True) as draw_screen:
        gr.HTML("""
        <div class="app-header">
            <h1 class="app-title">VERTADRAW - NHẬN DIỆN NÉT VẼ TAY</h1>
            <p class="app-subtitle">Hãy vẽ phác thảo một món đồ thời trang!</p>
        </div>
        """)

        drawing_canvas = gr.ImageEditor(
            value=blank_canvas,
            sources=[],
            image_mode="RGBA",
            type="numpy",
            label="Bảng vẽ",
            height=340,
            interactive=True,
            show_download_button=False,
            show_share_button=False,
            brush=gr.Brush(colors=["#000000"], default_color="#000000", default_size=8)
        )

        btn_predict = gr.Button("🔍 Nhận Diện Nét Vẽ", elem_classes="action-button")

    with gr.Column(visible=False) as result_screen:
        with gr.Row():
            btn_back = gr.Button("← Quay lại", elem_classes="back-button", scale=0)
            gr.HTML("<div style='flex-grow: 1;'></div>")

        gr.HTML("""
        <div class="app-header" style="margin-top: 10px;">
            <h1 class="app-title">Kết Quả Phân Tích:</h1>
            <p class="app-subtitle">Nói chung bạn vẽ cũng được</p>
        </div>
        """)

        analysis_result = gr.HTML()

    btn_predict.click(
        fn=preprocess_and_predict,
        inputs=drawing_canvas,
        outputs=[draw_screen, result_screen, analysis_result]
    )

    btn_back.click(
        fn=navigate_back,
        inputs=None,
        outputs=[draw_screen, result_screen, drawing_canvas],
        js="() => { setTimeout(() => { window.dispatchEvent(new Event('resize')); }, 150); }"
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"🚀 [HỆ THỐNG] Đang chuẩn bị mở cổng {port}...")
    # Tắt share để nhẹ máy, Render đã tự cấp link web cho bạn rồi
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)