import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import gradio as gr
from PIL import Image
import urllib.request

opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

def download_quickdraw_subset(category, num_samples=3000):
    url = f"https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{category}.npy"
    bytes_to_fetch = num_samples * 784 + 1024
    req = urllib.request.Request(url)
    req.add_header('Range', f'bytes=0-{bytes_to_fetch}')
    req.add_header('User-Agent', 'Mozilla/5.0')

    try:
        print(f"   [+] Đang tải nhanh dữ liệu vẽ tay mẫu của: '{category}'...")
        with urllib.request.urlopen(req) as response:
            content = response.read()

        magic = content[:6]
        if magic != b'\x93NUMPY':
            raise ValueError("Không khớp chữ ký file npy của NumPy.")

        major = content[6]
        if major == 1:
            header_len = int.from_bytes(content[8:10], byteorder='little')
            header_start = 10
        elif major == 2:
            header_len = int.from_bytes(content[8:12], byteorder='little')
            header_start = 12
        else:
            raise ValueError("Phiên bản định dạng NumPy không được hỗ trợ.")

        raw_start = header_start + header_len
        raw_data = content[raw_start:]

        arr = np.frombuffer(raw_data, dtype=np.uint8)
        num_imgs = len(arr) // 784
        arr = arr[:num_imgs * 784].reshape(num_imgs, 784)

        return arr[:num_samples]
    except Exception as e:
        print(f"   [!] Gặp lỗi Range Request cho {category}: {e}. Chuyển sang tải truyền thống...")

        os.makedirs("temp_data", exist_ok=True)
        fallback_path = f"temp_data/{category}.npy"
        urllib.request.urlretrieve(url, fallback_path)
        arr = np.load(fallback_path)[:num_samples]
        if os.path.exists(fallback_path):
            os.remove(fallback_path)
        return arr

def initialize_and_train():
    model = FashionANN().to(device)

    if os.path.exists(MODEL_PATH):
        print("[HỆ THỐNG] Phát hiện trọng số ANN vẽ tay đã lưu! Đang tiến hành nạp vào mô hình...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        return model

    print("[HỆ THỐNG] Không tìm thấy mô hình có sẵn. Bắt đầu huấn luyện...")
    # Quá trình huấn luyện (Sẽ bị bỏ qua nếu bạn đã upload file .pth lên GitHub)
    all_images = []
    all_labels = []
    samples_per_class = 50000

    for idx, category in enumerate(CATEGORIES):
        class_data = download_quickdraw_subset(category, num_samples=samples_per_class)
        all_images.append(class_data)
        all_labels.append(np.full(len(class_data), idx))

    X = np.concatenate(all_images, axis=0).astype(np.float32) / 255.0
    y = np.concatenate(all_labels, axis=0)

    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]

    split_idx = int(len(X) * 0.85)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)

    epochs = 20
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        train_acc = 100 * correct_train / total_train
        val_acc = 100 * correct_val / total_val
        print(f"   => Epoch {epoch+1:02d}/{epochs:02d} | Loss: {running_loss/len(train_loader):.4f} | Accuracy: {train_acc:.2f}% | Val Accuracy: {val_acc:.2f}%")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[HỆ THỐNG] Đã lưu mô hình tại: '{MODEL_PATH}'")
    model.eval()
    return model

model = initialize_and_train()

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
body {
    background-color: #F8FAFC !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}
.gradio-container {
    max-width: 480px !important;
    margin: 0 auto !important;
    padding: 16px !important;
    border-radius: 20px !important;
    background-color: #FFFFFF !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.04) !important;
    border: 1px solid #E2E8F0 !important;
}
.app-header {
    text-align: center;
    margin-bottom: 12px;
}
.app-title {
    font-size: 22px !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    margin-bottom: 6px !important;
}
.app-subtitle {
    font-size: 13px !important;
    color: #64748B !important;
}
.action-button {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%) !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    border-radius: 12px !important;
    padding: 12px !important;
    border: none !important;
    cursor: pointer;
    box-shadow: 0 4px 10px rgba(79, 70, 229, 0.25) !important;
    transition: all 0.2s ease !important;
}
.action-button:active {
    transform: scale(0.97) !important;
}
.back-button {
    background-color: #F1F5F9 !important;
    color: #475569 !important;
    border: 1px solid #E2E8F0 !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    border-radius: 8px !important;
    padding: 6px 14px !important;
}
.result-card {
    background: #F8FAFC;
    padding: 14px;
    border-radius: 12px;
    margin-bottom: 12px;
    border: 1px solid #E2E8F0;
}
.result-meta {
    display: flex;
    align-items: center;
    margin-bottom: 6px;
}
.rank-badge {
    color: #FFFFFF;
    font-size: 10px;
    font-weight: 800;
    padding: 2px 6px;
    border-radius: 999px;
    margin-right: 8px;
}
.class-name {
    font-size: 14px;
    font-weight: 700;
    color: #0F172A;
}
.fuzzy-info {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    margin-bottom: 4px;
}
.progress-track {
    background-color: #E2E8F0;
    height: 8px;
    border-radius: 999px;
    width: 100%;
    overflow: hidden;
}
.progress-bar {
    height: 100%;
    border-radius: 999px;
    transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
}
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
    # root_path="" giúp Gradio tự tìm đúng đường dẫn API trên Proxy của Render
    demo.queue().launch(
        server_name="0.0.0.0", 
        server_port=port, 
        root_path=os.environ.get("RENDER_EXTERNAL_URL", ""),
        share=False
    )