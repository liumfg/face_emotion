import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np
import tkinter as tk
from tkinter import filedialog, font as tkFont, messagebox, ttk
import subprocess
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from ResEmoteNet import ResEmoteNet
import sys

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# -------------------------- 模型初始化 --------------------------
device = torch.device('cpu')
emotions = ['happy', 'surprise', 'sad', 'anger', 'disguise', 'fear', 'neutral']

# 加载情绪识别模型
model = ResEmoteNet().to(device)
try:
    model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
    print("模型加载成功")
except Exception as e:
    messagebox.showerror("模型错误", f"加载best_model.pth失败：{str(e)}")
model.eval()

# 图像预处理转换
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 人脸检测器
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


# -------------------------- UI界面类 --------------------------
class EmotionRecognitionUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("人脸情绪识别系统")
        self.geometry("1400x800")
        self.configure(bg="#f5f5f5")

        # 人脸采集存储路径
        self.face_collection_path = "data/data_faces_from_camera/"

        # 初始化变量
        self.current_image = None
        self.photo_image = None
        self.detection_history = []  # 存储识别历史

        # 字体设置
        self.font_title = tkFont.Font(family='微软雅黑', size=18, weight='bold')
        self.font_subtitle = tkFont.Font(family='微软雅黑', size=14, weight='bold')
        self.font_info = tkFont.Font(family='微软雅黑', size=12)

        # 创建主框架
        self.main_frame = tk.Frame(self, bg="#f5f5f5")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # -------------------- 顶部标题区域 --------------------
        title_frame = tk.Frame(self.main_frame, bg="#f5f5f5")
        title_frame.pack(fill=tk.X, pady=(0, 10))
        tk.Label(title_frame, text="人脸情绪识别系统", font=self.font_title, bg="#f5f5f5").pack(pady=10)

        # -------------------- 操作按钮区域 --------------------
        self.btn_frame = tk.Frame(self.main_frame, bg="#f5f5f5")
        self.btn_frame.pack(fill=tk.X, pady=10)

        tk.Button(self.btn_frame, text="上传图片", command=self.upload_image,
                  font=self.font_info, width=15, bg="#4CAF50", fg="white").grid(row=0, column=0, padx=5)
        tk.Button(self.btn_frame, text="启动人脸采集", command=self.start_face_collection,
                  font=self.font_info, width=15, bg="#2196F3", fg="white").grid(row=0, column=1, padx=5)
        tk.Button(self.btn_frame, text="开始识别", command=self.recognize_emotion,
                  font=self.font_info, width=15, bg="#F44336", fg="white").grid(row=0, column=2, padx=5)

        # -------------------- 主内容区域（分割为左右两部分） --------------------
        content_frame = tk.Frame(self.main_frame, bg="#f5f5f5")
        content_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # 左侧：图像显示区域
        left_frame = tk.Frame(content_frame, bg="#f5f5f5")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # 图像标题
        tk.Label(left_frame, text="图像预览", font=self.font_subtitle, bg="#f5f5f5").pack(anchor=tk.W)

        # 图像显示框
        self.image_frame = tk.Frame(left_frame, bg="#e0e0e0", bd=2, relief=tk.SUNKEN)
        self.image_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.image_label = tk.Label(self.image_frame, bg="#e0e0e0")
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # 右侧：结果显示区域
        right_frame = tk.Frame(content_frame, bg="#f5f5f5")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        # 结果标题
        tk.Label(right_frame, text="识别结果", font=self.font_subtitle, bg="#f5f5f5").pack(anchor=tk.W)

        # 结果文本框
        self.result_text = tk.Text(right_frame, font=self.font_info, width=60, height=10, wrap=tk.WORD)
        self.result_text.pack(fill=tk.X, pady=5)

        # 创建图表框架
        self.chart_frame = tk.Frame(right_frame, bg="#e0e0e0", bd=2, relief=tk.SUNKEN)
        self.chart_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # 创建Matplotlib图表
        self.fig, self.ax = plt.subplots(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # -------------------- 历史记录区域 --------------------
        history_frame = tk.Frame(self.main_frame, bg="#f5f5f5")
        history_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        tk.Label(history_frame, text="识别历史", font=self.font_subtitle, bg="#f5f5f5").pack(anchor=tk.W)

        # 创建历史记录表格
        columns = ("序号", "时间", "主要情绪", "置信度")
        self.history_tree = ttk.Treeview(history_frame, columns=columns, show="headings", height=3)

        for col in columns:
            self.history_tree.heading(col, text=col)
            self.history_tree.column(col, width=100, anchor=tk.CENTER)

        self.history_tree.pack(fill=tk.X, pady=5)

        # 添加滚动条
        scrollbar = ttk.Scrollbar(history_frame, orient=tk.HORIZONTAL, command=self.history_tree.xview)
        self.history_tree.configure(xscroll=scrollbar.set)
        scrollbar.pack(fill=tk.X)

    # -------------------- 功能函数 --------------------
    def start_face_collection(self):
        os.makedirs(self.face_collection_path, exist_ok=True)
        try:
            # 使用当前 Python 解释器的路径
            subprocess.run([sys.executable, "get_faces_from_camera_tkinter_two.py"], check=True, text=True, capture_output=True)
            messagebox.showinfo("提示", f"人脸采集完成！图片存储在：\n{self.face_collection_path}")
        except subprocess.CalledProcessError as e:
            # 捕获详细的错误信息
            error_message = f"采集界面启动失败：{e.stderr}"
            messagebox.showerror("错误", error_message)
        except Exception as e:
            messagebox.showerror("错误", f"采集界面启动失败：{str(e)}")

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("图片文件", "*.jpg;*.jpeg;*.png")])
        if file_path:
            self.current_image = cv2.imread(file_path)
            if self.current_image is None:
                messagebox.showerror("错误", "无法读取图片，请检查格式！")
                return
            self.show_image(self.current_image)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "已加载图片，请点击'开始识别'")
            print(f"上传图片成功，尺寸：{self.current_image.shape}")

    def recognize_emotion(self):
        if self.current_image is None:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "错误：请先上传图片！")
            return

        print("开始情绪识别...")
        img = self.current_image.copy()

        # 优化格式转换逻辑
        try:
            if len(img.shape) == 3:
                if img.shape[2] == 3:
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    print("格式转换：RGB → BGR")
                elif img.shape[2] == 4:
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                    print("格式转换：RGBA → BGR")
                else:
                    img_bgr = img
                    print("格式提示：单通道图像，未转换")
            else:
                img_bgr = img
                print("格式提示：灰度图，未转换")
        except Exception as e:
            messagebox.showerror("格式错误", f"图像转换失败：{str(e)}")
            return

        # 检测人脸
        faces = self.detect_bounding_box(img_bgr)
        if len(faces) == 0:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "未检测到人脸，请调整角度或光线")
            self.clear_chart()
            print("识别结果：未检测到人脸")
            return

        # 显示结果
        self.show_image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        # 更新历史记录
        if faces:
            max_emotion, scores = self.get_face_emotion(img_bgr, faces[0][0], faces[0][1], faces[0][2], faces[0][3])
            confidence = scores[np.argmax(scores)]
            self.update_history(max_emotion, confidence)

        print(f"识别完成，检测到 {len(faces)} 个人脸")

    def detect_bounding_box(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
        print(f"人脸检测结果：{len(faces)} 个人脸")

        # 清空结果文本
        self.result_text.delete(1.0, tk.END)

        if len(faces) > 0:
            self.result_text.insert(tk.END, f"检测到 {len(faces)} 个人脸\n\n")

        for i, (x, y, w, h) in enumerate(faces):
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            try:
                max_emotion, scores = self.get_face_emotion(image, x, y, w, h)
                self.print_max_emotion(image, x, y, max_emotion)
                self.print_all_emotions(image, x, y, w, h, scores)

                # 更新结果文本
                self.result_text.insert(tk.END, f"人脸 {i + 1}: {max_emotion}\n")
                for j, emotion in enumerate(emotions):
                    self.result_text.insert(tk.END, f"  {emotion}: {scores[j]:.2f}\n")
                self.result_text.insert(tk.END, "\n")

                # 绘制概率柱状图（只绘制第一个人脸）
                if i == 0:
                    self.update_chart(scores)

                print(f"人脸 {i + 1} 情绪：{max_emotion}，概率：{scores[np.argmax(scores)]:.2f}")
            except Exception as e:
                print(f"人脸 {i + 1} 情绪识别异常：{str(e)}")
                self.result_text.insert(tk.END, f"人脸 {i + 1} 识别失败：{str(e)}\n\n")
        return faces

    def get_face_emotion(self, image, x, y, w, h):
        crop_img = image[y:y + h, x:x + w]
        if crop_img.size == 0:
            raise ValueError("裁剪的人脸区域为空")

        pil_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
        img_tensor = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy().flatten()

        max_index = np.argmax(probabilities)
        max_emotion = emotions[max_index]
        return max_emotion, probabilities

    def show_image(self, img, resize=(800, 450)):
        if img is None:
            return
        try:
            pil_img = Image.fromarray(img)
            pil_img = pil_img.resize(resize, Image.Resampling.LANCZOS)
            self.photo_image = ImageTk.PhotoImage(pil_img)
            self.image_label.config(image=self.photo_image)
        except Exception as e:
            messagebox.showerror("显示错误", f"图像显示失败：{str(e)}")

    def print_max_emotion(self, image, x, y, emotion):
        org = (x, y - 15)
        cv2.putText(image, emotion, org, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)

    def print_all_emotions(self, image, x, y, w, h, scores):
        text = "\n".join([f"{emotions[i]}: {scores[i]:.2f}" for i in range(len(emotions))])
        y_pos = y - 20
        for line in text.split('\n'):
            y_pos += 30
            cv2.putText(image, line, (x + w + 10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    def update_chart(self, scores):
        """更新情绪概率柱状图"""
        self.ax.clear()
        bars = self.ax.bar(emotions, scores, color='skyblue')

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            self.ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                         f'{height:.2f}', ha='center', va='bottom')

        self.ax.set_ylim(0, 1.1)  # 设置y轴范围
        self.ax.set_title('情绪概率分布')
        self.ax.set_xlabel('情绪类型')
        self.ax.set_ylabel('概率')
        self.fig.tight_layout()
        self.canvas.draw()

    def clear_chart(self):
        """清空图表"""
        self.ax.clear()
        self.ax.set_title('情绪概率分布')
        self.ax.set_xlabel('情绪类型')
        self.ax.set_ylabel('概率')
        self.canvas.draw()

    def update_history(self, emotion, confidence):
        """更新识别历史记录"""
        import datetime
        current_time = datetime.datetime.now().strftime("%H:%M:%S")

        # 限制历史记录数量
        if len(self.detection_history) >= 5:
            self.detection_history.pop(0)

        self.detection_history.append((current_time, emotion, confidence))

        # 清空并重新填充表格
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)

        for i, (time, emotion, conf) in enumerate(self.detection_history, 1):
            self.history_tree.insert("", "end", values=(i, time, emotion, f"{conf:.2f}"))

    def on_closing(self):
        self.destroy()


if __name__ == "__main__":
    app = EmotionRecognitionUI()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()