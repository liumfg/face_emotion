import dlib
import numpy as np
import cv2
import os
import logging
import tkinter as tk
from tkinter import font as tkFont
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
from ResEmoteNet import ResEmoteNet
# Dlib
detector = dlib.get_frontal_face_detector()

class Face_Capturer:
    def __init__(self):
        self.save_cnt = 0
        self.current_faces_cnt = 0  # 当前检测到的人脸数量

        self.emotion_labels = ['happy', 'surprise', 'sad', 'anger', 'disgust', 'fear', 'neutral']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ResEmoteNet().to(self.device)
        self.model.load_state_dict(torch.load("best_model.pth", map_location=self.device))
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.win = tk.Tk()     #主窗口
        self.win.title("Face Emotion Capturer")
        self.win.geometry("1300x600")

        self.left_camera = tk.Frame(self.win)
        self.label = tk.Label(self.win)
        self.label.pack(side=tk.LEFT)
        self.left_camera.pack()

        self.right_text = tk.Frame(self.win)
        self.input_emotion = tk.Entry(self.right_text)
        self.input_emotion_char = ""
        self.label_warning = tk.Label(self.right_text)
        self.label_face_cnt = tk.Label(self.right_text, text="Faces in current frame: ")
        self.log = tk.Label(self.right_text)
        self.recognition_result = tk.Label(self.right_text, text="Emotion Recognition: ", fg="blue")
        #字体
        self.font = cv2.FONT_ITALIC
        self.font_title = tkFont.Font(family='Helvetica', size=20, weight='bold')
        self.font_step_title = tkFont.Font(family='Helvetica', size=15, weight='bold')
        self.font_warning = tkFont.Font(family='Helvetica', size=15, weight='bold')

        self.path_photos_from_camera = "data/data_faces_from_camera/"
        self.current_frame = np.ndarray
        self.face_ROI_image = np.ndarray
        self.face_ROI_left_x = 0
        self.face_ROI_left_y = 0
        self.face_ROI_width = 0
        self.face_ROI_height = 0
        self.w = 0
        self.h = 0

        self.out_of_range_flag = False
        self.face_folder_created_flag = False
        self.valid_emotion_input = False
        self.cap = cv2.VideoCapture(0)

    def get_info(self):
        tk.Label(self.right_text,
                 text="Face Emotion Capturer",
                 font=self.font_title).grid(row=0, column=0, columnspan=3, sticky=tk.E+tk.W, padx=2, pady=20)

        tk.Label(self.right_text,
                 text="Faces in current frame: ").grid(row=1, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        self.label_face_cnt.grid(row=1, column=2, columnspan=3, sticky=tk.W, padx=5, pady=2)

        self.label_warning.grid(row=2, column=0, columnspan=3, sticky=tk.W, padx=5, pady=2)

        tk.Label(self.right_text,
                 font=self.font_step_title,
                 text="Step 1: Clear face photos").grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=5, pady=20)

        tk.Button(self.right_text,
                  text='Clear',
                  command=self.clear_data).grid(row=4, column=0, columnspan=3, sticky=tk.W, padx=5, pady=2)

        tk.Label(self.right_text,
                 font=self.font_step_title,
                 text="Step 2: Input emotion category").grid(row=5, column=0, columnspan=2, sticky=tk.W, padx=5, pady=20)

        tk.Label(self.right_text, text="Category: ").grid(row=6, column=0, sticky=tk.W, padx=5, pady=0)
        self.input_emotion.grid(row=6, column=1, sticky=tk.W, padx=0, pady=2)

        tk.Button(self.right_text,
                  text='Set',
                  command=self.get_input_emotion).grid(row=6, column=2, padx=5)

        tk.Label(self.right_text,
                 font=self.font_step_title,
                 text="Step 3: Save face images & recognize emotion").grid(row=7, column=0, columnspan=2, sticky=tk.W, padx=5, pady=20)

        tk.Button(self.right_text,
                  text='Save & Recognize',
                  command=self.save_current_face).grid(row=8, column=0, columnspan=3, sticky=tk.W)

        self.log.grid(row=9, column=0, columnspan=3, sticky=tk.W, padx=5, pady=10)
        self.recognition_result.grid(row=10, column=0, columnspan=3, sticky=tk.W, padx=5, pady=10)
        self.right_text.pack()

    def clear_data(self):
        # 删除之前存的人脸数据
        for filename in os.listdir(self.path_photos_from_camera):
            file_path = os.path.join(self.path_photos_from_camera, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
        self.log["text"] = "Face images removed!"
        self.save_cnt = 0
        self.recognition_result["text"] = "Emotion Recognition: "

    #创建目录
    def create(self):
        if os.path.isdir(self.path_photos_from_camera):
            pass
        else:
            os.mkdir(self.path_photos_from_camera)

    def recognize_emotion(self, face_img):
        try:
            #转换为PIL图像
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(face_img)

            img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                _, predicted = torch.max(outputs.data, 1)
                emotion_idx = predicted.item()
            emotion_probs = {}
            for i, label in enumerate(self.emotion_labels):
                emotion_probs[label] = round(probabilities[i].item(), 2)

            #排序（从高到低）
            sorted_probs = sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True)
            result_text = f"Probability: {sorted_probs[0][0]} {sorted_probs[0][1]}\n"
            remaining_probs = []
            for i in range(1, len(sorted_probs)):
                remaining_probs.append(f"{sorted_probs[i][0]} {sorted_probs[i][1]}")

            for i in range(0, len(remaining_probs), 2):
                if i + 1 < len(remaining_probs):
                    result_text += f"{remaining_probs[i]:<20}{remaining_probs[i + 1]:<20}\n"
                else:
                    result_text += f"{remaining_probs[i]}\n"
            return sorted_probs[0][0], result_text
        except Exception as e:
            print(f"Error in emotion recognition: {e}")
            return "recognition failed"

    def save_current_face(self):
        if not self.valid_emotion_input:
            self.log["text"] = "Please input emotion first!"
            self.label_warning["text"] = "Please input emotion first!"
            self.label_warning["fg"] = 'red'
            return
        if not self.out_of_range_flag:
            self.save_cnt += 1
            # 获取图像尺寸
            img_height, img_width = self.current_frame.shape[:2]

            # 计算裁剪区域边界
            x1 = max(0, self.face_ROI_left_x - self.w)  # 左侧边界
            y1 = max(0, self.face_ROI_left_y - self.h)  # 上侧边界
            x2 = min(img_width, self.face_ROI_left_x + self.face_ROI_width + self.w)  # 右侧边界
            y2 = min(img_height, self.face_ROI_left_y + self.face_ROI_height + self.h)  # 下侧边界

            # 提取安全区域
            self.face_ROI_image = self.current_frame[y1:y2, x1:x2].copy()

            # 保存图像
            save_path = os.path.join(self.path_photos_from_camera,f"{self.input_emotion_char}_{self.save_cnt}.jpg")
            cv2.imwrite(save_path, cv2.cvtColor(self.face_ROI_image, cv2.COLOR_RGB2BGR))
            recognized_emotion,prob_text = self.recognize_emotion(self.face_ROI_image)
            self.recognition_result["text"] = f"Recognition Emotion:{recognized_emotion}  Input Emotion: {self.input_emotion_char}\n{prob_text}"
            self.log["text"] = f"\"{save_path}\" saved!"
            logging.info("%-40s %s", "写入本地 / Save into：", save_path)
        else:
            self.log["text"] = "Please do not out of range!"

    def get_frame(self):
        try:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    self.label_warning["text"] = "Failed to read frame!"
                    self.label_warning["fg"]= 'red'
                    return False,None
                return ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                self.label_warning["text"] = "Camera not opened!"
                self.label_warning["fg"] = 'red'
                return False, None
        except Exception as e:
            print(f"Error reading frame: {e}")
            self.label_warning["text"] = "Error reading frame!"
            self.label_warning['fg'] = 'red'
            return False, None

    def get_input_emotion(self):
        VALID_EMOTIONS = ['happy', 'surprise', 'sad', 'anger', 'disgust', 'fear', 'neutral']
        emotion = self.input_emotion.get().strip()

        if not emotion:
            self.log["text"] = "Emotion cannot be empty!"
            self.label_warning["text"] = "Please input emotion first!"
            self.label_warning["fg"] = 'red'
            self.valid_emotion_input = False
            return False

        # 检查是否为有效表情
        if emotion.lower() not in VALID_EMOTIONS:
            self.log["text"] = f"Invalid emotion! Must be one of: {', '.join(VALID_EMOTIONS)}"
            self.label_warning["text"] = "Invalid emotion!"
            self.label_warning["fg"] = 'red'
            self.valid_emotion_input = False
            return False

        self.valid_emotion_input = True
        self.input_emotion_char = emotion.lower()
        self.log["text"] = f"Emotion '{emotion}' set successfully"
        return True

    # 获取人脸
    def process(self):
        ret, self.current_frame = self.get_frame()
        faces = detector(self.current_frame, 0)

        if ret:
            self.label_face_cnt["text"] = str(len(faces))
            img_Image = Image.fromarray(self.current_frame)
            img_PhotoImage = ImageTk.PhotoImage(image=img_Image)
            self.label.img_tk = img_PhotoImage
            self.label.configure(image=img_PhotoImage)

            self.out_of_range_flag = False
            if self.valid_emotion_input:
                if len(faces) != 0:
                    for i, d in enumerate(faces):
                        face_left_x = d.left()
                        face_left_y = d.top()
                        face_width = d.right() - face_left_x
                        face_height = d.bottom() - face_left_y

                        self.w = int(face_width * 0.3)
                        self.h = int(face_height * 0.4)
                        left = face_left_x - self.w
                        top = face_left_y - self.h
                        right = face_left_x + face_width + self.w
                        bottom = face_left_y + face_height + self.h

                        # 判断是否超出范围
                        if (left < 0 or top < 0 or right > 640 or bottom > 480):
                            self.out_of_range_flag = True
                            color_rectangle = (0, 0, 255)  # 红色
                            status_text = f"Face {i + 1} (OUT OF RANGE)"
                        else:
                            color_rectangle = (0, 255, 0)  # 绿色
                            status_text = f"Face {i + 1}"

                        #绘制人脸框
                        self.current_frame = cv2.rectangle(self.current_frame,
                                                       (left, top), (right, bottom),
                                                       color_rectangle, 2)

                        #绘制标签
                        label_bg = self.current_frame.copy()
                        cv2.rectangle(label_bg, (left, top - 30), (right, top), color_rectangle, -1)
                        cv2.addWeighted(label_bg, 0.7, self.current_frame, 0.3, 0, self.current_frame)
                        self.current_frame = cv2.putText(self.current_frame, status_text,
                                                     (left + 5, top - 10),
                                                     self.font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                        #当前人脸ROI信息
                        self.face_ROI_left_x = face_left_x
                        self.face_ROI_left_y = face_left_y
                        self.face_ROI_width = face_width
                        self.face_ROI_height = face_height
                        self.log["text"] = "Face quality is good."

                    if self.out_of_range_flag:
                        self.label_warning["text"] = "One or more faces out of range!"
                        self.label_warning["fg"] = 'red'
                    else:
                        self.label_warning["text"] = f"Detected {len(faces)} faces, ready to save!"
                        self.label_warning["fg"] = 'green'
                        self.log["text"] = f"Face detection: {len(faces)} face(s) detected"
                else:
                    # 没有检测到人脸或未设置情绪时的提示
                    if len(faces) == 0:
                        self.label_warning["text"] = "No face detected!"
                        self.label_warning["fg"] = 'red'
                        self.log["text"] = "No face detected in current frame"

            self.current_faces_cnt = len(faces)
            img_Image = Image.fromarray(self.current_frame)
            img_PhotoImage = ImageTk.PhotoImage(image=img_Image)
            self.label.img_tk = img_PhotoImage
            self.label.configure(image=img_PhotoImage)
        self.win.after(20, self.process)

    def run(self):
        self.create()
        self.get_info()
        self.process()
        self.win.mainloop()

def main():
    logging.basicConfig(level=logging.INFO)
    Face_Register_one = Face_Capturer()
    Face_Register_one.run()

if __name__ == '__main__':
    main()