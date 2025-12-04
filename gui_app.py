import wx
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Configuration
MODEL_PATH = 'flower_model.keras'
CLASS_NAMES_PATH = 'class_names.txt'
IMG_HEIGHT = 180
IMG_WIDTH = 180

class ImageDropTarget(wx.FileDropTarget):
    def __init__(self, obj):
        wx.FileDropTarget.__init__(self)
        self.obj = obj

    def OnDropFiles(self, x, y, filenames):
        if filenames:
            self.obj.process_image(filenames[0])
        return True

class FlowerApp(wx.Frame):
    def __init__(self):
        super().__init__(parent=None, title='Flower Classifier', size=(700, 800))
        
        self.SetBackgroundColour(wx.Colour(240, 240, 245))
        
        self.panel = wx.Panel(self)
        self.panel.SetBackgroundColour(wx.Colour(255, 255, 255))
        
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        
        # UI Elements
        self.title_text = wx.StaticText(self.panel, label="Flower Classifier")
        font_title = wx.Font(24, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        self.title_text.SetFont(font_title)
        self.title_text.SetForegroundColour(wx.Colour(50, 50, 100))
        
        self.select_btn = wx.Button(self.panel, label="Select Image")
        font_btn = wx.Font(12, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        self.select_btn.SetFont(font_btn)
        self.select_btn.SetBackgroundColour(wx.Colour(70, 130, 180))
        self.select_btn.SetForegroundColour(wx.Colour(255, 255, 255))
        self.select_btn.Bind(wx.EVT_BUTTON, self.on_select_image)
        
        self.image_ctrl = wx.StaticBitmap(self.panel, size=(450, 450))
        self.image_ctrl.SetBackgroundColour(wx.Colour(230, 230, 230))
        
        self.result_text = wx.StaticText(self.panel, label="Drag & Drop an image here or click Select Image")
        font_result = wx.Font(14, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        self.result_text.SetFont(font_result)
        
        # Layout
        self.sizer.Add(self.title_text, 0, wx.ALL | wx.CENTER, 20)
        self.sizer.Add(self.image_ctrl, 0, wx.ALL | wx.CENTER, 20)
        self.sizer.Add(self.result_text, 0, wx.ALL | wx.CENTER, 15)
        self.sizer.Add(self.select_btn, 0, wx.ALL | wx.CENTER, 20)
        
        self.panel.SetSizer(self.sizer)
        
        # Drag and Drop
        dt = ImageDropTarget(self)
        self.panel.SetDropTarget(dt)
        
        # Load Model
        self.model = None
        self.class_names = []
        wx.CallAfter(self.load_model)
        
        self.Center()
        self.Show()

    def load_model(self):
        self.result_text.SetLabel("Loading model... Please wait.")
        self.panel.Layout()
        wx.Yield()
        
        if os.path.exists(MODEL_PATH):
            try:
                self.model = keras.models.load_model(MODEL_PATH)
                print("Model loaded successfully.")
            except Exception as e:
                self.result_text.SetLabel(f"Error loading model: {e}")
                return
        else:
            self.result_text.SetLabel(f"Error: {MODEL_PATH} not found.")
            return

        if os.path.exists(CLASS_NAMES_PATH):
            with open(CLASS_NAMES_PATH, 'r') as f:
                self.class_names = [line.strip() for line in f.readlines()]
            self.result_text.SetLabel("Drag & Drop an image here or click Select Image")
        else:
            self.result_text.SetLabel(f"Error: {CLASS_NAMES_PATH} not found.")

    def on_select_image(self, event):
        if self.model is None:
            wx.MessageBox("Model not loaded!", "Error", wx.OK | wx.ICON_ERROR)
            return

        with wx.FileDialog(self, "Open Image", wildcard="Image files (*.jpg;*.jpeg;*.png)|*.jpg;*.jpeg;*.png",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:
            
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return
            
            pathname = fileDialog.GetPath()
            self.process_image(pathname)

    def process_image(self, filepath):
        # Display Image
        img = wx.Image(filepath, wx.BITMAP_TYPE_ANY)
        # Scale image to fit
        W = img.GetWidth()
        H = img.GetHeight()
        max_size = 450
        if W > max_size or H > max_size:
            aspect = W / H
            if W > H:
                new_w = max_size
                new_h = int(max_size / aspect)
            else:
                new_h = max_size
                new_w = int(max_size * aspect)
            img = img.Scale(new_w, new_h, wx.IMAGE_QUALITY_HIGH)
            
        self.image_ctrl.SetBitmap(wx.Bitmap(img))
        self.result_text.SetLabel("Analyzing...")
        self.panel.Layout()
        wx.Yield()

        try:
            # Preprocess
            img_keras = keras.utils.load_img(
                filepath, target_size=(IMG_HEIGHT, IMG_WIDTH)
            )
            img_array = keras.utils.img_to_array(img_keras)
            img_array = tf.expand_dims(img_array, 0)

            # Predict
            predictions = self.model.predict(img_array)
            score = tf.nn.softmax(predictions[0])

            best_match = self.class_names[np.argmax(score)]
            confidence = 100 * np.max(score)

            self.result_text.SetLabel(f"Result: {best_match} ({confidence:.2f}%)")
            
        except Exception as e:
            self.result_text.SetLabel(f"Error analyzing image: {e}")
            print(e)
        
        self.panel.Layout()

if __name__ == '__main__':
    app = wx.App()
    frame = FlowerApp()
    app.MainLoop()
