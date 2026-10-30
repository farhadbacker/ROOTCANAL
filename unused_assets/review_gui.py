#!/usr/bin/env python3

import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import pandas as pd
from pathlib import Path
import csv

class LabelReviewGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Label Verification Tool - Endodontic X-rays")
        self.root.geometry("1000x750")
        
        self.base_dir = Path(__file__).parent.absolute()
        self.csv_path = self.base_dir / 'image_labels_clean.csv'
        
        if not self.csv_path.exists():
            messagebox.showerror("Error", f"Labels file not found: {self.csv_path}")
            self.root.destroy()
            return
        
        self.df = pd.read_csv(self.csv_path)
        self.images_dir = self.base_dir / 'dataset' / 'images'
        
        self.class_names = {
            0: 'No Endodontic Treatment',
            1: 'Incomplete Endodontic Treatment',
            2: 'Complete Endodontic Treatment',
            3: 'Total Endodontic Failure'
        }
        
        self.current_index = 0
        self.changes_made = {}
        
        self.setup_ui()
        self.show_image(0)
    
    def setup_ui(self):
        top_frame = tk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        tk.Label(top_frame, text="Image:", font=("Arial", 12)).pack(side=tk.LEFT)
        self.image_label = tk.Label(top_frame, text="", font=("Arial", 12, "bold"))
        self.image_label.pack(side=tk.LEFT, padx=5)
        
        tk.Label(top_frame, text="Confidence:", font=("Arial", 12)).pack(side=tk.LEFT, padx=(20,0))
        self.confidence_label = tk.Label(top_frame, text="", font=("Arial", 12))
        self.confidence_label.pack(side=tk.LEFT, padx=5)
        
        middle_frame = tk.Frame(self.root)
        middle_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.canvas = tk.Canvas(middle_frame, bg='gray')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        label_frame = tk.LabelFrame(self.root, text="Correct Label (if needed):", font=("Arial", 11, "bold"))
        label_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.var = tk.IntVar(value=0)
        for i in range(4):
            tk.Radiobutton(label_frame, text=f"{i}: {self.class_names[i]}", 
                          variable=self.var, value=i, font=("Arial", 10)).pack(anchor=tk.W, padx=10)
        
        button_frame = tk.Frame(self.root)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Button(button_frame, text="Previous", command=self.prev_image, width=12).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Accept Label", command=self.accept_label, width=12, bg="lightgreen").pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Correct Label", command=self.correct_label, width=12, bg="lightyellow").pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Next", command=self.next_image, width=12).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Save & Exit", command=self.save_and_exit, width=12, bg="lightcoral").pack(side=tk.LEFT, padx=5)
        
        self.status_label = tk.Label(self.root, text="", font=("Arial", 10), fg="blue")
        self.status_label.pack(fill=tk.X, padx=10, pady=5)
    
    def show_image(self, index):
        if index < 0 or index >= len(self.df):
            return
        
        self.current_index = index
        row = self.df.iloc[index]
        
        image_path = self.images_dir / row['image_path']
        if not image_path.exists():
            self.image_label.config(text=f"{row['image_path']} (NOT FOUND)")
            self.confidence_label.config(text="N/A")
            return
        
        self.image_label.config(text=f"{index + 1}/{len(self.df)} - {row['image_path']}")
        self.confidence_label.config(text=f"{row['confidence']:.4f}")
        
        img = Image.open(image_path)
        img.thumbnail((800, 600), Image.Resampling.LANCZOS)
        
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(400, 300, image=self.photo)
        
        current_label = int(row['label'])
        self.var.set(current_label)
        
        status = f"Current label: {current_label} ({self.class_names[current_label]})"
        if row['image_path'] in self.changes_made:
            status += f" → Changed to {self.changes_made[row['image_path']]}"
        self.status_label.config(text=status)
    
    def next_image(self):
        self.show_image(self.current_index + 1)
    
    def prev_image(self):
        self.show_image(self.current_index - 1)
    
    def accept_label(self):
        self.next_image()
    
    def correct_label(self):
        row = self.df.iloc[self.current_index]
        new_label = self.var.get()
        self.changes_made[row['image_path']] = new_label
        self.df.at[self.current_index, 'label'] = new_label
        self.df.at[self.current_index, 'class_name'] = self.class_names[new_label]
        messagebox.showinfo("Success", f"Label updated to: {new_label} ({self.class_names[new_label]})")
        self.next_image()
    
    def save_and_exit(self):
        if not self.changes_made:
            self.root.destroy()
            return
        
        if messagebox.askyesno("Save Changes", f"Save {len(self.changes_made)} label changes?"):
            self.df.to_csv(self.csv_path, index=False)
            messagebox.showinfo("Success", f"✓ Saved {len(self.changes_made)} changes to {self.csv_path}")
        
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = LabelReviewGUI(root)
    root.mainloop()
