import os
import shutil
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

class ImageSortingApp:
    def __init__(self, root, move_files=True):
        self.root = root
        self.root.title("AI-Assisted Image Sorting")
        self.root.geometry("900x700")
        
        # Control whether to move files or just view them
        self.move_files = move_files
        
        # Define folders
        self.source_folder = "dataset/new"
        self.wanted_folder = "dataset/wanted"
        self.unwanted_folder = "dataset/unwanted"
        
        # Create directories if they don't exist
        os.makedirs(self.wanted_folder, exist_ok=True)
        os.makedirs(self.unwanted_folder, exist_ok=True)
        
        # Load the model
        try:
            self.model = load_model("image_classifier.h5")
            self.model_loaded = True
        except:
            self.model_loaded = False
            messagebox.showwarning("Model Not Found", "Could not load 'image_classifier.h5'. Auto-classification disabled.")
        
        # Get list of images
        self.image_files = [f for f in os.listdir(self.source_folder) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.current_index = 0
        
        # Keep track of classifications in view-only mode
        self.classifications = {}
        
        # Set up the GUI
        self.setup_ui()
        
        # Load first image if available
        if self.image_files:
            self.load_current_image()
        else:
            messagebox.showinfo("No Images", "No images found in the source folder.")
    
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Mode indicator
        mode_text = "MOVE FILES MODE" if self.move_files else "VIEW-ONLY MODE"
        mode_color = "red" if self.move_files else "green"
        self.mode_label = ttk.Label(main_frame, text=mode_text, foreground=mode_color, font=("Arial", 12, "bold"))
        self.mode_label.pack(fill=tk.X, pady=5)
        
        # Toggle mode button
        self.toggle_mode_btn = ttk.Button(main_frame, text="Toggle Mode", command=self.toggle_mode)
        self.toggle_mode_btn.pack(pady=5)
        
        # Image frame
        self.image_frame = ttk.Frame(main_frame)
        self.image_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Image canvas
        self.canvas = tk.Canvas(self.image_frame, bg="black", height=500)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Information frame
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill=tk.X, pady=5)
        
        # File information
        self.file_info = ttk.Label(info_frame, text="")
        self.file_info.pack(side=tk.LEFT)
        
        # AI prediction
        self.prediction_label = ttk.Label(info_frame, text="")
        self.prediction_label.pack(side=tk.RIGHT)
        
        # Classification info (for view-only mode)
        self.classification_label = ttk.Label(info_frame, text="", foreground="blue")
        self.classification_label.pack(side=tk.RIGHT, padx=10)
        
        # Control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        # Progress
        self.progress_var = tk.DoubleVar()
        progress = ttk.Progressbar(control_frame, variable=self.progress_var, length=200)
        progress.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        
        # Buttons
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill=tk.X)
        
        self.auto_btn = ttk.Button(btn_frame, text="Auto Sort Remaining", command=self.auto_sort_remaining)
        self.auto_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="Previous", command=self.previous_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Skip", command=self.next_image).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="Wanted (w)", command=self.mark_wanted).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="Unwanted (u)", command=self.mark_unwanted).pack(side=tk.RIGHT, padx=5)
        
        # Keyboard bindings
        self.root.bind('w', lambda event: self.mark_wanted())
        self.root.bind('u', lambda event: self.mark_unwanted())
        self.root.bind('<Left>', lambda event: self.previous_image())
        self.root.bind('<Right>', lambda event: self.next_image())
        self.root.bind('<Escape>', lambda event: self.root.quit())
        self.root.bind('t', lambda event: self.toggle_mode())
    
    def toggle_mode(self):
        self.move_files = not self.move_files
        mode_text = "MOVE FILES MODE" if self.move_files else "VIEW-ONLY MODE"
        mode_color = "red" if self.move_files else "green"
        self.mode_label.config(text=mode_text, foreground=mode_color)
        
        # Update classification label
        self.update_classification_label()
    
    def update_classification_label(self):
        if not self.move_files and self.current_index < len(self.image_files):
            current_image = self.image_files[self.current_index]
            if current_image in self.classifications:
                self.classification_label.config(
                    text=f"Marked as: {self.classifications[current_image]}",
                    foreground="blue" if self.classifications[current_image] == "wanted" else "orange"
                )
            else:
                self.classification_label.config(text="Not classified yet")
        else:
            self.classification_label.config(text="")
    
    def load_current_image(self):
        if not self.image_files or self.current_index >= len(self.image_files):
            messagebox.showinfo("Complete", "No more images to sort.")
            return False
        
        # Update progress
        self.progress_var.set((self.current_index / len(self.image_files)) * 100)
        
        # Get current image
        self.current_image = self.image_files[self.current_index]
        img_path = os.path.join(self.source_folder, self.current_image)
        
        # Update file info
        self.file_info.config(text=f"Image {self.current_index + 1}/{len(self.image_files)}: {self.current_image}")
        
        # Display image
        try:
            # Open image with PIL
            pil_img = Image.open(img_path)
            
            # Calculate resize dimensions keeping aspect ratio
            img_width, img_height = pil_img.size
            max_size = 700
            if img_width > max_size or img_height > max_size:
                ratio = min(max_size / img_width, max_size / img_height)
                new_width = int(img_width * ratio)
                new_height = int(img_height * ratio)
                pil_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
            
            # Convert to Tkinter compatible image
            self.tk_img = ImageTk.PhotoImage(pil_img)
            
            # Update canvas size and image
            self.canvas.config(width=self.tk_img.width(), height=self.tk_img.height())
            self.canvas.create_image(self.tk_img.width()//2, self.tk_img.height()//2, image=self.tk_img)
            
            # Get AI prediction if model is loaded
            if self.model_loaded:
                prediction = self.classify_image(img_path)
                self.prediction_label.config(text=f"AI suggests: {prediction}")
            else:
                self.prediction_label.config(text="AI prediction not available")
            
            # Update classification label
            self.update_classification_label()
            
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Could not load image: {str(e)}")
            self.next_image()
            return False
    
    def classify_image(self, img_path):
        try:
            img = image.load_img(img_path, target_size=(128, 128))
            img_array = image.img_to_array(img) / 255.0  # Normalize
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            prediction = self.model.predict(img_array, verbose=0)[0][0]
            return "wanted" if prediction > 0.5 else "unwanted"
        except Exception as e:
            print(f"Error classifying image: {str(e)}")
            return "unknown"
    
    def handle_image(self, classification):
        if not self.image_files or self.current_index >= len(self.image_files):
            return
        
        current_image = self.image_files[self.current_index]
        
        if self.move_files:
            # Move the file to the appropriate folder
            destination = self.wanted_folder if classification == "wanted" else self.unwanted_folder
            source_path = os.path.join(self.source_folder, current_image)
            dest_path = os.path.join(destination, current_image)
            
            try:
                shutil.move(source_path, dest_path)
                # Remove from list after moving
                self.image_files.pop(self.current_index)
                
                # Don't increment current_index since the list is now shorter
                if self.image_files:
                    self.load_current_image()
                else:
                    messagebox.showinfo("Complete", "All images have been sorted!")
                    self.file_info.config(text="No more images to sort")
                    self.prediction_label.config(text="")
                    self.classification_label.config(text="")
                    self.canvas.delete("all")
            except Exception as e:
                messagebox.showerror("Error", f"Could not move file: {str(e)}")
        else:
            # Just record the classification but don't move the file
            self.classifications[current_image] = classification
            self.update_classification_label()
            self.next_image()
    
    def mark_wanted(self):
        self.handle_image("wanted")
    
    def mark_unwanted(self):
        self.handle_image("unwanted")
    
    def next_image(self):
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.load_current_image()
    
    def previous_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_image()
    
    def auto_sort_remaining(self):
        if not self.model_loaded:
            messagebox.showwarning("Model Not Loaded", "Cannot auto sort without a trained model.")
            return
        
        if not self.image_files:
            messagebox.showinfo("No Images", "No images to sort.")
            return
        
        action_text = "classify" if not self.move_files else "sort"
        if messagebox.askyesno("Confirm", f"Auto {action_text} all remaining images based on AI predictions?"):
            # Count for summary
            wanted_count = 0
            unwanted_count = 0
            failed_count = 0
            
            # Get current position to start from
            starting_index = self.current_index
            total_remaining = len(self.image_files) - starting_index
            original_files = self.image_files.copy()
            
            for i in range(starting_index, len(original_files)):
                # Update progress for long operations
                progress_index = i - starting_index
                self.progress_var.set((progress_index / total_remaining) * 100)
                self.root.update()
                
                # If we're in move mode, the list keeps shrinking
                if self.move_files:
                    if not self.image_files:
                        break
                    img_name = self.image_files[self.current_index]
                else:
                    # In view-only mode, the list stays the same
                    if i >= len(self.image_files):
                        break
                    img_name = self.image_files[i]
                    
                img_path = os.path.join(self.source_folder, img_name)
                
                try:
                    prediction = self.classify_image(img_path)
                    if prediction == "wanted":
                        if self.move_files:
                            self.handle_image("wanted")
                        else:
                            self.classifications[img_name] = "wanted"
                        wanted_count += 1
                    else:
                        if self.move_files:
                            self.handle_image("unwanted")
                        else:
                            self.classifications[img_name] = "unwanted"
                        unwanted_count += 1
                except:
                    failed_count += 1
                    if self.move_files:
                        self.next_image()
            
            # Show summary
            messagebox.showinfo(f"Auto {action_text.title()} Complete", 
                              f"{action_text.title()}ed {wanted_count + unwanted_count} images\n"
                              f"- Wanted: {wanted_count}\n"
                              f"- Unwanted: {unwanted_count}\n"
                              f"- Failed: {failed_count}")
            
            # If in view-only mode, go back to the first image
            if not self.move_files:
                self.current_index = 0
                self.load_current_image()
                
# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSortingApp(root, move_files=False)  # Set move_files=False for view-only mode
    root.mainloop()