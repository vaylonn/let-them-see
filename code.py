import cv2
import numpy as np
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from gtts import gTTS
import pygame
import io
import threading
import time
import os
from tkinter import Tk, filedialog

class AssistanceVisionSystem:
    def __init__(self):
        """Initialise le système d'assistance visuelle avec le modèle BLIP."""
        self.setup_model()
        pygame.mixer.init()
        
    def setup_model(self):
        """Configure le modèle BLIP pour le captioning."""
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        print("Modèle BLIP chargé avec succès")
    
    def generate_caption(self, image):
        """Génère une caption concise en français pour l'image fournie."""
        if isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        inputs = self.processor(image, return_tensors="pt")
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_length=20,
                num_beams=3,
                early_stopping=True,
                do_sample=False
            )
        caption = self.processor.decode(output_ids[0], skip_special_tokens=True)
        return caption.strip()
    
    def text_to_speech(self, text, lang='en'):
        """Convertit un texte en parole et le joue avec pygame."""
        try:
            tts = gTTS(text=text, lang=lang, slow=False)
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            pygame.mixer.music.load(audio_buffer)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        except Exception as e:
            print(f"Erreur lors de la synthèse vocale: {e}")
    
    def process_webcam(self, camera_index=0):
        """Utilise la webcam pour générer des descriptions vocales en temps réel."""
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("Erreur: Impossible d'ouvrir la caméra")
            return

        print("Mode webcam activé. Appuyez sur 'ESPACE' pour décrire l'image actuelle, 'q' pour quitter")
        last_description_time = 0
        description_interval = 3

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow('Vision Assistant - Appuyez sur ESPACE pour description', frame)

            key = cv2.waitKey(1) & 0xFF
            current_time = time.time()

            if key == ord(' '):
                if current_time - last_description_time > description_interval:
                    print("Analyse de l'image en cours...")

                    def describe_async():
                        caption = self.generate_caption(frame)
                        print(f"Description: {caption}")
                        self.text_to_speech(caption)

                    thread = threading.Thread(target=describe_async)
                    thread.daemon = True
                    thread.start()
                    last_description_time = current_time
                else:
                    print("Attendez quelques secondes avant la prochaine description")

            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    
    def process_image_file(self, image_path):
        """Traite une image depuis un fichier et lit sa description."""
        if not os.path.exists(image_path):
            print(f"Erreur: Le fichier {image_path} n'existe pas")
            return

        try:
            image = Image.open(image_path).convert('RGB')
            print("Analyse de l'image en cours...")
            caption = self.generate_caption(image)
            print(f"Description de l'image: {caption}")
            self.text_to_speech(caption)
        except Exception as e:
            print(f"Erreur lors du traitement de l'image: {e}")

def open_file_dialog():
    """Ouvre une boîte de dialogue pour choisir un fichier image."""
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_path = filedialog.askopenfilename(
        title="Choisissez une image",
        filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff")]
    )
    root.destroy()
    return file_path

def main():
    print("=== Système d'Assistance Visuelle pour Aveugles ===")
    print("Utilisation du modèle : BLIP")

    assistant = AssistanceVisionSystem()

    while True:
        print("\n=== Menu Principal ===")
        print("1. Mode webcam (temps réel)")
        print("2. Analyser une image (fichier)")
        print("3. Quitter")

        choice = input("Votre choix: ").strip()

        if choice == "1":
            assistant.process_webcam()
        elif choice == "2":
            image_path = open_file_dialog()
            if image_path:
                assistant.process_image_file(image_path)
            else:
                print("Aucune image sélectionnée.")
        elif choice == "3":
            print("Au revoir!")
            break
        else:
            print("Choix invalide, veuillez réessayer")

if __name__ == "__main__":
    main()
