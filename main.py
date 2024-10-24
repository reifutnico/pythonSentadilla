import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
#https://youtube.com/shorts/T0m9Q3dGu1c?si=vRn1hpDO7ReOkzOW
#https://www.youtube.com/watch?v=9LSIy2NWmiA


#NumPy (np) maneja operaciones con arreglos y cálculos matemáticos.
#FastDTW: Algoritmo de Dynamic Time Warping que compara secuencias temporales.
#SciPy: Contiene la función euclidean para calcular distancias entre puntos.
#cv2: Usada para manipulación de video.
#YOLO: Red neuronal que detecta la pose humana en imágenes/videos.
#Matplotlib (plt): Crea gráficas para visualizar resultados.


class AnalizadorSentadillas:
    def __init__(self):
        self.model = YOLO('yolov8n-pose.pt') # Carga el modelo YOLO entrenado para detección de poses
        self.umbrales = {  # Umbrales para detectar posibles errores
            'rodillas': {
                'min_flexion': 80,  # Ángulo mínimo de flexión de rodilla
                'max_flexion': 100,  # Ángulo máximo de flexión de rodilla
                'diferencia_maxima': 15  # Diferencia máxima entre rodillas
            },
            'cadera': {
                'min_flexion': 70,  # Ángulo mínimo de flexión de cadera
                'max_flexion': 100  # Ángulo máximo de flexión de cadera
            },
            'tobillos': {
                'min_angulo': 70,  # Ángulo mínimo de tobillo
                'max_angulo': 90   # Ángulo máximo de tobillo
            },
            'espalda': {
                'max_inclinacion': 45  # Inclinación máxima de la espalda
            }
        }
        
    def calcular_angulo(self, p1, p2, p3):
        """
        Calcula el ángulo entre tres puntos
        """
        a = np.array(p1) #convierte el punto en array
        b = np.array(p2)
        c = np.array(p3)
        
        ba = a - b 
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle) #radianes
        
        return np.degrees(angle) #angulos

    def extraer_angulos_frame(self, keypoints):
        """
        Extrae los ángulos relevantes de los keypoints de un frame
        """
        # Índices de los keypoints relevantes en YOLO
        RODILLA_DER = 14
        CADERA_DER = 12
        TOBILLO_DER = 16
        RODILLA_IZQ = 13
        CADERA_IZQ = 11
        TOBILLO_IZQ = 15
        HOMBRO_DER = 6
        HOMBRO_IZQ = 5
        
        try:
            # Calcular ángulos de rodillas
            rodilla_der = self.calcular_angulo(
                keypoints[CADERA_DER], 
                keypoints[RODILLA_DER], 
                keypoints[TOBILLO_DER]
            )
            
            rodilla_izq = self.calcular_angulo(
                keypoints[CADERA_IZQ], 
                keypoints[RODILLA_IZQ], 
                keypoints[TOBILLO_IZQ]
            )
            
            # Calcular ángulo de cadera
            cadera = self.calcular_angulo(
                keypoints[HOMBRO_DER], 
                keypoints[CADERA_DER], 
                keypoints[RODILLA_DER]
            )
            
            # Calcular ángulos de tobillos
            tobillo_der = self.calcular_angulo(
                keypoints[RODILLA_DER],
                keypoints[TOBILLO_DER],
                [keypoints[TOBILLO_DER][0], keypoints[TOBILLO_DER][1] + 10]  # Punto en el suelo
            )
            
            tobillo_izq = self.calcular_angulo(
                keypoints[RODILLA_IZQ],
                keypoints[TOBILLO_IZQ],
                [keypoints[TOBILLO_IZQ][0], keypoints[TOBILLO_IZQ][1] + 10]
            )
            
            # Calcular ángulo de espalda (con respecto a la vertical)
            espalda = self.calcular_angulo(
                keypoints[HOMBRO_DER],
                keypoints[CADERA_DER],
                [keypoints[CADERA_DER][0], keypoints[CADERA_DER][1] - 10]  # Punto vertical
            )
            
            return np.array([
                rodilla_der, rodilla_izq,
                cadera,
                tobillo_der, tobillo_izq,
                espalda
            ])
            
        except:
            return None

    def procesar_video(self, video_path):
        """
        Procesa un video y extrae los ángulos de cada frame
        """
        cap = cv2.VideoCapture(video_path)
        angulos_frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            results = self.model(frame)
            keypoints = results[0].keypoints.cpu().numpy()[0]
            
            angulos = self.extraer_angulos_frame(keypoints)
            if angulos is not None:
                angulos_frames.append(angulos)
                
        cap.release()
        return np.array(angulos_frames)

    def analizar_sentadilla(self, video_referencia, video_prueba):
        """
        Analiza una sentadilla comparándola con una referencia
        """
        # Procesar videos
        angulos_referencia = self.procesar_video(video_referencia)
        angulos_prueba = self.procesar_video(video_prueba)
        
        # Comparar usando DTW
        distancia, camino = fastdtw(angulos_referencia, angulos_prueba)
        
        # Analizar errores
        errores = self.detectar_errores(angulos_prueba)
        
        # Generar visualización
        self.visualizar_comparacion(angulos_referencia, angulos_prueba, camino)
        
        return errores, distancia

    def detectar_errores(self, angulos):
        """
        Detecta errores específicos en la ejecución de la sentadilla
        """
        errores = []
        
        # Analizar rodillas
        rodilla_der = angulos[:, 0]
        rodilla_izq = angulos[:, 1]
        
        if np.min(rodilla_der) > self.umbrales['rodillas']['min_flexion']:
            errores.append("La rodilla derecha no se flexiona lo suficiente")
        elif np.min(rodilla_der) < self.umbrales['rodillas']['max_flexion']:
            errores.append("La rodilla derecha se flexiona demasiado")
            
        if np.max(np.abs(rodilla_der - rodilla_izq)) > self.umbrales['rodillas']['diferencia_maxima']:
            errores.append("Las rodillas no están simétricas durante el movimiento")
        
        # Analizar cadera
        cadera = angulos[:, 2]
        if np.min(cadera) < self.umbrales['cadera']['min_flexion']:
            errores.append("La cadera se flexiona demasiado")
        elif np.min(cadera) > self.umbrales['cadera']['max_flexion']:
            errores.append("La cadera no se flexiona lo suficiente")
        
        # Analizar tobillos
        tobillo_der = angulos[:, 3]
        tobillo_izq = angulos[:, 4]
        
        if np.min(tobillo_der) < self.umbrales['tobillos']['min_angulo']:
            errores.append("El tobillo derecho tiene demasiada flexión")
        if np.min(tobillo_izq) < self.umbrales['tobillos']['min_angulo']:
            errores.append("El tobillo izquierdo tiene demasiada flexión")
        
        # Analizar espalda
        espalda = angulos[:, 5]
        if np.max(espalda) > self.umbrales['espalda']['max_inclinacion']:
            errores.append("La espalda está demasiado inclinada hacia adelante")
        
        return errores

    def visualizar_comparacion(self, angulos_ref, angulos_prueba, camino):
        """
        Genera gráficas comparativas de los ángulos
        """
        partes = ['Rodilla Der', 'Rodilla Izq', 'Cadera', 'Tobillo Der', 'Tobillo Izq', 'Espalda']
        
        plt.figure(figsize=(15, 10))
        for i, parte in enumerate(partes):
            plt.subplot(3, 2, i+1)
            plt.plot(angulos_ref[:, i], label='Referencia', color='green')
            plt.plot(angulos_prueba[:, i], label='Prueba', color='red')
            plt.title(parte)
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()

# Ejemplo de uso
def main():
    analizador = AnalizadorSentadillas()
    
    # Rutas de los videos (ajusta estas rutas según tus archivos)
    video_referencia = "sentadilla_perfecta.mp4"
    video_prueba = "sentadilla_prueba.mp4"
    
    # Analizar sentadilla
    errores, distancia = analizador.analizar_sentadilla(video_referencia, video_prueba)
    
    # Mostrar resultados
    print("\nResultados del análisis:")
    print(f"Distancia total DTW: {distancia:.2f}")
    
    if errores:
        print("\nErrores detectados:")
        for i, error in enumerate(errores, 1):
            print(f"{i}. {error}")
    else:
        print("\n¡Excelente! No se detectaron errores significativos.")

if __name__ == "__main__":
    main()