import cv2
import numpy as np
from ultralytics import YOLO
import math
import os

def calculate_angle(a, b, c):
    """
    Calcula el ángulo entre tres puntos
    Args:
        a: primer punto [x, y]
        b: punto medio [x, y]
        c: último punto [x, y]
    Returns:
        ángulo en grados
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def calculate_body_angles(keypoints):
    """
    Calcula los ángulos de todas las partes principales del cuerpo
    Args:
        keypoints: array de puntos clave detectados
    Returns:
        diccionario con todos los ángulos corporales
    """
    body_angles = {}
    points = {}
    
    # Verificar confianza mínima para los puntos
    confidence_threshold = 0.5
    
    # Calcular puntos medios útiles
    if all(keypoints[[5,6], 2] > confidence_threshold):  # hombros
        points['mid_shoulders'] = (keypoints[5][:2] + keypoints[6][:2]) / 2
    if all(keypoints[[11,12], 2] > confidence_threshold):  # caderas
        points['mid_hips'] = (keypoints[11][:2] + keypoints[12][:2]) / 2
    if all(keypoints[[13,14], 2] > confidence_threshold):  # rodillas
        points['mid_knees'] = (keypoints[13][:2] + keypoints[14][:2]) / 2
    
    # 1. Análisis de Columna
    if all(key in points for key in ['mid_shoulders', 'mid_hips', 'mid_knees']) and keypoints[0,2] > confidence_threshold:
        # Punto de referencia vertical (100px arriba de los hombros)
        vertical_reference = np.array([points['mid_shoulders'][0], points['mid_shoulders'][1] - 100])
        
        body_angles['columna'] = {
            'superior': calculate_angle(keypoints[0][:2], points['mid_shoulders'], points['mid_hips']),
            'inferior': calculate_angle(points['mid_shoulders'], points['mid_hips'], points['mid_knees']),
            'inclinacion_lateral': calculate_angle(vertical_reference, points['mid_shoulders'], points['mid_hips'])
        }
    
    # 2. Análisis de Brazos
    for side, (shoulder, elbow, wrist) in {'izquierdo': (5,7,9), 'derecho': (6,8,10)}.items():
        if all(keypoints[[shoulder,elbow,wrist], 2] > confidence_threshold):
            body_angles[f'brazo_{side}'] = {
                'hombro': calculate_angle(points['mid_shoulders'], keypoints[shoulder][:2], keypoints[elbow][:2]),
                'codo': calculate_angle(keypoints[shoulder][:2], keypoints[elbow][:2], keypoints[wrist][:2])
            }
    
    # 3. Análisis de Piernas
    for side, (hip, knee, ankle) in {'izquierda': (11,13,15), 'derecha': (12,14,16)}.items():
        if all(keypoints[[hip,knee,ankle], 2] > confidence_threshold):
            body_angles[f'pierna_{side}'] = {
                'cadera': calculate_angle(points['mid_hips'], keypoints[hip][:2], keypoints[knee][:2]),
                'rodilla': calculate_angle(keypoints[hip][:2], keypoints[knee][:2], keypoints[ankle][:2])
            }
    
    # 4. Análisis de Cabeza/Cuello
    if all(keypoints[[0,5,6], 2] > confidence_threshold):  # nariz y hombros
        vertical_head = np.array([keypoints[0][0], keypoints[0][1] - 100])  # punto 100px arriba de la nariz
        body_angles['cabeza'] = {
            'inclinacion': calculate_angle(vertical_head, keypoints[0][:2], points['mid_shoulders'])
        }
    
    body_angles['puntos'] = points
    return body_angles

def draw_body_analysis(frame, body_angles):
    """
    Dibuja las líneas y ángulos de todas las partes del cuerpo analizadas
    """
    if not body_angles or 'puntos' not in body_angles:
        return frame
    
    # Colores para diferentes partes
    colors = {
        'columna': (0, 0, 255),    # Rojo
        'brazos': (255, 0, 0),     # Azul
        'piernas': (0, 255, 0),    # Verde
        'cabeza': (255, 255, 0)    # Amarillo
    }
    
    # Función auxiliar para dibujar texto
    def put_angle_text(text, position, line_num):
        cv2.putText(frame, text, (10, 30 * line_num), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    line_count = 1
    
    # 1. Mostrar ángulos de columna
    if 'columna' in body_angles:
        col = body_angles['columna']
        put_angle_text(f"Columna Sup.: {col['superior']:.1f}°", 10, line_count)
        line_count += 1
        put_angle_text(f"Columna Inf.: {col['inferior']:.1f}°", 10, line_count)
        line_count += 1
        put_angle_text(f"Incl. Lateral: {col['inclinacion_lateral']:.1f}°", 10, line_count)
        line_count += 1
    
    # 2. Mostrar ángulos de brazos
    for side in ['izquierdo', 'derecho']:
        key = f'brazo_{side}'
        if key in body_angles:
            angles = body_angles[key]
            put_angle_text(f"Hombro {side}: {angles['hombro']:.1f}°", 10, line_count)
            line_count += 1
            put_angle_text(f"Codo {side}: {angles['codo']:.1f}°", 10, line_count)
            line_count += 1
    
    # 3. Mostrar ángulos de piernas
    for side in ['izquierda', 'derecha']:
        key = f'pierna_{side}'
        if key in body_angles:
            angles = body_angles[key]
            put_angle_text(f"Cadera {side}: {angles['cadera']:.1f}°", 10, line_count)
            line_count += 1
            put_angle_text(f"Rodilla {side}: {angles['rodilla']:.1f}°", 10, line_count)
            line_count += 1
    
    # 4. Mostrar ángulos de cabeza
    if 'cabeza' in body_angles:
        put_angle_text(f"Incl. Cabeza: {body_angles['cabeza']['inclinacion']:.1f}°", 10, line_count)
    
    return frame

def process_video(source_path):
    try:
        if not os.path.exists(source_path):
            print(f"Error: El archivo {source_path} no existe")
            return
            
        print("Cargando modelo YOLOv8...")
        model = YOLO('yolov8s-pose.pt')
        
        # Definir las conexiones entre keypoints
        connections = [
            # Cara
            (0, 1), (0, 2),  # Nariz a ojos
            (1, 3), (2, 4),  # Ojos a orejas
            # Brazos
            (5, 7), (7, 9),    # Brazo izquierdo
            (6, 8), (8, 10),   # Brazo derecho
            # Torso
            (5, 6),    # Hombros
            (5, 11), (6, 12),  # Hombros a caderas
            (11, 12),  # Caderas
            # Piernas
            (11, 13), (13, 15),  # Pierna izquierda
            (12, 14), (14, 16)   # Pierna derecha
        ]
        
        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            print("Error: No se pudo abrir el video")
            return
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        output_path = 'output_video.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        keypoint_names = {
            0: "nariz", 1: "ojo_izq", 2: "ojo_der", 3: "oreja_izq", 4: "oreja_der",
            5: "hombro_izq", 6: "hombro_der", 7: "codo_izq", 8: "codo_der",
            9: "muñeca_izq", 10: "muñeca_der", 11: "cadera_izq", 12: "cadera_der",
            13: "rodilla_izq", 14: "rodilla_der", 15: "tobillo_izq", 16: "tobillo_der"
        }
        
        print("Procesando video... Presiona 'q' para salir")
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Fin del video")
                break
                
            frame_count += 1
            print(f"Procesando frame {frame_count}", end='\r')
            
            results = model(frame)
            
            if len(results[0].keypoints.data) > 0:
                for person in results[0].keypoints.data:
                    keypoints = person.cpu().numpy()
                    
                    # Calcular ángulos del cuerpo
                    body_angles = calculate_body_angles(keypoints)
                    
                    # Dibujar análisis del cuerpo
                    frame = draw_body_analysis(frame, body_angles)
                    
                    # Dibujar las conexiones del esqueleto
                    for connection in connections:
                        start_idx, end_idx = connection
                        if keypoints[start_idx][2] > 0.5 and keypoints[end_idx][2] > 0.5:
                            start_point = tuple(map(int, keypoints[start_idx][:2]))
                            end_point = tuple(map(int, keypoints[end_idx][:2]))
                            cv2.line(frame, start_point, end_point, (0, 255, 255), 2)
                    
                    # Dibujar los keypoints
                    for idx, point in enumerate(keypoints):
                        x, y = point[:2]
                        confidence = point[2]
                        
                        if confidence > 0.5:
                            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                            cv2.putText(frame, keypoint_names[idx], 
                                      (int(x), int(y)-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.5, (255, 255, 255), 1)
            
            cv2.imshow('Pose Detection', frame)
            out.write(frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nProceso interrumpido por el usuario")
                break
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"\nVideo procesado y guardado como: {output_path}")
        
    except Exception as e:
        print(f"\nError durante la ejecución: {str(e)}")
        print(f"Tipo de error: {type(e).__name__}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    video_path = "sentadilla.mp4"
    process_video(video_path)