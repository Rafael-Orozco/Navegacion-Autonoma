import numpy as np
import cv2
import tensorflow as tf
from controller import Keyboard, Camera
from vehicle import Driver

# Cargar modelo entrenado
model = tf.keras.models.load_model(
    r"D:\Personal\Maestria\Navegacion Autonoma\Proyecto final\Proyecto final 2\MR4010 Proyecto Final 2025\MR4010 Proyecto Final 2025\controllers\autonomous_driver\modelo_nvidia.h5",
    compile=False
)

# Preprocesamiento igual al entrenamiento
def preprocess_image(image):
    image = image[60:-25, :, :]
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=150)
    line_img = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) > 20:  # ignorar líneas casi horizontales
                cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    combined = cv2.addWeighted(image, 0.8, line_img, 1, 0)
    resized = cv2.resize(combined, (200, 66))
    return resized

# Obtener imagen RGB desde Webots
def get_image(camera):
    raw = camera.getImage()
    img = np.frombuffer(raw, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
    return img[:, :, :3]

# Corrección visual centrada en línea punteada del medio
def calculate_center_line_offset(preprocessed):
    gray = cv2.cvtColor(preprocessed, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=40, maxLineGap=50)

    width = preprocessed.shape[1]
    cx = width // 2
    x_coords = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2((y2 - y1), (x2 - x1)) * 180 / np.pi
            if abs(angle) > 60 and width * 0.3 < x1 < width * 0.7 and width * 0.3 < x2 < width * 0.7:
                x_coords.extend([x1, x2])

    if x_coords:
        detected_center = int(np.mean(x_coords))
        error = (detected_center - cx) / cx
        return error * 0.5  # ganancia suave
    return 0.0

# Main
def main():
    robot = Driver()
    timestep = int(robot.getBasicTimeStep())

    camera = robot.getDevice("camera")
    camera.enable(timestep)

    keyboard = Keyboard()
    keyboard.enable(timestep)

    speed = 70.0
    robot.setCruisingSpeed(speed)

    while robot.step() != -1:
        image = get_image(camera)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        preprocessed = preprocess_image(image_rgb)
        input_tensor = np.expand_dims(preprocessed, axis=0)

        # Predicción de la red neuronal
        steering_angle = float(model.predict(input_tensor, verbose=0)[0][0]) * 6

        # Corrección visual basada en la línea punteada
        visual_corr = calculate_center_line_offset(preprocessed)
        steering_angle += visual_corr

        # Aplicar atenuación para giros grandes
        if abs(steering_angle) > 0.5:
            steering_angle *= 0.85

        # Control manual suave con teclas derecha e izquierda
        key = keyboard.getKey()
        if key == keyboard.RIGHT:
            steering_angle = 0.25
        elif key == keyboard.LEFT:
            steering_angle = -0.25
        else:
            # Control manual de velocidad
            if key == keyboard.UP:
                speed += 2
            elif key == keyboard.DOWN:
                speed = max(0, speed - 2)

        steering_angle = np.clip(steering_angle, -1.0, 1.0)
        robot.setSteeringAngle(steering_angle)
        robot.setCruisingSpeed(speed)

        print(f"Steering: {steering_angle:.4f} | Visual Corr: {visual_corr:.4f} | Speed: {speed:.1f}")

if __name__ == "__main__":
    main()
