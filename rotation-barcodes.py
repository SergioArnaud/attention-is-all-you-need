from pylsd import lsd
from scipy import ndimage
def rotate_image(barcode_img):
    '''Obtemos rotacion de imagen de acuerdo a patron de lineas usando Hough,
    se obtiene la orientacion de estas lineas, y de acuerdo a la moda de estas
    lineas usando histogramas se rota la imagen'''

    # Se pasa a grayscale si esta a color
    img_gray = cv2.cvtColor(barcode_img, cv2.COLOR_BGR2GRAY)

    # Se buscan lineas
    lines = lsd(img_gray, scale=0.5)

    # Obtencion de angulos
    angles = []
    if lines is not None:
        for a in lines:
            x1,y1,x2,y2,w = a
            angle = 180/3.1416 * np.arctan2(y2-y1, x2-x1)
            angles.append(angle)

    if len(angles) > 0:
        angles = np.array(angles)
        hist = np.histogram(angles, 40)
        max_arg = np.argmax(hist[0])
        angle = 90 + hist[1][max_arg]
    else:
        # Angulo default para cancelar la rotacion en el siguiente paso
        angle = -90

    # Rotate
    rotated = ndimage.rotate(barcode_img, angle, cval=255)

    return rotated
