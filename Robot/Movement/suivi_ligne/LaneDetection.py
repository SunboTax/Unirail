# -*- coding: utf-8 -*-

# BIBLIOTHEQUES
import cv2
import numpy as np
from math import cos, sin, pi
from time import perf_counter

# DESCRIPTIF DES FONCTIONS
# get_ROI() renvoie une image aux mêmes dimensions, dont seule la région d'intérêt n'est pas masquée [divers polygones, choix dans la fonction]
# get_edges() renvoie une image B&W aux mêmes dimensions, dont seuls les contours détectés sont en blanc
# detect_lines() renvoie une liste de tuples (rho, theta) décrivant en polaire les lignes détectées sur une image de contours, selon des paramètres [choix dans la fonction]
# take_average() renvoie un tuple de 3 listes [x1,x2] où f(x1) = hauteur et f(x2) = hauteur//4, à partir d'une liste de tuple (rho, theta), décrivant les extrémités de :
#     - la moyenne des droites décrivant la ligne de gauche (si possible)
#     - la moyenne des droites décrivant la ligne de droite (si possible)
#     - la moyenne de ces 2 lignes (ligne de cap, si possible)
# detect_lanes() synthétise toutes ces fonctions et renvoie, à partir d'une image, une liste de 3 tuples (pt1,pt2) décrivant les lignes de gauche, de droite et de cap entre hauteur et hauteur//4 (si possible)
# print_lanes() permet, à partir d'une liste décrivant les lignes, de les dessiner sur une image 



def getEdges(image : np.ndarray) -> np.ndarray :
    """Détecte les contours séparant les éléments contrastés de la région d'intérêt d'une image.

    Les contours à détecter doivent séparer des éléments assez contrastés (attention à la luminosité),
      puisqu'une fonction de seuil automatique est appliquée avant de calculer le gradient d'intensité
      en chaque point.
    """
    #On ne retient que la région d'intérêt de l'image en noir et blanc
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #On applique une fonction de seuil (seuil trouvé automatiquement) pour distinguer les lignes
    ret, bin_image = cv2.threshold(gray_image,70,255,cv2.THRESH_BINARY_INV)
    
    # On ouvre l'image (= érosion + dilatation), pour supprimer le bruit blanc parasite autour :
    #  1 - on garde un point blanc uniquement si TOUS ceux du kernel sont blancs
    #  2 - si un point du kernel est blanc, alors TOUS les points du kernel sont mis en blanc
    kernel = np.ones((7,7),np.uint8)
    opened_image = cv2.morphologyEx(bin_image, cv2.MORPH_OPEN, kernel)
    
    #On détecte les contours, et on renvoie l'image
    canny_image = cv2.Canny(opened_image, 0, 1)
    return canny_image 



def detectCapAndDirection(edges : np.ndarray, y1 : int, y2 : int, L : list, plot : bool = True) -> (list, bool) :
    """
    """
    points = []
    h, w = edges.shape[:2]
    if not(0 <= y1 < h and 0 <= y2 < h) :
        return None
    if y1 < y2 :
        y1, y2 = y2, y1
    
    # Pour trouver les abscisses du faux contour droit de la chenille
    x_edge1, x_edge2 = w-1, w-1
    while x_edge1 >= 0 and edges[y1, x_edge1] == 0 :
        x_edge1 -= 1
    x_edge1 += 1
    while x_edge2 >= 0 and edges[y2, x_edge2] == 0 :
        x_edge2 -= 1
    x_edge2 += 1
    if x_edge1 == 0 or x_edge2 == 0 :
        return None
    
    # Pour trouver les abscisses du vrai contour gauche de la chenille
    x_cater1, x_cater2 = x_edge1 - 3, x_edge2 - 3
    while x_cater1 >= 0 and edges[y1, x_cater1] == 0 :
        x_cater1 -= 1
    x_cater1 += 1
    while x_cater2 >= 0 and edges[y2, x_cater2] == 0 :
        x_cater2 -= 1
    x_cater2 += 1
    if x_cater1 == 0 or x_cater2 == 0 :
        return None
    points.append([(x_cater1, y1), (x_cater2, y2)])
    
    # Pour trouver les abscisses de la ligne (ext droite)
    x_line1, x_line2 = x_cater1 - 3, x_cater2 - 3
    while x_line1 >= 0 and edges[y1, x_line1] == 0 :
        x_line1 -= 1
    x_line1 += 1
    while x_line2 >= 0 and edges[y2, x_line2] == 0 :
        x_line2 -= 1
    x_line2 += 1
    if x_line1 == 0 or x_line2 == 0 :
        return None
    points.append([(x_line1, y1), (x_line2, y2)])
    
    #Pour trouver les abscisses de la ligne (ext gauche)
    x_line3, x_line4 = x_line1 - 1, x_line2 - 1
    while x_line3 >= 0 and edges[y1, x_line3] == 0 :
        x_line3 -= 1
    x_line3 += 1
    while x_line4 >= 0 and edges[y2, x_line4] == 0 :
        x_line4 -= 1
    x_line4 += 1
    if x_line3 == 0 or x_line4 == 0 :
        return None
    points.append([(x_line3, y1), (x_line4, y2)])

    if plot :
        cv2.line(edges, (0,y1), (w-1,y1), (255,255,255), 2)
        cv2.line(edges, (0,y2), (w-1,y2), (255,255,255), 2)
    
    turnRight = False
    if len(L) > 0 :
        last_Pt3, last_Pt4 = L[-1]
        last_x_line3, last_x_line4 = last_Pt3[0], last_Pt4[0]
        if abs(last_x_line3 - x_cater1) < 10 or abs(last_x_line4 - x_cater2) < 10 :
            turnRight = True

    # On renvoie les coordonnées des 4 points trouvés
    return (points, turnRight)
    


def printLines(image : np.ndarray, lines : list) -> None :
    """
    """
    pt1, pt2 = lines[0]
    cv2.line(image, pt1, pt2, (255,0,255), 1)
    x, y = pt2
    cv2.putText(image, "Robot", (x-15, y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.4, (255,0,255), 1)
    
    pt1, pt2 = lines[1]
    cv2.line(image, pt1, pt2, (0,0,255), 1)
    x, y = pt2
    cv2.putText(image, "Centre", (x-25, y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.4, (0,0,255), 1)



def main() :
    #On connecte la caméra du Raspberry
    camera = cv2.VideoCapture(0)
    ori_h, ori_w = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    rescale = 25
    reth, retw = camera.set(cv2.CAP_PROP_FRAME_HEIGHT, ori_h*rescale//100), camera.set(cv2.CAP_PROP_FRAME_WIDTH, ori_w*rescale//100)

    h, w = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    image_center = (w//2, h//2)
    rot_mat = cv2.getRotationMatrix2D(image_center, -123, 1.0)
    
    last_time = perf_counter()
    last_theta = 0

    while True :
        #On prend l'image de la caméra
        ret, frame = camera.read()
        if ret is False:
            break
        image = cv2.warpAffine(frame, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255,255,255))
        
        #On mesure le temps écoulé depuis la dernière prise pour en déduire les fps, qu'on écrit sur l'image
        cur_time = perf_counter()
        fps = int(1/(cur_time-last_time))
        last_time = cur_time
        
        #On dessine la détection des lignes sur l'image
        
        #lanes_desc = detect_lanes(image)
        #print_lanes(image, lanes_desc)
        edges = getEdges(image)
        lines = detectCapAndDirection(edges, 97*h//100, 70*h//100)
        if lines is not None :
            printLines(image, lines)

        #On affiche l'image, et si on appuie sur 'q', on quitte
        # resize image
        resized_image = cv2.resize(image, (ori_h, ori_w), interpolation = cv2.INTER_LINEAR)
        resized_edges = cv2.resize(edges, (ori_h, ori_w), interpolation = cv2.INTER_LINEAR)
        cv2.imshow("Image", resized_image)
        cv2.imshow("Contours", resized_edges)

        key = cv2.waitKey(1)
        if key == ord('q') :
            break

    #On déconnecte la caméra du Raspberry et on ferme tous les affichages
    camera.release()
    cv2.destroyAllWindows()
    

if __name__=="__main__" :
    main()