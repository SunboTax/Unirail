import cv2
from time import perf_counter, sleep
from LaneDetection import getEdges, detectCapAndDirection, printLines
from LaneComputation import findDeviationAndGap
from SerialCom_LaneFollowing import MegaPi

megapi = MegaPi()

#On connecte la caméra du Raspberry
camera = cv2.VideoCapture(0)
ori_h, ori_w = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
rescale = 60
reth, retw = camera.set(cv2.CAP_PROP_FRAME_HEIGHT, ori_h*rescale//100), camera.set(cv2.CAP_PROP_FRAME_WIDTH, ori_w*rescale//100)

h, w = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
image_center = (w//2, h//2)
rot_mat = cv2.getRotationMatrix2D(image_center, -123, 1.0)
    
last_time = perf_counter()
lines = []
pont = False
droite = False

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
    edges = getEdges(image)
    lines_turnIndication = detectCapAndDirection(edges, 85*h//100, 70*h//100, lines)
    if lines_turnIndication is None :
        lines = []
        turnRight = False
    else :
        (lines, turnRight) = lines_turnIndication
    theta, eps = 0, 0
    if turnRight :
        theta = 100
    if len(lines) > 0 :
        printLines(image, lines[:2])
        theta, eps = findDeviationAndGap(lines[:2])
        eps -= 60
        
    cv2.putText(image, f"FPS: {fps}, Theta: {theta}, Eps: {eps}", (2, 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.4, (0,0,255), 1)
    if pont :
        megapi.sendThetaEpsilonU(0, 0, 10)
        cv2.putText(image, "PONT", (2, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.4, (255,0,0), 1)
    elif droite :
        megapi.sendThetaEpsilonU(theta, eps, 20)
        cv2.putText(image, "DROITE", (2, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.4, (255,0,0), 1)
    else :
        megapi.sendThetaEpsilonU(theta, eps, 10)
        
    #On affiche l'image, et si on appuie sur 'q', on quitte
    resized_image = cv2.resize(image, (ori_h, ori_w), interpolation = cv2.INTER_LINEAR)
    resized_edges = cv2.resize(edges, (ori_h, ori_w), interpolation = cv2.INTER_LINEAR)
    cv2.imshow("Image + lignes détectées", resized_image)
    cv2.imshow("Contours", resized_edges)

    
    key = cv2.waitKey(1)
    if key == ord('q') :
        break
    if key == ord('p') :
        pont = not(pont)
    if key == ord('d') :
        droite = not(droite)
        
#On déconnecte la caméra du Raspberry et on ferme tous les affichages
camera.release()
cv2.destroyAllWindows()
megapi.endCom()
