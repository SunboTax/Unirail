# -*- coding: utf-8 -*-

# BIBLIOTHEQUES
import cv2
import numpy as np
from math import acos as arccos, pi, sqrt
from time import perf_counter


def findDeviationAndGap(lines : list) -> tuple:
    """Calcul de l'angle que fait la direction du robot avec le cap, en degrés
    L'angle est compté négativement si le robot va trop à droite, positivement s'il va trop à gauche
    """
    
    coord_r, coord_c = lines
    pt1_r, pt2_r = coord_r
    x1_r, y1_r = pt1_r
    x2_r, y2_r = pt2_r
    pt1_c, pt2_c = coord_c
    x1_c, y1_c = pt1_c
    x2_c, y2_c = pt2_c
    
    a = sqrt((x2_c-x1_c)**2 + (y2_c-y1_c)**2)
    b = sqrt((x2_r-x1_r)**2 + (y2_r-y1_r)**2)
    c = x2_c-x1_c + x2_r-x1_r
    
    theta = abs(round((arccos((c**2 - a**2 - b**2) / (2*a*b))-pi)*180.0/pi))
    if x2_c < x1_c :
        theta = -theta

    eps = x2_r - x2_c

    return (theta, eps)
