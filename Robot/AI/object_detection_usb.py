import argparse
import os
import sys
import time
import cv2 
import numpy as np
import traceback

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

def main():
    cam_w, cam_h = 640, 480
    default_model_dir = '/home/pi/Unirail/Robot/AI'
    default_model = 'road_signs_quantized.tflite'
    default_labels = 'road_signs_labels.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir,default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=5,
                        help='number of categories with highest score to display')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='classifier score threshold')
    args = parser.parse_args()

    with open(args.labels, 'r') as f:
        pairs = (l.strip().split(maxsplit=1) for l in f.readlines())
        labels = dict((int(k), v) for k, v in pairs)

    print('Loading {} with {} labels.'.format(args.model, args.labels))

    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = read_label_file(args.labels)

    
    camera = cv2.VideoCapture(0)
    ret = camera.set(3,cam_w)
    ret = camera.set(4,cam_h)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,cam_h-10)
    fontScale = 1
    fontColor = (255,255,255)  # white
    boxColor = (0,0,255)   # RED?
    boxLineWidth = 1
    lineType = 2

    inference_size = input_size(interpreter)
    scale_x, scale_y = cam_w / inference_size[0], cam_h / inference_size[1]

    try:
        last_time = time.monotonic()
        while camera.isOpened():
            try:
                ret, frame = camera.read()
                frame=cv2.rotate(frame,cv2.ROTATE_180)
                if ret == False :
                    print('can NOT read from camera')
                    break
                
                start_time = time.monotonic()
                input = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img=cv2.resize(input,inference_size)
                run_inference(interpreter, img.flatten())
                results = get_objects(interpreter, args.threshold)[:args.top_k]
                stop_time = time.monotonic()
                inference_ms = (stop_time - start_time)*1000.0
                fps_ms = 1.0 / (stop_time - last_time)
                last_time = stop_time

                for result in results:
                    bbox = result.bbox.scale(scale_x, scale_y)
                    coord_top_left = (int(bbox.xmin), int(bbox.ymin))
                    coord_bottom_right = (int(bbox.xmin+bbox.width), int(bbox.ymin+bbox.height))
                    cv2.rectangle(frame, coord_top_left, coord_bottom_right, boxColor, boxLineWidth)
                    annotate_text = "%s, %.0f%%" % (labels.get(result.id, result.id), result.score * 100)
                    coord_top_left = (coord_top_left[0],coord_top_left[1]+15)
                    cv2.putText(frame, annotate_text, coord_top_left, font, fontScale, boxColor, lineType )	
            
                annotate_text = 'Inference: {:5.2f}ms FPS: {:3.1f}'.format(inference_ms, fps_ms)
                cv2.putText(frame, annotate_text, bottomLeftCornerOfText, font, fontScale, boxColor, lineType )	
                cv2.imshow('Detected Objects', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            except:
                # catch it and don't exit the while loop
                print('In except')
                traceback.print_exc()

    finally:
        print('In Finally')
        camera.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
