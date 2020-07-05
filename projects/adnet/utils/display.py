import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import cv2

def draw_box_bigline(image, box,text):
    im_with_bb = image.copy()
    cv2.rectangle(im_with_bb, (int(box[0]), int(box[1])), (int(box[0] + box[2]), int(box[1] + box[3])), (0, 0, 255),4)
    # imgzi = cv2.putText(img, '000', (50, 300), font, 1.2, (255, 255, 0), 2)
    cv2.putText(im_with_bb, text, (int(box[0])+10, int(box[1])+40), cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 255, 0), 2)
    return im_with_bb

def draw_box(image, box):
    im_with_bb = image.copy()
    cv2.rectangle(im_with_bb, (int(box[0]), int(box[1])), (int(box[0] + box[2]), int(box[1] + box[3])), (0, 0, 255))
    return im_with_bb

def draw_boxes(image, boxes):
    if isinstance(boxes,list):
        im_with_bb = image
        for i in range(len(boxes)):
            im_with_bb = draw_box(im_with_bb, boxes[i])
    else:
        im_with_bb = draw_box(image, boxes)
    # if len(boxes[0])==1:
    #     im_with_bb = draw_box(image, boxes)
    # else:
    #     im_with_bb =image
    #     for i in range(len(boxes)):
    #         im_with_bb = draw_box(im_with_bb, boxes[i])
    return im_with_bb

def display_result(image, boxes):
    if isinstance(boxes,list):
        im_with_bb =image
        for i in range(len(boxes)):
            im_with_bb = draw_box(im_with_bb, boxes[i])
    else:
        im_with_bb = draw_box(image, boxes)
    cv2.imshow("result", im_with_bb)
    cv2.waitKey(1)

    return im_with_bb
