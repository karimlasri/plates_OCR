import cv2
import numpy as np
import math
import os
#
# def find_rect_corners(contour):
#     cont_list = []
#     for i in range(len(contour[0])):
#         cont_list.append(tuple(list(contour[0][i][0])))
#     print(cont_list)
#     segments = []
#     for i in range(len(cont_list)):
#     x_tl, y_tl = cont_list[0]
#     return(find_lines(cont_list))
def distance(p1, p2):
    return(math.pow(math.pow(p2[0]-p1[0], 2) + math.pow(p2[1]-p1[1], 2), 0.5))

def find_closest(p, points):
    closest = points[0]
    min_dist = distance(p, closest)
    for point in points:
        dist = distance(p, point)
        if dist < min_dist:
            closest = point
            min_dist = dist
    return(closest)

def find_quadrilateral_corners(contour, img):
    orig_tl = (0, 0)
    orig_tr = (img.shape[1], 0)
    orig_br = (img.shape[1], img.shape[0])
    orig_bl = (0, img.shape[0])
    orig_corners = orig_tl, orig_tr, orig_br, orig_bl
    cont_list = []
    for i in range(len(contour[0])):
        cont_list.append(tuple(list(contour[0][i][0])))
    quad_corners = []
    for i in range(len(orig_corners)):
        quad_corners.append(find_closest(orig_corners[i], cont_list))
    return(quad_corners)

def intersection(rect0, rect1):
    top = max(rect0[1], rect1[1])
    bottom = min(rect0[1]+rect0[3], rect1[1]+rect1[3])
    left = max(rect0[0], rect1[0])
    right = min(rect0[0]+rect0[2], rect1[0]+rect1[2])
    if top < bottom and left < right:
        return((bottom-top)*(right-left))
    else:
        return(0)

def area(rect):
    return(rect[2]*rect[3])

def merge_letter_rects(rect0, rect1, width_thresh):
    """ Merges rectangles that don't overlap but represent the same letter """
    left = max(rect0[0], rect1[0])
    right = min(rect0[0]+rect0[2], rect1[0]+rect1[2])
    inter_width = max(0, right - left)
    if inter_width > rect0[2]*width_thresh or inter_width > rect1[2]*width_thresh:
        mx = min(rect0[0], rect1[0])
        my = min(rect0[1], rect1[1])
        mw = max(rect0[0]+rect0[2], rect1[0]+rect1[2])-mx
        mh = max(rect0[1] + rect0[3], rect1[1] + rect1[3]) - my
        return(True, (mx, my, mw, mh))
    else:
        return(False, (0, 0, 0, 0))


def process_contour_boxes(rects, inter_thresh, width_thresh, height_thresh, img):
    # Removing a rectangle that would contain all others
    areas = [area(rect) for rect in rects]
    delete = [False for i in range(len(rects))]
    biggest = max(zip([i for i in range(len(rects))], areas), key= lambda x : x[1])
    if biggest[1] > img.shape[0]*img.shape[1]*0.3:
        delete[biggest[0]] = True
    # Dealing with rectangles that overlap or belong to the same letter
    merge = [(False, rect) for rect in rects]
    for i, rect0 in enumerate(rects[:len(rects)-1]):
            area0 = areas[i]
            for k, rect1 in enumerate(rects[i+1:]):
                j = i+k+1
                if not delete[i] and not delete[j]:
                    area1 = areas[j]
                    inter = intersection(rect0, rect1)
                    if inter > area1*inter_thresh:
                        delete[j] = True
                    elif inter > area0*inter_thresh:
                        delete[i] = True
                    else:
                        same_letter, merged = merge_letter_rects(rect0, rect1, width_thresh)
                        if same_letter:
                            merge[i] = (True, merged)
                            delete[j] = True
    # Replacing rects that have been merged
    for i in range(len(rects)):
        if merge[i][0]:
            rects[i] = merge[i][1]
    # Removing rectangles that should be removed
    merged_rects_wo_overlap = [rect for i, rect in enumerate(rects) if not delete[i]]

    # Readjusting rects that are too small
    max_height = max([rect[3] for rect in merged_rects_wo_overlap])
    final_rects = [rect for i, rect in enumerate(merged_rects_wo_overlap) if rect[3] > max_height*height_thresh]
    return(final_rects)


# def find_lines(points_list):
#     lines = []
#     if len(points_list) == 0:
#         return lines
#     else:
#         curline = [points_list[0], points_list[0]]
#         # for i, point in enumerate(points_list[1:]):
#         #     if abs(curline[1][0]-curline[0][0]) <= 1 and abs(curline[1][1]-curline[0][1]) <= 1:
#         #         if ((curline[1][0]-curline[0][0])/(max(1, point[0]-curline[0][0])) >= 0) and ((curline[1][1]-curline[0][1])/(max(1, point[1]-curline[0][1])) >= 0):
#         #             curline[1] = point
#         #         else:
#         #             lines.append(curline)
#         #             curline = [point, point]
#         #     elif curline[1][0]-curline[0][0] == 0:
#         #         if curline[1][1]-curline[0][1] == 0:
#         #             curline[1] = point
#         #         else:
#         #             cur_ratio = (point[0] - curline[0][0])/(point[1]-curline[0][1])
#         #             if abs(cur_ratio) < 0.2:
#         #                 curline[1] = point
#         #             else:
#         #                 lines.append(curline)
#         #                 curline = [point, point]
#         #     else:
#         #         if curline[1][1]-curline[0][1] == 0:
#         #             cur_ratio = (point[1] - curline[0][1]) / (point[0] - curline[0][0])
#         #             if abs(cur_ratio) < 0.2:
#         #                 curline[1] = point
#         #             else:
#         #                 lines.append(curline)
#         #                 curline = [point, point]
#         #         else:
#         #             best_ratio = (curline[1][0] - curline[0][0]) / (curline[1][1] - curline[0][1])
#         #             cur_ratio = (point[0] - curline[0][0]) / (point[1] - curline[0][1])
#         #             if abs(best_ratio - cur_ratio) / best_ratio < 0.2:
#         #                 curline[1] = point
#         #             else:
#         #                 lines.append(curline)
#         #                 curline = [point, point]
#         curline = [points_list[0], points_list[1]]
#         for i in range(len(points_list[2:])//2):
#             cur_angle = 0
#             new_angle = 0
#             point0 = points_list[i]
#             point1 = points_list[i+1]
#             if curline[1][0] - curline[0][0] == 0:
#                 if abs(point1[0] - point0[0]) <= 1:
#                     curline[1] = point1
#                 else:
#                     lines.append(curline)
#                     curline = [point0, point1]
#             elif curline[1][1] - curline[0][1] == 0:
#                 if abs(point1[1] - point0[1]) <= 1:
#                     curline[1] = point1
#                 else:
#                     lines.append(curline)
#                     curline = [point0, point1]
#             else:
#                 cur_tan = (curline[1][1] - curline[0][1]) / (curline[1][0] - curline[0][0])
#                 cur_atan = math.atan(cur_tan)
#                 if curline[1][0] - curline[0][0] > 0:
#                     cur_angle = cur_atan
#                 else:
#                     cur_angle = math.pi - cur_atan
#                 if point1[0] - point0[0] == 0:
#                     if point1[1] - point0[1] < 0:
#                         new_angle = math.pi*3/2
#                     else:
#                         new_angle = math.pi * 1/2
#                 else:
#                     new_tan = (point1[1] - point0[1])/(point1[0] - point0[0])
#
#                     new_atan = math.asin(new_tan)
#
#                     if point1[0] - point0[0] > 0:
#                         new_angle = new_atan
#                     else:
#                         new_angle = math.pi - new_atan
#                 if abs(cur_angle-new_angle) < 0.8*(math.pi/4):
#                     curline[1] = point1
#                 else:
#                     lines.append(curline)
#                     curline = [point1, point0]
#         lines.append(curline)
#         return(lines)



def center_plate(img, show = False):
    # Applying binary threshold to image
    ret, thres_img = cv2.threshold(img,200,255,cv2.THRESH_BINARY)
    (_, cnts, _) = cv2.findContours(thres_img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    rect_conts = cnts
    cv2.drawContours(img, rect_conts[:1], -1, (0, 255, 0), 3)

    # Finding corners of quadrilateral
    corners = find_quadrilateral_corners(rect_conts, img)
    if show:
        copy_img = img.copy()
        for c in corners:
            cv2.circle(copy_img, c, 3, (255, 255, 255), 3)
        cv2.imshow("Corners", copy_img)

    # Centering image
    dest_corners = [(0, 0), (199, 0), (199, 39), (0, 39)]
    warp_mat = cv2.getPerspectiveTransform(np.array(corners, np.float32), np.array(dest_corners, np.float32))
    centered_img = np.zeros((199, 39), dtype=np.float32)
    centered_img = cv2.warpPerspective(thres_img, warp_mat, centered_img.shape, centered_img, cv2.INTER_AREA + cv2.WARP_FILL_OUTLIERS, cv2.BORDER_REPLICATE, (255, 255, 255))

    # Adding borders to image so it can be closed later
    border_img = cv2.copyMakeBorder(centered_img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    if show:
        cv2.imshow("Added borders", border_img)

    return(border_img)

# def center_plate(img):
#     edges = cv2.Canny(img, 50, 150, apertureSize=3)
#     cv2.imshow('yo', edges)
#     cv2.waitKey()
#     print(edges)
#     lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
#     print(lines)
#     for rho, theta in lines[0]:
#         a = np.cos(theta)
#         b = np.sin(theta)
#         x0 = a * rho
#         y0 = b * rho
#         x1 = int(x0 + 1000 * (-b))
#         y1 = int(y0 + 1000 * (a))
#         x2 = int(x0 - 1000 * (-b))
#         y2 = int(y0 - 1000 * (a))
#
#         cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
#     cv2.imshow('yo', img)
#     cv2.waitKey()
#     # loop over our contours
#     return (img)

def crop_characters(plate, show=False):
    """ Detecting characters in plate returning separate images containing one character each """

    # Applying binary threshold to image
    thresh = 200
    ret, img = cv2.threshold(plate, thresh, 255, cv2.THRESH_BINARY)
    if show:
        cv2.imshow('Binary image', img)
        cv2.waitKey()

    # Closing the image to remove noise
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    if show:
        cv2.imshow('Closed image', img)
        cv2.waitKey()

    # Detecting contours
    _, contours, _ = cv2.findContours(img, 1, cv2.CHAIN_APPROX_NONE)
    # Finding bounding boxes of all contours
    col = 0
    rects = []
    for cnt in contours:
        rect = cv2.boundingRect(cnt)
        rects.append(rect)
    # Processing boxes to keep intersting ones only
    processed_rects = process_contour_boxes(rects, 0.8, 0.5, 0.7, img)

    if show:
        # Drawing base rectangles
        img_base_rects = img.copy()
        for x, y, w, h in rects:
            img_base_rects = cv2.rectangle(img_base_rects, (x, y), (x + w, y + h), (col, col, col), 2)
        cv2.imshow('Base rectangles', img_base_rects)
        cv2.waitKey()
        # Drawing processed rectangles
        img_proc = img.copy()
        for x, y, w, h in processed_rects:
            img_proc = cv2.rectangle(img_proc, (x, y), (x + w, y + h), (col, col, col), 2)
        cv2.imshow('Processed rectangles', img_proc)
        cv2.waitKey()

    # Order rectangles from left to right so they correspond to letters of label file
    processed_rects.sort(key=lambda x: x[0])

    # Returning crops
    crops = []
    for x, y, w, h in processed_rects:
        crops.append(img[y:y+h, x:x+w])
    return(crops)


def detect_characters(img_path, dest_path, show=False):
    # Getting letters from raw file name
    img_filename = img_path.split('/')[-1]
    img_filename = img_filename.split('\\')[-1]
    img_raw_name = img_filename.split('.')[0]
    letters = list(img_raw_name)

    # Centering image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    centered_plate = center_plate(img, show)
    if show:
        cv2.imshow('Centered', centered_plate)
        cv2.waitKey()

    # Cropping characters in centered plate
    letter_crops = crop_characters(centered_plate, show)
    if show :
        for letter in letter_crops:
            cv2.imshow('letter', letter)
            cv2.waitKey()

    # If we found as many letters as we wanted, we save the labeled images
    if len(letters) == len(letter_crops):
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        for i, letter in enumerate(letter_crops):
            cv2.imwrite(dest_path + '/' + letters[i] + '.jpg', letter)