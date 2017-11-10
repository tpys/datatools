import cv2,math
def _112x96_mc40(input_image, points, output_size = (96, 112), ec_mc_y = 40):
    eye_center = ((points[0][0] + points[1][0]) / 2, (points[0][1] + points[1][1]) / 2)
    mouth_center = ((points[3][0] + points[4][0]) / 2, (points[3][1] + points[4][1]) / 2)
    angle = math.atan2(mouth_center[0] - eye_center[0], mouth_center[1] - eye_center[1]) / math.pi * -180.0
    # angle = math.atan2(points[1][1] - points[0][1], points[1][0] - points[0][0]) / math.pi * 180.0
    scale = ec_mc_y / math.sqrt((mouth_center[0] - eye_center[0])**2 + (mouth_center[1] - eye_center[1])**2)
    center = ((points[0][0] + points[1][0] + points[3][0] + points[4][0]) / 4, (points[0][1] + points[1][1] + points[3][1] + points[4][1]) / 4)
    rot_mat = cv2.getRotationMatrix2D(center, angle, scale)
    rot_mat[0][2] -= (center[0] - output_size[0] / 2)
    rot_mat[1][2] -= (center[1] - output_size[1] / 2)
    warp_dst = cv2.warpAffine(input_image, rot_mat, output_size)
    return warp_dst
