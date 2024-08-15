import cv2
import os
 
# 视频路径 输出路径
def video_to_frames(video, path_output_dir):
    # extract frames from a video and save to directory as 'x.png' where
    # x is the frame index
    # 打开摄像头 参数为输入流，可以为摄像头或视频文件
    vidcap = cv2.VideoCapture(video)
    count = 0
    # 视频的帧率
    timeF = 24
    while vidcap.isOpened():
        success, image = vidcap.read()
        if success:
            # 24帧保存一次
            if count % timeF == 0:
                cv2.imwrite(os.path.join(path_output_dir, '%d.png') % count, image)
            count += 1
        else:
            break
    cv2.destroyAllWindows()
    vidcap.release()
 
 
video_to_frames('video_traffic_light.mp4','video_captures')