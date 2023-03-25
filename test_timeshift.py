import cv2
import numpy as np

# 加载视频文件
cap = cv2.VideoCapture('video.mp4')

# 提取目标通道数据
channel1 = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    channel1.append(frame[:, :, 0])

# 进行时间位移操作
time_shift = 2  # 2秒钟
fps = cap.get(cv2.CAP_PROP_FPS)
time_diff = int(fps * time_shift)
channel1_shifted = np.roll(channel1, time_diff)

# 合并通道数据
new_frames = []
for i in range(len(channel1)):
    new_frame = np.dstack((channel1_shifted[i], frame[:, :, 1:], ))
    new_frames.append(new_frame)

# 保存新的视频文件
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('new_video.mp4', fourcc, fps, (640, 480))
for frame in new_frames:
    out.write(frame)
out.release()

# 释放资源
cap.release()
cv2.destroyAllWindows()
