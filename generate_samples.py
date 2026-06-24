import cv2
import numpy as np
import scipy.io.wavfile as wavfile

# Generate a 2-second dummy video
width, height = 640, 480
fps = 30
duration = 2
out = cv2.VideoWriter('sample_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
for i in range(fps * duration):
    # create a moving circle to simulate some motion
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.circle(frame, (int(width/2 + 50*np.sin(i*0.1)), int(height/2)), 50, (255, 255, 255), -1)
    out.write(frame)
out.release()
print("Generated sample_video.mp4")

# Generate a 2-second dummy audio
sample_rate = 44100
t = np.linspace(0, duration, sample_rate * duration, False)
audio_data = np.sin(2 * np.pi * 440 * t) * 0.5  # 440 Hz sine wave
wavfile.write('sample_audio.wav', sample_rate, (audio_data * 32767).astype(np.int16))
print("Generated sample_audio.wav")
