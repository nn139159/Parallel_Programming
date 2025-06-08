下面指令主要負責兩個task:
1. 把選定的資料夾image轉換成input video
2. ffmpeg把input video傳到pipe => main讀取 => 輸出影像

# 1.把圖像轉換成input video
## 下面指令就是把s1資料夾轉換成video
fmpeg -framerate 1 -i s1_%d.pgm -c:v libx264 -r 10 -pix_fmt yuv420p s1.mp4

# 2.ffmpeg把input video傳到pipe => main讀取 => 輸出影像(有兩種做法)
## 2.1. 有bug(有機率失敗)
ffmpeg -i s1.mp4 -f rawvideo -pix_fmt rgb24 - | stdbuf -oL ./main | ffmpeg -y -f rawvideo -pix_fmt rgb24 -s 92x112 -i - -c:v libx264 output.mp4

## 2.2. work
### 說明: main輸出到dump.raw, 用ffmpeg把dump.raw轉成video
ffmpeg -i s1.mp4 -f rawvideo -pix_fmt rgb24 - | stdbuf -oL ./main | tee dump.raw | wc -c
ffmpeg -y -f rawvideo -pix_fmt rgb24 -s 92x112 -i dump.raw -c:v libx264 preview.mp4