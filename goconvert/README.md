SP-mapping/GoConvert
====================

マッピング処理の Go 版



### 使い方

```
$ goconvert -i input.mp4 -d map_xxxx output.mp4
```

### 標準出力モード

出力名を`-`とすると rawvideo ストリームを標準出力に書き込む。
ffmpeg にパイプ入力する用途を想定している。

```
$ goconvert -i path2input.mp4 -d path2workfolder - | ffmpeg -f rawvideo -pix_fmt bgr24 -s 7680x1080 -framerate 60 -i - -c libx264 -preset veryfast -pix_fmt yuv420p output.mp4 
```

ffmpeg のオプションが長くて忘れるので，`--template-ffmpeg`で雛形を提示するようにしている。画像サイズやフレームレートが反映されている。

```
$ goconvert -i input.mp4 -d map_xxxxxx --template-ffmpeg
ffmpeg template:
goconvert -i input.mp4 -d map_xxxxxx - | ffmpeg -f rawvideo -pix_fmt bgr24 -s 7680x1080 -framerate 60 -i - output.mp4
```


