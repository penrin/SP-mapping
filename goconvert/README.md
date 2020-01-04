SP-mapping/GoConvert
====================

マッピング処理の Go 版。
処理速度向上を目指す。

Requirements
------------

 * OpenCV 4.2.0
 * Go


Install
-------

#### macOS

```
brew install opencv
brew install pkgconfig
```

#### Linux

#### Windows


Usage
-----

#### 通常

```
$ goconvert -i input.mp4 -d map_xxxx -o output.mp4
```

#### ffmpeg にパイプする

出力名を`-`とすると rawvideo ストリームを標準出力に書き込む。
このffmpeg にパイプ入力する用途を想定している。
ffmpeg 側で詳細なエンコード設定ができるほか，NVENC(Windows/Linux)を利用すればCPUリソースをgoconvert側に割けるので処理速度向上が期待できる。

例：

```
$ goconvert -i path2input.mp4 -d map_xxxx -o - | ffmpeg -f rawvideo -pix_fmt bgr24 -s 7680x1080 -framerate 60 -i - -c libx264 -preset veryfast -pix_fmt yuv420p output.mp4 
```

オプション`--ffmpeg-template`により ffmpeg のオプションの雛形を提示する。画像サイズやフレームレートが反映されている。

```
$ goconvert -i input.mp4 -d map_xxxx --ffmpeg-template
ffmpeg template:
goconvert -i input.mp4 -d map_xxxx -o - | ffmpeg -f rawvideo -pix_fmt bgr24 -s 7680x1080 -framerate 60 -i - output.mp4
```


### Tasks to be done

* implemet: Edge-blur, Overlap
* speed up: BilinearInterpolation





