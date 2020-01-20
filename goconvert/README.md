SP-mapping/GoConvert
====================

マッピング処理の Go 版。
処理速度向上を目指す。

Requirements
------------

 * OpenCV 4
 * Go


Install
-------

I haven't made Makefile yet.

#### macOS



#### Linux

Install dependent packages

```
$ go get -u -d gocv.io/x/gocv
$ go get -u github.com/penrin/gonpy
$ go get -u github.com/akamensky/argparse
```

Build

```
$ git clone -b develop_go_converter https://github.com/penrin/SP-mapping path2install/SP-mapping
$ go build -o goconvert path2install/SP-mapping/*go
```



#### Windows


#### OpenCV 4
see: https://gocv.io/getting-started/



Usage
-----

#### 通常

```
$ goconvert -i input.mp4 -d map_xxxx -o output.mp4
```

#### ffmpeg にパイプする場合

`-o -` をつけると rawvideo ストリームが標準出力に書き込まれ，ffmpeg にパイプ入力することができる。
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


Tasks to be done
----------------

* optimize: GammaCorrect, EdgeBlur, Overlap
	- no copy, do mutex
	- skip if edgeblur is zero
	- skip if overlap is zero
* version management
* use os.Stderr for error messages
* add appropriate context to error messages
* make test file





