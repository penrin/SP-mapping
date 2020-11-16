SP-mapping/GoConvert
====================

逆歪み付加処理(SP-mapping/convert)の Go 実装版

Requirements
------------

 * Go
 * OpenCV 4 (see [GoCV Getting Started](https://gocv.io/getting-started/))


Install
-------

Install dependent packages

```
$ go get -u -d gocv.io/x/gocv
$ go get -u github.com/penrin/gonpy
$ go get -u github.com/akamensky/argparse
```

Build

```
$ git clone -b develop_go_converter https://github.com/penrin/SP-mapping path2install/SP-mapping
$ go build -o goconvert path2install/SP-mapping/*.go
```


Usage
-----


### Help

```
$ goconvert -h
```

### Base

SP-mapping/makemap で作成した対応点データ `mapping_table.npz` のパスと，変換したいエクイレクタングラー形式の動画または静止画ファイルを指定する。

```
$ goconvert -d map_xxxxxx -i input.mp4
```




### Pipe to FFMPEG (動画のみ)

`-o -` をつけると rawvideo ストリームが標準出力に書き込まれ，ffmpeg にパイプ入力することができる。ffmpeg 側で詳細なエンコード設定が可能。このとき NVENC(Windows/Linux) や VideoToolbox(macOS) などのハードウェアエンコードすると，CPU リソースを goconvert 側に割けるので処理速度向上が期待できる。


オプション`--ffmpeg-template`または`-f`により ffmpeg のオプションの雛形を提示する。雛形には画像サイズやフレームレートが反映されている。

```
$ goconvert -i input.mp4 -d map_xxxx --ffmpeg-template 
./goconvert -i input.mp4 -d map_xxxx -o - | ffmpeg -f rawvideo -pix_fmt bgr24 -s 3840x2160 -framerate 60 -i - -c:v h264_videotoolbox -b:v 60M output.mp4
```




Tasks to be done
----------------

* version management
* use os.Stderr for error messages
* add appropriate context to error messages.
* make test file
* Makefile





 