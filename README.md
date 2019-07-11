SP-mapping
==========

Spherical picture mapping.




Requirments
-----------
* python3
* numpy
* scipy
* opencv-python

Equipment
---------
* Multi-Projector
	- 仮想的に単一の統合モニタとして扱えること。
	- 例えば，Matrox TripleHead2Go や NVIDIA MOSAIC などのアプリケーションを利用する。
* RICOH THETA S
	- プロジェクタ投影点位置の測定に利用する。
	- API 利用のため無線 LAN 接続をしておく。
	- API v2.1 に対応するファームウェア v01.62 以降。
	- THETA V や Z1 でも動作するかも。




Making mapping-table
--------------------

### 1. multi-projector configration
* スクリーンのパーティション設定は，プロジェクタ台数分の数のpngファイルでそれぞれ表現する
* pngの画像の仕様
	- アスペクト比は，統合モニタ全体のサイズと等しくする。
	- プロジェクタのパートは，緑レイヤーに0以外の画素値の長方形領域で表現する。
	- それ以外の画素値は0にする。
	- ファイル名は`projector_#.png`とする。#にはプロジェクタの番号が入る。番号は1からナンバリングする。
* 各pngファイルのアスペクト比は等しいこと。
* パート同士はオーバーラップできない。


### 2. screen configration


### 3. Wireless LAN connection with THETA


### 4. 


Mapping spherical image
-----------------------


