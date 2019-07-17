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
* 複数のプロジェクタが，仮想敵に単一モニタとして統合されているとする。各プロジェクタがそれぞれ統合モニタ上のどの区画に対応しているか，プロジェクタ台数分の数の png ファイルでそれぞれ表現する。
* png 画像の仕様：
	- アスペクト比は，統合モニタ全体のサイズと等しくする。
	- プロジェクタの区画は，緑レイヤーに0以外の画素値の長方形領域で表現する。
	- それ以外の画素値は0にする。
	- ファイル名は`projector_#.png`とする。# にはプロジェクタの番号が入る。番号は1からナンバリングする。この番号をプロジェクタIDと呼ぶことにする。
* 各 png ファイルのアスペクト比は等しいこと。
* 区画はオーバーラップできない。


### 2. screen configration
* 各プロジェクタが担当する投影領域に関して，プロジェクタ台数分の png ファイルで設定する。
* png 画像の仕様：
	- 正距円筒座標形式
	- 緑レイヤー：投影領域。正距円筒図法画像上に0以上の画素値の長方形領域で表す。真上や真下をまたがる領域は設定できない。左右の境界をまたがる領域は設定できる。
	- 青レイヤー：他のプロジェクタとのオーバラップ領域。正距円筒図法画像上で長方形で表す。
	- 赤レイヤー：マスク領域。投影領域のうち，プロジェクタ投影点測定時に測定データを無効にする領域。画素値を0以上にすると，その点の計測データは無効になり，周囲のデータから推定される。カメラ三脚の影など，投影領域のうち事前に測定誤差になる箇所を塗りつぶす。
	- ファイル名は`screen_#.png`とする。# には対応するプロジェクタIDが入る。
* png ファイルはプロジェクタの台数と等しいこと。



### 3. Wireless LAN connection with THETA


### 4. 


Mapping spherical image
-----------------------


