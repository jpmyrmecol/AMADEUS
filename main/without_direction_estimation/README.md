# Without direction estimation

Easy / Advanced の `Without direction estimation` をオンにするとこの処理系を使います。
初期値はオフです。YAML のキーは `WITHOUT_DIRECTION_ESTIMATION: true` です。

```sh
python main/batch.py /path/to/session/config.yaml
```

個別の工程も `python main/without_direction_estimation/<stage>.py config.yaml`
で実行できます。学習の監視プロセス、自動的な学習画像補充、refine の反復から
起動する子プロセスも同じ処理系を使います。

## 処理の契約

- セグメンテーション済みの個体を、静止、短い軌跡、移動方向の不確定、
  OBB の縦横比だけを理由に除外しません。重なり、入力の outlier 指定、
  非有効形状などの元の品質判定は維持します。
- YOLO OBB のクラスは `0: animal` の1種類です。学習時の画像回転でも変わりません。
  検出に方向クラスのモデルを渡した場合は明示的にエラーにします。
- OBB の頂点と回転は形状として保持します。前後の極性、頭尾、進行方向は推定しません。
  refine は信頼度と空間的重なりを使い、方向差では削除しません。
- 固定個体数・可変個体数とも、ID 対応付けは位置、OBB の重なり、
  距離と既存の欠測処理を使います。方向コスト・方向ゲート・方向 KF は使いません。
  補正の角度判定と方向反転補正も無効です。
- embedding の入力画像は **OBB の長軸を水平に配置**します。角度は180度を法とする
  幾何学的な軸です。正向きと180度反転画像のネットワーク出力を平均し、
  学習・推論の両方で頭尾の選択に依存しない特徴にします。
  ネットワークに渡す画像数は2倍になります。
  fragment の選別に方向の安定性を使いません。連続4フレーム以上の孤立観測という
  embedding のサンプリング条件は維持し、それ未満の個体を追跡CSVから消しません。
- OBB の三角形・方向矢印は描画しません。
- 最終結果CSVは `frame, cx0, cy0, w0, h0, ...` です。`heading*` 列はありません。
  OBB の頂点は中間CSVに残ります。

## 出力と互換性

出力先は元セッションの `without_direction_estimation/` 以下です。
設定、データセット、学習済み重み、追跡、補正、embedding キャッシュを通常モードと
分けます。元の動画と segmentation pickle を参照し、背景画像と segmentation 設定を
コピーします。元の設定ファイルは書き換えません。
初回に通常モード用の独自 `CREATE_DATASET_SOURCE_DIRS` は引き継ぎません。
同キーを空にして新モードのデータセットを生成してください。
工程の skip 設定は引き続き有効です。再利用する成果物は新モードの出力先に必要です。

共通コードを再利用するため、private な donor manifest の `direction_vec` 相当の列と
`head_angle_deg` は幾何学的な長軸の格納に限って使います。`direction_valid` は0です。
これらは生物学的な方向の測定値ではありません。private な
`directions_tracked*.csv` は既存viewerのファイル探索用にフレーム索引だけを残し、
方向値の列は保存しません。旧形式の補正用pickle内の方向値は欠測です。
旧来の方向・最小縦横比の設定値は、このモードの方向判定には使いません。

## スクリプト構成

通常版と同名のスクリプトが、それぞれの処理を直接実装します。
`runtime.py` による関数の実行時差し替えはありません。
幾何処理、donor出力、embedding は対応する通常名のファイルに統合しています。
可変個体数では通常版と同じ `multi_staged_association_variable.py`、`refinement_variable.py`、
`identity_correction_variable.py` を使います。batch が `VARIABLE_NUM_OBJECTS` で
固定個体数用・可変個体数用の実行スクリプトを選びます。
`_variable` 側は共通関数を無印版から import し、可変個体数固有の処理だけを実装します。
共有関数内部の分岐は、必要な関数を引数で渡して選択します。
通常版と方向推定なし版は同一プロセスでインポートしても関数を上書きしません。

## 候補画像数と補充

初回の生成枚数と明示的な `PASTE_BLOBS_NUM_FRAMES`／`CLUSTER_FRAMES` の優先順位は
通常版と同じです。`create_dataset.py` も通常版と同じreserve計算と不足数推定を使います。
例えば `NUM_IMAGES=10000`、reserve 2%（最小100）、`CLUSTERED_RATIO=0.05` では、
データセット作成時に候補総数10200、non-clustered 9690／clustered 510を目標にします。
clustered が500枚のままであれば、non-clustered を9700枚以上にするよう補充します。
補充のために増やした生成枚数は、初回目標へ戻しません。

batchで貼り付け工程をskipしていても、既存の貼り付け画像が不足する場合は
データセット作成が補充します。設定ファイル自体のskip値は変更しません。
通常版の単一個体用 `single_animal_images` 直接利用経路は維持します。
