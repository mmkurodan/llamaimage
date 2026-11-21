# llamaimage — SD 画像生成 PoC（Android / llama.cpp 系 ggml 基盤）

`/root/llama`（**LLM AI Server**: 端末上で llama.cpp を JNI で動かし、対話 UI と Ollama 互換 API を
提供するアプリ）の**姉妹 PoC** です。将来 LLM AI Server に画像生成機能として組込むことを前提に、
同じ「JNI ネイティブ + シングルトン Manager + 最小 UI」構成を踏襲した **txt2img / img2img の実装**です。

> **技術メモ**: SD の GGUF 推論は llama.cpp 本体ではなく、**同じ ggml 基盤の
> [`stable-diffusion.cpp`](https://github.com/leejet/stable-diffusion.cpp)** が担います。本 PoC は
> これを JNI で組込みます。バックエンドは **CPU のみ / arm64-v8a**。

---

## 1. モデル（アプリ内カタログ・初回のみダウンロード）

ダウンロード先 URL・サイズ・SHA-256 は **アプリに内蔵**（`ModelCatalog.java`）。上部の **「モデル」** から選ぶと、
初回だけ Android の DownloadManager でダウンロードし（アプリを閉じても継続・通知に進捗表示）、
**サイズと SHA-256 を検証してから**使います。2 回目以降は端末内のファイルを読み込むだけです。

| モデル | ファイル (Hugging Face) | サイズ | 選択時に適用される推奨設定 |
|---|---|---|---|
| **DreamShaper 8 LCM（推奨・既定）** | `haven-ai-companion/dreamshaper8-lcm-gguf` `DreamShaper8_LCM_q8_0.gguf` | 1.80 GB | LCM / 6 step / CFG 1.0 |
| SD-Turbo（最速） | `Green-Sky/SD-Turbo-GGUF` `sd_turbo-f16-q8_0.gguf` | 2.02 GB | Euler a / 1 step / CFG 1.0 |
| Stable Diffusion 1.5（標準） | `second-state/stable-diffusion-v1-5-GGUF` `…-Q4_0.gguf` | 1.57 GB | DPM++ 2M / 20 step / CFG 7.0 |
| TAESD（高速 VAE、既定 ON） | `madebyollin/taesd` `diffusion_pytorch_model.safetensors` | 9.8 MB | — |

- 保存先: `Android/data/com.micklab.llamaimage/files/models/`（ストレージ権限不要、アンインストールで削除）。
  ここに同名ファイルを `adb push` しておけば、初回使用時に SHA-256 を検証してそのまま使います。
- モデルの削除・切替もカード上のボタンから。推奨設定は「生成設定」→「このモデルの推奨値に戻す」で再適用できます。
- 旧バージョンでファイル選択からコピーした `files/model.gguf` / `files/taesd.bin` は起動時に引き継ぎます
  （SD1.5 Q4_0 など一致するカタログ項目があればそれとして扱い、再ダウンロード不要）。

## 2. 性能（Snapdragon 8 Elite 実測, 512px, DreamShaper 8 LCM 6 step）

| 構成 | サンプリング | VAE デコード | 合計 |
|---|---|---|---|
| 旧ビルド（armv8-a, 実質 4 スレッド） | 86.8 s | 56.5 s | 143.5 s |
| 旧既定の VAE タイル処理 | — | 約 2.2 倍 | — |
| **fp16+dotprod ビルド, 8 スレッド** | 50.5 s | 27.5 s | 78.5 s |
| **＋ TAESD（新既定）** | 51.7 s | 1.9 s | **53.8 s** |

- ネイティブライブラリは 2 種類を同梱し、起動時に CPU 機能（`/proc/cpuinfo` の asimdhp/asimddp）で選択:
  `libsd_jni_fp16.so`（armv8.2-a+fp16+dotprod）／ `libsd_jni.so`（armv8-a、古い CPU 用）。
  この ggml は SIMD カーネルをコンパイル時に選ぶため、NDK 既定の armv8-a では F16 の conv/matmul が遅い経路になります。
  i8mm 版と llamafile(tinyBLAS) も試しましたが改善なし（llamafile は低速化）のため不採用。
- 自動スレッド数 = 性能コア数（LITTLE コアを除外）。sd.cpp 自身のコア数検出は sysfs パスの不具合で半数（4）になっていました。
- VAE タイル処理は RAM 6GB 未満の端末でのみ既定 ON（「生成設定」で切替）。
- 素の SD1.5（20 step / CFG 7）は UNet 評価が LCM の約 7 倍なので数分かかります。

## 3. 使い方

1. 初回起動でモデル選択が開く → **「ダウンロードして使う」**（Wi-Fi 推奨）。完了・検証後に自動で読み込み。
2. プロンプト（日本語可）・ステップ数・サイズ・シード（-1=ランダム）を入力 → **「生成」**。
   進捗は「生成中 3/6（残り約 xx s）」→「仕上げ（VAE デコード）中」の 2 段階で表示。**「中止」**で次のステップ境界で停止。
3. img2img: 「入力画像」→「画像から生成」（変化量 0.05〜1。中央を正方形に切り抜き、EXIF の向きも補正）。
4. **「画像を保存」**でギャラリー（`Pictures/llamaimage`、ファイル名にシード入り）へ PNG 保存。画像の長押しでそのシードを固定。
5. 日本語プロンプトは **「LLM設定」** の Ollama 互換 API（LLM AI Server）で英訳してから生成（英語のみのプロンプトは LLM を呼ばない）。
   CFG>1 のときはネガティブプロンプトも英訳。

## 4. ビルド

ネイティブ依存（stable-diffusion.cpp + ggml）は**リポジトリに同梱せず**、固定コミットを
ビルド時に取得します（`scripts/fetch-sdcpp.sh`、`.gitignore` 済み）。取得後に `scripts/patch-sdcpp.sh` が
進捗フェーズ・プレビュー・中止用のフックを当てます（冪等・アンカー一致を検証）。

### CI（推奨）
`main` への push で `.github/workflows/android.yml` が動作し、NDK 27.2.12479018 + CMake を導入 →
`fetch-sdcpp.sh` → `assembleDebug` / `assembleRelease` / `bundleRelease`。同じ CMakeLists を
`:app`（armv8-a）と `:sdjni-fp16`（fp16）の 2 モジュールでビルドし、両 `.so` が APK に入ったことを検証します。

### ローカル（この proot 環境）
NDK のリンカがエミュレーション下で動かないため、**CI 製の `.so` を取り込んでローカルで APK 化**します:
```bash
./build-local.sh --release            # 最新の成功 CI から .so を取得 → -PsdPrebuiltNative=true で packaging
```
CI の `.so` のコミットとローカルのネイティブコードが食い違う場合（JNI 不一致で起動失敗するため）は停止します。
先に push して CI の完了を待ってください。

## 5. 構成

| パス | 役割 |
|------|------|
| `app/src/main/java/.../MainActivity.java` | UI（モデル選択・ダウンロード進捗・生成・中止・保存） |
| `app/src/main/java/.../ModelCatalog.java` | 内蔵モデル一覧（URL / サイズ / SHA-256 / 推奨設定） |
| `app/src/main/java/.../ModelStore.java` | DownloadManager による初回ダウンロード・検証・削除・旧ファイル引継ぎ |
| `app/src/main/java/.../SdModelManager.java` | 単一ネイティブコンテキストの所有（同一設定なら再読込しない）・中止・結果 |
| `app/src/main/java/.../StableDiffusionNative.java` | JNI インターフェース（CPU に合う `.so` を選択してロード） |
| `app/src/main/java/.../CpuInfo.java` | CPU 機能・性能コア数・RAM の判定 |
| `app/src/main/java/.../PromptTranslator.java` | LLM による英訳（キャッシュ付き）・NSFW 判定 |
| `app/src/main/cpp/jni/jni_sd.cpp` | sd.cpp を叩く JNI 実装（進捗フェーズ・プレビュー・中止） |
| `app/src/main/cpp/CMakeLists.txt` | CPU 専用・arm64・16KB ページアライン。`SD_CPU_VARIANT` で armv8-a / fp16 を切替 |
| `sdjni-fp16/` | 同じ CMakeLists を fp16 版としてビルドするネイティブ専用モジュール |
| `scripts/fetch-sdcpp.sh` / `patch-sdcpp.sh` | 固定コミットの取得 / 独自フックの適用 |

**pin バージョン**: stable-diffusion.cpp `9578fdcc`（2024-11-30, フラットな `txt2img` API・単一 CLIP
vocab の軽量版）/ ggml `6fcbd60`（上記コミットの gitlink）。

## 6. 将来の LLM AI Server 組込み（設計のみ）

`SdModelManager` を、親アプリの `OllamaForegroundService` / `OllamaApiServer` 相当から駆動すれば、
`/v1/images/generations` 的なエンドポイントへ発展可能です。JNI/パッケージ命名を親と揃えてあるため、
後段のマージが容易です。

## 7. スコープ外（未実装）

GPU(Vulkan/OpenCL)/NPU バックエンド（Vulkan は試行の上で撤回）、LoRA / ControlNet / アップスケール、
HTTP API サーバー、SDXL 系モデル。
