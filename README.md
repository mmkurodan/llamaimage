# llamaimage — SD1.5 画像生成 PoC（Android / llama.cpp 系 ggml 基盤）

`/root/llama`（**LLM AI Server**: 端末上で llama.cpp を JNI で動かし、対話 UI と Ollama 互換 API を
提供するアプリ）の**姉妹 PoC** です。将来 LLM AI Server に画像生成機能として組込むことを前提に、
同じ「JNI ネイティブ + シングルトン Manager + 最小 UI」構成を踏襲した **text2img の最小実装**です。

> **技術メモ**: SD1.5 の GGUF 推論は llama.cpp 本体ではなく、**同じ ggml 基盤の
> [`stable-diffusion.cpp`](https://github.com/leejet/stable-diffusion.cpp)** が担います。本 PoC は
> これを JNI（`libsd_jni.so`）で組込みます。バックエンドは **CPU のみ / arm64-v8a**。

対象モデル: **`Sashkanik13/sd1.5-text2img-gguf`** の単一ファイル `model_q4_0.gguf`
（約 1.57GB, UNet+CLIP+VAE 一体型, Q4_0）。

---

## 1. モデルの準備（端末側）

`model_q4_0.gguf` を端末の **ダウンロードフォルダ** に置いておきます（例: PC からコピー、または
ブラウザでダウンロード）。アプリはここから **ファイルピッカー（SAF）経由で読み込み**ます
（アプリ内ダウンロードや特別なストレージ権限は不要）。

```
# 例: adb で端末の Download に転送
adb push model_q4_0.gguf /sdcard/Download/
```

## 2. ビルド

ネイティブ依存（stable-diffusion.cpp + ggml）は**リポジトリに同梱せず**、固定コミットを
ビルド時に取得します（`scripts/fetch-sdcpp.sh`、`.gitignore` 済み）。

### CI（推奨）
`main` への push で `.github/workflows/android.yml` が動作し、NDK 27.2.12479018 + CMake を導入 →
`fetch-sdcpp.sh` → `assembleDebug` / `assembleRelease` / `bundleRelease` を実行して APK/AAB を生成します。

### ローカル
```bash
bash scripts/fetch-sdcpp.sh          # 依存取得（冪等。CMake configure 時にも自動実行）
./gradlew assembleDebug              # 要: Android SDK + NDK 27.2.12479018 + CMake
```

## 3. 使い方

1. アプリ起動後、上部に「準備完了 — モデルを選択してください」と出れば native 初期化成功
   （arm64-v8a 実機が必要。エミュレータ等では「ネイティブ初期化失敗」と表示）。
2. **「モデル選択 (.gguf を Download から)」** → `model_q4_0.gguf` を選択。ファイルは
   コピーせず**ファイルディスクリプタ経由で直接読込**（失敗時のみ内部ストレージへコピー）。
3. プロンプト（既定 `a photo of a cat`）、steps（既定 20）、size（既定 512）を入力。
4. **「生成」** → 進捗（step x/y）→ 完了後に画像が表示されます。

> CPU 生成のため SD1.5 は端末で **数十秒〜数分/枚** かかります（PoC の動作確認用）。

---

## 4. 構成

| パス | 役割 |
|------|------|
| `app/src/main/java/.../StableDiffusionNative.java` | JNI インターフェース（親の `LlamaNative` 相当） |
| `app/src/main/java/.../SdModelManager.java` | 単一ネイティブコンテキストの所有・busy 管理（親の `ModelManager` 相当） |
| `app/src/main/java/.../MainActivity.java` | 最小 UI（モデル選択→生成→表示） |
| `app/src/main/cpp/jni/jni_sd.cpp` | `new_sd_ctx`/`txt2img` を叩く JNI 実装（`libsd_jni.so`） |
| `app/src/main/cpp/CMakeLists.txt` | CPU 専用・arm64・16KB ページアライン。`sd/` を取得して `add_subdirectory` |
| `scripts/fetch-sdcpp.sh` | 固定コミットの stable-diffusion.cpp + ggml を取得 |

**pin バージョン**: stable-diffusion.cpp `9578fdcc`（2024-11-30, フラットな `txt2img` API・単一 CLIP
vocab の軽量版）/ ggml `6fcbd60`（上記コミットの gitlink）。

## 5. 将来の LLM AI Server 組込み（設計のみ）

`SdModelManager` を、親アプリの `OllamaForegroundService` / `OllamaApiServer` 相当から駆動すれば、
`/v1/images/generations` 的なエンドポイントへ発展可能です。JNI/パッケージ命名を親と揃えてあるため、
後段のマージが容易です。

## 6. スコープ外（PoC では未実装）

GPU(OpenCL)/NPU バックエンド、img2img / LoRA / ControlNet / アップスケール、HTTP API サーバー、
アプリ内モデルダウンロード。
