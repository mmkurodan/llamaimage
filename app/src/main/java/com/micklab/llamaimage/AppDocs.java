package com.micklab.llamaimage;

import android.content.Context;

/** Texts for the ⋮ menu: user manual, rights / licenses, privacy policy. */
final class AppDocs {

    private AppDocs() {}

    static final String CONTACT = "micklab2026@gmail.com";

    private static String version(Context ctx) {
        try {
            return ctx.getPackageManager().getPackageInfo(ctx.getPackageName(), 0).versionName;
        } catch (Exception e) {
            return "";
        }
    }

    static String manual(Context ctx) {
        int port = com.micklab.llamaimage.api.ImageApiService.port(ctx);
        return "llamaimage " + version(ctx) + " 操作マニュアル\n\n"
                + "1. はじめに\n"
                + "画像生成（Stable Diffusion）をすべてこの端末の CPU 上で行うアプリです。"
                + "初回のみモデル（約 1.6〜2.0 GB）のダウンロードが必要で、以降はオフラインで生成できます。\n\n"
                + "2. 基本の使い方（テキスト→画像）\n"
                + "・起動時は推奨モデル（DreamShaper 8 LCM）が選択されています。\n"
                + "・未ダウンロードの場合は「モデルをダウンロード」を押します（Wi-Fi 推奨。アプリを閉じても続きます）。\n"
                + "・読み込みが終わったらプロンプトを入力して「生成する」を押します。\n"
                + "・生成中は進捗と残り時間が表示され、「中止」で止められます。\n"
                + "・できた画像は下の「保存」で Pictures/llamaimage に PNG で保存されます。\n"
                + "・「シードを固定」（または画像の長押し）で同じ構図のまま調整できます。\n\n"
                + "3. 画像→画像（img2img）\n"
                + "・タブを「画像→画像」に切り替え、左の枠をタップして入力画像を選びます（中央を正方形に切り抜き）。\n"
                + "・スライダーで変化量を決めます（0.05 = ほぼ元画像のまま、1.00 = ほぼ新規生成）。\n"
                + "・生成した画像の下の「再加工」を押すと、その画像が入力画像になり、このタブに切り替わります。"
                + "プロンプトや変化量を変えて「生成する」で繰り返し加工できます。\n"
                + "・プロンプト以下はテキスト→画像と共通です。\n\n"
                + "4. 詳細設定\n"
                + "・ネガティブプロンプト：CFG が 1 より大きいときに有効です。\n"
                + "・ステップ数／サイズ（512〜256px）／シード（-1 でランダム）。\n"
                + "・「生成設定」：サンプラー、CFG、TAESD（高速デコード）、VAE タイル処理（省メモリ）、重み精度、スレッド数、性表現制限。\n"
                + "・モデルを選ぶとそのモデルの推奨サンプラー／ステップ数／CFG が適用されます。\n\n"
                + "5. モデル\n"
                + "画面上部のモデルカードをタップすると、ダウンロード・切替・削除ができます。"
                + "ダウンロードしたファイルは SHA-256 で検証してから使います。\n\n"
                + "6. 日本語プロンプト（LLM 翻訳）\n"
                + "日本語のプロンプトは、⋮ → LLM設定 で指定した Ollama 互換 LLM（既定: com.micklab.llama、"
                + MainActivity.DEFAULT_LLM_BASE_URL + "）で英訳してから生成します。英語のみなら設定不要です。\n\n"
                + "7. 画像生成 API（API タブ）\n"
                + "スイッチを ON にすると、この端末内の他のアプリから画像生成を呼び出せます。\n"
                + "・アドレス: http://127.0.0.1:" + port + "（外部ネットワークからは接続できません）\n"
                + "・POST /generate  {\"prompt\": \"…\"} → {\"image\": PNG の Base64, \"seed\", …}\n"
                + "・POST /v1/images/generations（OpenAI 互換）\n"
                + "・GET /health ・ GET /models ・ POST /cancel\n"
                + "モデル・サイズ・ステップ数などは、指定がなければこのアプリの設定を使います。"
                + "他のアプリから呼ばれたときは自動で起動し、動作中は通知が表示されます。\n"
                + "例: curl http://127.0.0.1:" + port + "/generate -H 'Content-Type: application/json' "
                + "-d '{\"prompt\":\"a cat in the snow\"}'\n\n"
                + "8. 困ったとき\n"
                + "・生成中にアプリが落ちる：生成設定で「タイル処理」を ON、サイズを小さく。\n"
                + "・遅い：TAESD を ON、ステップ数を減らす、SD-Turbo（1 step）を使う。\n"
                + "・画面下部のログに詳細が表示されます（長押しで選択・コピー可）。\n";
    }

    static String rights(Context ctx) {
        return "llamaimage " + version(ctx) + "\n"
                + "Copyright (c) 2026 Mitsuo Kuroda（Mick Lab）\n\n"
                + "■ 本アプリ\n"
                + "本アプリの著作権は作者に帰属します。作者の書面による事前の許可なく、本アプリを商用目的"
                + "（販売、有償サービスでの利用、商用製品への組み込みを含む）で利用することはできません。"
                + "お問い合わせ: " + CONTACT + "\n"
                + "本アプリは「現状のまま」提供され、明示・黙示を問わずいかなる保証もありません。"
                + "本アプリの利用により生じた損害について、作者は責任を負いません。\n\n"
                + "■ 生成した画像について\n"
                + "・生成した画像の利用は、利用者ご自身の責任で行ってください。\n"
                + "・画像生成モデルは本アプリに同梱されておらず、利用者が各配布元（Hugging Face / Civitai）から"
                + "ダウンロードします。モデルおよび生成物の利用には、各モデルのライセンス"
                + "（CreativeML OpenRAIL-M などの利用制限を含む）が適用されます。\n"
                + "・他者の権利（著作権・肖像権・商標権など）を侵害する画像、違法または有害な画像の生成・公開は行わないでください。\n\n"
                + "■ ダウンロードして利用するモデル（同梱なし）\n"
                + "・DreamShaper 8 LCM — DreamShaper 8（Lykon）を LCM 化した SD1.5 系モデル。CreativeML OpenRAIL-M\n"
                + "・AbsoluteReality 1.8.1 / AbsoluteReality LCM — Lykon。CreativeML OpenRAIL-M（Civitai の追加条件あり）\n"
                + "・Photon v1 — Photographer。CreativeML OpenRAIL-M（Civitai の追加条件あり）\n"
                + "・Photon LCM — machinedelusions（Photon の LCM 版）。Civitai 記載の条件に従う\n"
                + "・Juggernaut Reborn — KandooAI / RunDiffusion。CreativeML OpenRAIL-M（作者クレジット表記が必要）\n"
                + "・SD-Turbo — Stability AI。Stability AI のライセンス（配布ページ参照）\n"
                + "・Stable Diffusion 1.5 — CreativeML OpenRAIL-M\n"
                + "・TAESD — madebyollin。MIT License\n"
                + "詳細と最新の条件は、各モデルの配布ページ（Hugging Face / Civitai）をご確認ください。\n\n"
                + "================================================================\n"
                + "THIRD-PARTY SOFTWARE NOTICES\n"
                + "================================================================\n\n"
                + "--- stable-diffusion.cpp ---\n"
                + "MIT License — Copyright (c) 2023 leejet\n\n"
                + "--- ggml ---\n"
                + "MIT License — Copyright (c) 2023-2026 The ggml authors\n\n"
                + "--- stable-diffusion.cpp に含まれるサードパーティコンポーネント ---\n"
                + "nlohmann/json（MIT）、stb（MIT / Public Domain）、zip / miniz（Unlicense / MIT）ほか。"
                + "各コンポーネントのライセンスに従います。\n\n"
                + "--- MIT License ---\n"
                + "Permission is hereby granted, free of charge, to any person obtaining a copy of this "
                + "software and associated documentation files (the \"Software\"), to deal in the Software "
                + "without restriction, including without limitation the rights to use, copy, modify, merge, "
                + "publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons "
                + "to whom the Software is furnished to do so, subject to the following conditions:\n\n"
                + "The above copyright notice and this permission notice shall be included in all copies or "
                + "substantial portions of the Software.\n\n"
                + "THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, "
                + "INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR "
                + "PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE "
                + "FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR "
                + "OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER "
                + "DEALINGS IN THE SOFTWARE.\n";
    }

    static String privacy() {
        return "プライバシーポリシー\n\n"
                + "1. 収集する情報\n"
                + "本アプリは個人情報を収集せず、作者や第三者のサーバーへ送信しません。"
                + "以下の情報を端末内にのみ保存します。\n"
                + "・設定（モデル選択、生成パラメータ、LLM の URL／モデル名、API のポートと有効状態）\n"
                + "・最後に入力したプロンプト／ネガティブプロンプト（次回起動時の復元用）\n"
                + "・ダウンロードしたモデルファイル\n"
                + "・ログ（sd.log。生成時のプロンプトや処理状況を含む場合があります）\n"
                + "・「画像を保存」を押した画像（Pictures/llamaimage）\n"
                + "・img2img で選択した画像は生成処理のためにメモリ上で使うのみで、アプリ内に保存しません。\n\n"
                + "2. 通信\n"
                + "・画像の生成は端末内で行い、プロンプトや画像をインターネットへ送信しません。\n"
                + "・モデルのダウンロード時は、アプリに内蔵した URL（Hugging Face または Civitai）へ Android の"
                + "ダウンロードマネージャーで接続します。この通信には各サービスのプライバシーポリシーが適用されます。\n"
                + "・日本語プロンプトの英訳のため、プロンプトを「LLM設定」で指定した LLM API に送信します。"
                + "既定はこの端末内（" + MainActivity.DEFAULT_LLM_BASE_URL + "）です。"
                + "端末外の URL を指定した場合はその送信先にプロンプトが送られ、取扱いは送信先に依存します。\n"
                + "・画像生成 API を有効にした場合、127.0.0.1（この端末内）でのみ接続を受け付けます。"
                + "外部ネットワークからは接続できません。\n\n"
                + "3. 利用目的\n"
                + "上記の情報は、画像生成・表示・設定の復元のためにのみ使用し、第三者へ提供しません。\n\n"
                + "4. 権限\n"
                + "・インターネット：モデルのダウンロードと LLM API への接続\n"
                + "・フォアグラウンドサービス／通知／ウェイクロック：画像生成 API の動作中表示と、生成中のスリープ防止\n\n"
                + "5. 保存期間と削除\n"
                + "設定・モデル・ログは、利用者が削除するかアプリのデータを消去（またはアンインストール）するまで保持されます。"
                + "モデルはモデル選択画面から個別に削除できます。保存した画像はギャラリー等から削除できます。\n\n"
                + "6. お問い合わせ\n"
                + CONTACT + "\n\n"
                + "© Mick Lab — 生成AIアプリケーション研究\n";
    }
}
