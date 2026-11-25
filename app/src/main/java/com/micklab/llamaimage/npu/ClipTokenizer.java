package com.micklab.llamaimage.npu;

import org.json.JSONObject;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * CLIP byte-level BPE tokenizer (same output as transformers' CLIPTokenizer without ftfy):
 * lower-cased, whitespace-normalized text, CLIP's split regex, bytes->unicode mapping, BPE with
 * "</w>" word ends. encode() returns 77 ids: BOS, up to 75 tokens, EOS, padded with EOS.
 */
public final class ClipTokenizer {
    public static final int BOS = 49406, EOS = 49407, LENGTH = 77;

    private static final Pattern PAT = Pattern.compile(
            "<\\|startoftext\\|>|<\\|endoftext\\|>|'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+",
            Pattern.CASE_INSENSITIVE);

    private final Map<String, Integer> vocab = new HashMap<>();
    private final Map<String, Integer> ranks = new HashMap<>();
    private final Map<String, String[]> cache = new HashMap<>();
    private final char[] byteToChar = new char[256];

    public ClipTokenizer(File vocabJson, File mergesTxt) throws IOException {
        try {
            JSONObject v = new JSONObject(IoUtil.readString(vocabJson));
            for (Iterator<String> it = v.keys(); it.hasNext(); ) {
                String k = it.next();
                vocab.put(k, v.getInt(k));
            }
        } catch (org.json.JSONException e) {
            throw new IOException("tokenizer vocab を解析できません", e);
        }
        List<String> lines = IoUtil.readLines(mergesTxt);
        // transformers: merges[1 : 49152 - 256 - 2 + 1] (skip the "#version" header line)
        int end = Math.min(lines.size(), 49152 - 256 - 2 + 1);
        for (int i = 1; i < end; i++) {
            String line = lines.get(i).trim();
            if (line.isEmpty()) continue;
            ranks.put(line, i - 1);   // key "a b"
        }
        // bytes_to_unicode(): printable bytes map to themselves, the rest to 256+n.
        int n = 0;
        for (int b = 0; b < 256; b++) {
            boolean printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
            byteToChar[b] = printable ? (char) b : (char) (256 + n++);
        }
    }

    public int[] encode(String text) {
        String clean = text.replaceAll("\\s+", " ").trim().toLowerCase(java.util.Locale.ROOT);
        List<Integer> ids = new ArrayList<>();
        Matcher m = PAT.matcher(clean);
        while (m.find()) {
            byte[] utf8 = m.group().getBytes(StandardCharsets.UTF_8);
            StringBuilder sb = new StringBuilder();
            for (byte b : utf8) sb.append(byteToChar[b & 0xFF]);
            for (String piece : bpe(sb.toString())) {
                Integer id = vocab.get(piece);
                if (id != null) ids.add(id);
            }
        }
        int[] out = new int[LENGTH];
        java.util.Arrays.fill(out, EOS);
        out[0] = BOS;
        int count = Math.min(ids.size(), LENGTH - 2);
        for (int i = 0; i < count; i++) out[i + 1] = ids.get(i);
        out[count + 1] = EOS;
        return out;
    }

    private String[] bpe(String token) {
        String[] hit = cache.get(token);
        if (hit != null) return hit;
        List<String> word = new ArrayList<>();
        for (int i = 0; i < token.length(); i++) {
            String c = String.valueOf(token.charAt(i));
            word.add(i == token.length() - 1 ? c + "</w>" : c);
        }
        while (word.size() > 1) {
            int best = Integer.MAX_VALUE, at = -1;
            for (int i = 0; i < word.size() - 1; i++) {
                Integer r = ranks.get(word.get(i) + " " + word.get(i + 1));
                if (r != null && r < best) {
                    best = r;
                    at = i;
                }
            }
            if (at < 0) break;
            String a = word.get(at), b = word.get(at + 1);
            List<String> merged = new ArrayList<>(word.size());
            for (int i = 0; i < word.size(); ) {
                if (i < word.size() - 1 && word.get(i).equals(a) && word.get(i + 1).equals(b)) {
                    merged.add(a + b);
                    i += 2;
                } else {
                    merged.add(word.get(i++));
                }
            }
            word = merged;
        }
        String[] res = word.toArray(new String[0]);
        cache.put(token, res);
        return res;
    }
}
