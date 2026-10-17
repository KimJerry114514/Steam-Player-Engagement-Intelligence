-- Build the cleaned review-text files that the classifier reads (run 2026-09-27; recovered from the session log,
-- where it ran inline). Input: data/neg_all.parquet from 01_extract_negatives.sql.
-- Cleaning: strip Steam BBCode tags such as [b] [/b] [url=...], collapse whitespace, keep the first 1,500 characters.
SET threads=6; SET memory_limit='20GB'; SET preserve_insertion_order=true;
COPY (SELECT rid, substr(trim(regexp_replace(regexp_replace(coalesce(review,''), '\[/?[a-zA-Z0-9*]+(=[^\]]*)?\]', ' ', 'g'), '\s+', ' ', 'g')), 1, 1500) AS text
      FROM 'data/neg_all.parquet' ORDER BY rid)
TO 'data/neg_text.parquet' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 200000);

-- The five deep-dive titles (Counter-Strike, Terraria, PUBG, Cyberpunk 2077, Apex Legends)
COPY (SELECT t.rid, t.text FROM 'data/neg_text.parquet' t
      JOIN (SELECT rid FROM 'data/neg_all.parquet' WHERE appid IN (730,105600,578080,1091500,1172470)) f USING(rid) ORDER BY t.rid)
TO 'data/neg_text_focus.parquet' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000);

-- A fixed 1,000,000-review sample of all other games
COPY (SELECT t.rid, t.text FROM 'data/neg_text.parquet' t JOIN 'data/neg_all.parquet' n USING(rid)
      WHERE n.appid NOT IN (730,578080,1172470,1091500,105600) ORDER BY hash(t.rid*2654435761+99) LIMIT 1000000)
TO 'data/neg_text_sample1m.parquet' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000);
