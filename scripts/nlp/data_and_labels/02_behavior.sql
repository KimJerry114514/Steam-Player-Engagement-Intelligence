SET threads=8; SET memory_limit='24GB';
-- per-app snapshot proxy from ALL reviews of the app (positive + negative)
CREATE OR REPLACE TABLE snap AS
SELECT appid::BIGINT AS appid, greatest(max(timestamp_created), max(author_last_played))::BIGINT AS snap_ts, count(*) AS app_reviews_all
FROM read_parquet('./raw_data/alternative_data/hf_adde023_steam_reviews_v20260312/part.*.parquet')
GROUP BY 1;
COPY snap TO 'data/app_snapshot.parquet' (FORMAT PARQUET);
COPY (
SELECT n.rid, n.appid, n.created, n.last_played, s.snap_ts,
  (s.snap_ts - n.created)/86400.0 AS age_days,
  CASE WHEN (s.snap_ts - n.created)/86400.0 >= 37  THEN (n.last_played >= n.created + 7*86400)::INT END   AS ret_d7,
  CASE WHEN (s.snap_ts - n.created)/86400.0 >= 60  THEN (n.last_played >= n.created + 30*86400)::INT END  AS ret_d30,
  CASE WHEN (s.snap_ts - n.created)/86400.0 >= 120 THEN (n.last_played >= n.created + 90*86400)::INT END  AS ret_d90,
  CASE WHEN (s.snap_ts - n.created)/86400.0 >= 210 THEN (n.last_played >= n.created + 180*86400)::INT END AS ret_d180,
  (n.last_played <= n.created + 86400)::INT AS no_play_after_review,
  greatest(n.last_played - n.created, 0)/86400.0 AS active_days_after_review,
  (n.last_played < s.snap_ts - 30*86400)::INT AS churned_30d_dormant,
  greatest(n.pt_forever - n.pt_at_review, 0)/60.0 AS extra_hours,
  n.pt_at_review/60.0 AS hours_at_review, n.pt_forever/60.0 AS hours_forever, (n.pt_2w > 0)::INT AS active_last_2w_at_snapshot,
  (n.updated > n.created + 3600)::INT AS edited,
  date_trunc('quarter', to_timestamp(n.created))::DATE AS review_quarter,
  n.language, n.num_games_owned, n.num_reviews, n.votes_up, n.votes_funny, n.weighted_vote_score, n.comment_count,
  n.steam_purchase, n.received_for_free, n.early_access, n.review_len
FROM 'data/neg_all.parquet' n JOIN snap s USING(appid)
) TO 'data/neg_behavior.parquet' (FORMAT PARQUET, COMPRESSION ZSTD);
SELECT count(*) n, round(avg(ret_d30),3) d30, count(ret_d30) elig_d30, round(avg(ret_d90),3) d90, round(avg(no_play_after_review),3) no_play_after FROM 'data/neg_behavior.parquet';
