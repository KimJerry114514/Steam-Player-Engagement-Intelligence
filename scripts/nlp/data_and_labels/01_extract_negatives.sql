SET threads=4;
SET memory_limit='16GB';
SET preserve_insertion_order=false;
COPY (
  SELECT row_number() OVER () AS rid,
    appid::BIGINT AS appid, recommendationid::BIGINT AS recommendationid, game,
    author_steamid AS author_steamid_approx,
    author_num_games_owned::BIGINT AS num_games_owned, author_num_reviews::BIGINT AS num_reviews,
    author_playtime_forever::BIGINT AS pt_forever, author_playtime_last_two_weeks::BIGINT AS pt_2w,
    author_playtime_at_review::BIGINT AS pt_at_review, author_last_played::BIGINT AS last_played,
    language, timestamp_created::BIGINT AS created, timestamp_updated::BIGINT AS updated,
    votes_up::BIGINT AS votes_up, votes_funny::BIGINT AS votes_funny, weighted_vote_score,
    comment_count::BIGINT AS comment_count, steam_purchase::INT AS steam_purchase,
    received_for_free::INT AS received_for_free, written_during_early_access::INT AS early_access,
    length(review) AS review_len, review
  FROM read_parquet('./raw_data/alternative_data/hf_adde023_steam_reviews_v20260312/part.*.parquet')
  WHERE voted_up = 0
) TO './data/neg_all.parquet' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000);
