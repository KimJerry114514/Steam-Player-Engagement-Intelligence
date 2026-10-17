# NLP Complaint Features: What Each Negative Review Complains About

We turned the text of all 16,412,862 negative reviews (80,274 games, 2010–2023) into 35 complaint features. Each feature says whether a review raises one specific complaint, such as crashes, cheating or unclear goals. The models and the K-means personas use these features, either one by one or aggregated into eight complaint domains.

## How the features were built

1. **Complaint codebook.** A large language model read a stratified sample of 3,248 negative reviews (the five deep-dive titles and 25 gameplay families) and described each complaint in its own words, without a predefined list. The descriptions were merged, drafted into a codebook from three perspectives, reconciled and checked for overlaps and gaps. The result is a codebook of 78 complaint factors: 46 apply to all games, 16 to one gameplay family and 16 to one title. Each factor has a definition, include and exclude rules, and typical wording in several languages. As a coverage check, we embedded all 2.19 million reviews of the five deep-dive titles, grouped them into 320 clusters and read each cluster; no complaint outside the codebook appeared.
2. **Labeled reviews.** The language model labeled 9,720 reviews with the codebook and quoted the review text as evidence for every factor. Round one (3,730 reviews) was labeled and then re-checked in a second pass; round two (5,990 reviews) was chosen by active learning to cover rare factors. A second model relabeled part of the sample independently (agreement F1 0.82). A human reference set of 140 reviews agreed with the model labels at domain level with F1 0.65–0.72.
3. **Classifier.** We fine-tuned a multilingual sentence encoder (multilingual-e5-large) as a multi-label classifier on 8,427 labeled reviews and tested it on 1,293 held-out reviews. Each factor's decision threshold was fixed on a validation slice of the training data. The 64 modeled factors reach a weighted F1 of 0.657 on the test reviews.
4. **Features delivered.** We kept the 35 factors with held-out F1 ≥ 0.40 and at least five held-out positives (F1 0.40–0.92) and scored every negative review. For each factor, `flag_*` is a 0/1 feature and `p_*` is the model probability; `n_factors_flagged` counts the flags. On average a review raises 1.19 complaints, and 64% of reviews raise at least one.

The 35 factors fall into eight domains: technical (7 factors), gameplay (4), progression (1), content (3), social (5), user experience (5), business (5) and service (5). The full list, with definitions and test accuracy, is in the factor dictionary.

## What the features show

- Holding playtime at review and other posting-time inputs fixed, and comparing reviews of the same game and quarter, technical and service complaints (crashes, connection failures, performance, broken functions, "updates made it worse") go with continued play, by about +1 to +7 percentage points.
- Fit and experience complaints (story, play rhythm, onboarding) go with leaving.
- The pattern holds in the five deep-dive titles and in all other games, at 30 and at 90 days, and on unedited reviews only.
- Adding the features improves every prediction model (see the model comparison).

## Limitations

- **Measurement error.** The features are predictions, not hand labels. Measurement error biases their estimated effects toward zero.
- **Associations, not causes.** A complaint that goes with leaving does not show that fixing it would keep the player.
- **Edited text.** Review text is the latest edited version. About 11% of reviews were edited more than 30 days after posting, so their text can reflect later events. The results hold on unedited reviews.
