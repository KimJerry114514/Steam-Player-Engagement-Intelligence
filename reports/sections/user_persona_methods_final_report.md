# User Persona Analysis Methods

## Historical review coverage

We constructed complaint profiles for all 16,412,862 cleaned historical negative-review records, covering October 15, 2010 through November 3, 2023. The analysis includes the training period and 711,163 records without a binary D30 outcome. Each observation represents a player–game review record rather than a unique player. Including records without observed outcomes allows the profiles to describe the full available negative-review history.

## Assignment to the four existing segments

We extended the existing P55/P55 segmentation framework to the historical cohort using the frozen LightGBM model and its saved preprocessing settings. Higher prior engagement was defined as playtime at review greater than 11.78 hours. Higher predicted later activity was defined as a raw model score at or above the relevant period cutoff: 0.603875151020 for 2010–2021, 0.563707377961 for 2022, and 0.531534888781 for 2023. Hardcore combines higher prior engagement and a higher predicted score; Dropout combines higher prior engagement and a lower score; New Blood combines lower prior engagement and a higher score; and Nobody combines lower prior engagement and a lower score. These names describe model-defined groups and do not confirm actual churn or customer value.

We preserved the original 4,937,916 eligible validation/test scores and segment assignments and scored the remaining 11,474,946 records with the existing model. The added training-period cutoff was the 55th percentile of predictions for the original eligible training cohort. The existing 2022 and 2023 cutoffs remained fixed. Records without binary outcomes received their period’s cutoff without changing its reference cohort. Training-cohort predictions are in sample and are used here for descriptive profiling.

## Linking complaint factors to review records

We joined the historical segment assignments to the existing NLP results using the combination of appid and recommendationid. We retained the 35 supplied binary complaint flags and their existing definitions. We grouped these factors into eight domains: Technical, Gameplay, Progression, Content, Social, User experience, Business, and Service. A domain was flagged when at least one of its constituent factors was flagged. A review could therefore contain multiple complaint factors and domains.

## Calculating complaint prevalence and rankings

For each segment, we calculated factor prevalence as the number of records with that factor flagged divided by all records in the segment. Domain prevalence used the same denominator and counted records with any factor in that domain flagged. Records with no flagged complaints remained in the denominator. Consequently, the percentages describe within-segment occurrence rates and do not sum to 100%. Domains contain unequal numbers of retained factors, so their prevalence is not a standardized measure of complaint severity.

We ranked the 35 factors by their unrounded prevalence within each segment and selected the five highest rates, breaking ties by factor ID. For supplementary comparisons, we calculated each segment’s prevalence minus the overall prevalence across all historical review records. These contrasts are percentage-point differences from a review-weighted baseline. We also produced separate summaries for 2010–2021, 2022, and 2023 using all available negative reviews in each period, including records without binary outcomes.

## Visualization and verification

We summarized the eight-domain comparison in a heatmap and displayed each segment’s Top 5 factors in a separate panel of a combined bar chart. A second heatmap presents all 35 factors for the appendix. We verified that every historical review matched exactly one segment and one NLP record, that all retained flags were binary and non-null, and that every segment assignment followed its numeric rules. An independent full-cohort recount confirmed the factor counts, domain unions, denominators, rates, and rankings. The original eligible validation/test scores and assignments were unchanged.

These profiles are retrospective and descriptive. Complaint flags are NLP estimates rather than human-confirmed labels. Because NLP features also contribute to the segmentation model, differences between groups partly reflect how the groups were constructed. The profiles do not provide independent predictive validation or establish that a complaint caused a player to stop playing.

## Required Figures

Figure numbers below are placeholders and should follow the numbering in the complete final report. All three figures use the complete historical cohort. These assembly notes are separate from the methods text above.

### Figure 1 Eight domain complaint prevalence by segment

Placement: Main text, after the prevalence calculation method.

PNG: [Open figure](../figures/user_persona/domain_prevalence_heatmap.png). Vector PDF: [Open figure](../figures/user_persona/domain_prevalence_heatmap.pdf).

Caption: Complaint prevalence across eight domains and four historical review segments. Each cell shows the percentage of all review records in that segment with at least one flagged factor in the domain. Domains overlap, and percentages do not sum to 100%.

### Figure 2 Top five complaint factors within each segment

Placement: Main text, after the description of the Top 5 ranking.

PNG: [Open figure](../figures/user_persona/top5_complaints_by_segment.png). Vector PDF: [Open figure](../figures/user_persona/top5_complaints_by_segment.pdf).

Caption: The five most prevalent complaint factors within each historical review segment. Bars show the percentage of all records in the corresponding segment with that factor flagged. Rankings use unrounded within-segment prevalence; the same factor may appear in multiple segments.

### Figure A1 Prevalence of all 35 complaint factors

Placement: Appendix.

PNG: [Open figure](../figures/user_persona/factor_prevalence_heatmap.png). Vector PDF: [Open figure](../figures/user_persona/factor_prevalence_heatmap.pdf).

Caption: Within-segment prevalence of all 35 retained NLP complaint factors, grouped by domain. Each cell shows the percentage of all review records in the corresponding segment with the factor flagged. Factors overlap. The analysis covers all 16,412,862 historical negative-review records.

The percentage-point difference heatmaps and period-specific figures are supplementary; they are not required for the core methods section.
