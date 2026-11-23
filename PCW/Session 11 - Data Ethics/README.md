# Session 11 PCW: Data Ethics

**Date**: Fall 2026, Session 11
**Student**: Katia Gwaneza Nkurunziza
**Status**: Complete (Question 1's quiz transcript and Question 3's full ethics discussion both live in this file; Question 2's slides were completed separately)

---

## Overview

This PCW covers all 6 questions of the "Data Ethics" assignment. Question 1 (the LLM-quiz exercise on bias vocabulary) is answered in full below, transcript included. Question 2 (the two-slide dataset-bias presentation) was completed separately, not duplicated here. Questions 3-5, imagining being a data scientist serving two clients (a city police department with homicide data, a bank with house-price data), figuring out what models each client could use, what happens when the two datasets get combined, and the ethical considerations that come with that combination, are covered in this README (the full Q3 ethics answer) and the paired notebook (Q1/Q2 modeling discussion, Q4's real code prediction, Q5's presentation notes, and the location-visualization extension). Question 6 (optional alternate upload format) doesn't apply, this notebook is the submission.

## Question 1 of 6: Data Ethics Quiz Transcript

Per the assignment's prompt structure: 2 given questions, 3 LLM-generated follow-ups, then specific feedback plus a real citation for each of the 5 answers.

**Q1. Describe the bias-variance trade-off in your own words. What does it mean that this is a dilemma?**

*My answer:* Bias is how far off a model's average prediction is from the true value, usually because the model is too simple to capture the real pattern. Variance is how much the model's predictions swing around if I trained it on a different sample from the same population, usually because the model is flexible enough to fit noise in that particular sample. As I make a model more flexible (more parameters, deeper trees), bias goes down because it fits the training data better, but variance goes up because it becomes sensitive to whatever specific sample it happened to see. It's a dilemma because I can't tune bias and variance independently to both be low: pushing one down pushes the other up, so I'm always choosing a point on that curve, not eliminating the problem.

*Feedback:* Vocabulary is used correctly and the direction of the tradeoff is right. Sharpen it by naming the irreducible error term (total expected error = bias² + variance + noise) and giving a concrete example, like comparing linear regression to a deep unpruned decision tree.

*Reference:* Geman, Bienenstock & Doursat (1992), "Neural Networks and the Bias/Variance Dilemma," *Neural Computation* 4(1). The paper that coined the term.

**Q2. Explain the differences between "bias" in a statistical context and "bias" in conversational English.**

*My answer:* Statistical bias means an estimator systematically over- or under-predicts the true value, defined as the gap between a model's expected prediction and the actual parameter, and by itself says nothing about fairness or group membership. Conversational bias, like "political bias" or "biased survey," means a judgment or measurement is skewed by an agenda, a sampling problem, or a lack of objectivity, usually carrying a value judgment that something is unfair. A model can be statistically unbiased overall and still produce outcomes people would call conversationally biased, because the error structure differs across subgroups even when the overall average is zero.

*Feedback:* Good distinction. Make it sharper by naming the two as entirely different categories: statistical bias is a mathematical property of an estimator, conversational/social bias is a normative claim about fairness or representation.

*Reference:* Mehrabi, Morstatter, Saxena, Lerman & Galstyan (2021), "A Survey on Bias and Fairness in Machine Learning," *ACM Computing Surveys* 54(6) (arXiv:1908.09635). The standard taxonomy for exactly this distinction.

**Q3 (follow-up). How might a model with low bias and high variance perform differently on training data versus new, unseen data?**

*My answer:* A model with low bias and high variance fits the training data very well, sometimes almost perfectly, because it's flexible enough to capture the specific patterns and noise in that exact sample. On new, unseen data, performance drops noticeably because the model learned quirks of the training set that don't generalize, that's overfitting. The signature is a big gap between training error and test error: low bias keeps training error small, but high variance means that low error doesn't transfer.

*Feedback:* Correct and well connected to overfitting. Add that this gap is exactly what cross-validation is built to detect, and that the irreducible error floor means even a perfect model won't hit zero test error.

*Reference:* James, Witten, Hastie & Tibshirani, *An Introduction to Statistical Learning*, 2nd ed., Section 2.2.2 (free at statlearning.com). The standard textbook treatment with real train/test error curves.

**Q4 (follow-up). Give an example of how statistical bias in a model could still result in outcomes conversationally described as "biased" against a group.**

*My answer:* A healthcare algorithm predicts how much extra care a patient needs by using past healthcare costs as a stand-in for health need. If less money has historically been spent on one group for reasons unrelated to how sick they actually are, the model systematically underestimates that group's risk scores, a biased estimator in the strict statistical sense, with nothing about race explicitly in the model. The effect is the model funnels fewer resources to that group across the board, exactly what people mean by "biased against" a group, even though the mechanism is a biased proxy variable, not an explicit rule about group membership.

*Feedback:* This matches a real, well-documented case closely. Name the mechanism more precisely: cost was used as a *proxy variable* for health need, and the proxy itself was statistically biased by group.

*Reference:* Obermeyer, Powers, Vogeli & Mullainathan (2019), "Dissecting racial bias in an algorithm used to manage the health of populations," *Science* 366(6464). The real case this answer draws from, a commercial health-risk algorithm that used cost as a proxy for need and underestimated risk for Black patients.

**Q5 (follow-up). Why can't you minimize both bias and variance simultaneously, and what does this imply about a "perfect" model?**

*My answer:* They move in opposite directions as a function of model complexity: flexible models have low bias but fit training data too closely, which raises variance; rigid models have low variance but miss real signal, which raises bias. There's no complexity level where both hit their minimum at once, so total expected error (bias² + variance + irreducible noise) bottoms out at some middle point, not at either extreme. That means there's no "perfect" model, only the best compromise for a given dataset, found in practice with cross-validation.

*Feedback:* Strong, references the decomposition and ties to cross-validation correctly. Push further: the "best compromise" is dataset-specific, so a model tuned to the optimal tradeoff on one dataset isn't guaranteed optimal on a different population, a preview of why fairness questions complicate "optimal" model selection.

*Reference:* Google Machine Learning Crash Course, "Overfitting" module (developers.google.com/machine-learning/crash-course). A widely used, accessible companion to the formal ISLR treatment above.

## Important: Substitute Datasets

The assignment links to "this homicide data" and "this data about house prices" through hyperlinks on the original assignment web page. Those hyperlinks were not captured as URLs in the PDF export of the assignment available for this work, so the exact datasets the instructor intended are unrecoverable from what's on hand.

**Substitutes used**, chosen to match the same exercise shape (a city crime dataset with location, a housing dataset with location):

1. **Chicago homicide data (2021-2025)**: City of Chicago Data Portal, "Crimes - 2001 to Present" dataset (id `ijzp-q8t2`), filtered to `primary_type = 'HOMICIDE'`, aggregated by community area (Chicago's official neighborhood unit). Pulled live via the Socrata API.
2. **Ames, Iowa housing data**: the De Cock (2011) Ames Housing dataset, the same data underlying the well-known Kaggle "House Prices: Advanced Regression Techniques" competition, with a `Neighborhood` column of real named neighborhoods. Pulled directly from its original academic host (jse.amstat.org).

Both are real, public datasets, fetched directly over HTTP with no API key or login. If the originally-linked datasets surface later, the loading cell in the notebook can be swapped out; the rest of the analysis pattern (train/test split, regression, neighborhood visualization) carries over unchanged.

## Learning Objectives

- Identify what model types fit a given client's actual decision (patrol allocation vs. loan/valuation decisions), not just "a model"
- Recognize that combining two location-linked datasets is a *specific, describable* feature-engineering move, not a vague "more data is better" claim
- Name the concrete ethical failure modes that specific combination creates (redlining-style proxy discrimination, predictive-policing feedback loops), not just "bias" in the abstract
- Build and evaluate one real regression model end to end, with an honest train/test split and a real metric, not a toy or hardcoded number
- Read a real geographic distribution (homicides by Chicago community area) and describe the actual pattern it shows, not a generic "crime varies by area" statement

## Results Summary

| Model | Metric | Value |
|---|---|---|
| Linear regression, Ames SalePrice (7 structural features, no location) | Test R² | 0.786 |
| | Test RMSE | $41,984 |
| | Test MAE | $24,741 |
| | Test-set mean SalePrice | $183,635 |
| | Rows used (after dropping missing values) | 2,928 / 2,930 |

| Visualization | Finding |
|---|---|
| Chicago homicides by community area, 2021-2025 | Top 15 of 77 areas account for ~61% of all 3,227 homicides; all 15 are West Side or South Side areas; zero North Side areas appear in the top 15; Austin is highest at 264 |

## What's in the Notebook

### Dataset Note
Prominent disclosure that the original assignment-linked datasets are unrecoverable, and what real substitutes were used instead and why.

### Q1: Models for the Police Department
Spatial hotspot / clustering models and time-series forecasting of incident counts by area. Describes combining in housing vacancy rate and price-trend-by-neighborhood as a feature to sharpen *where* resources are predicted to be needed.

### Q2: Models for the Bank
Regression models for valuation, classification/boosting models for default risk. Describes combining in neighborhood homicide counts/trend as a feature. Flags, right where this combination is proposed, that it is exactly where the ethical risk from Q3 shows up (redlining-style proxy discrimination one direction, predictive-policing feedback loops the other). Not presented as unambiguously good.

### Q3: Ethical Considerations
Pointer to this README section (below), not duplicated in the notebook.

### Code Cell 1 of 3: Load the Substitute Datasets
Loads cached CSVs from `data/` (re-downloads from the live sources if the cache is missing). Attaches Chicago community area names from a fixed, publicly documented reference list.

### Q4: A Real Prediction (Bank Use Case)
Linear regression predicting `SalePrice` from 7 structural features (`Overall Qual`, `Gr Liv Area`, `Year Built`, `Total Bsmt SF`, `Garage Cars`, `Full Bath`, `Lot Area`). Real 80/20 train/test split, real printed R²/RMSE/MAE. `Neighborhood` is deliberately excluded from the feature list, explained below.

### Q5: Presenting the Model
Short prep note: what to be ready to explain about the model, its metrics, why location was excluded, and its practical limits.

### Extension: Visualizing by Neighborhood
Horizontal bar chart of the top 15 Chicago community areas by 2021-2025 homicide count, with specific pattern description grounded in what the chart actually shows.

## Key Concepts

### Model Choice Follows the Decision, Not the Data
A police department's decision is "where do we send patrols," which is a spatial/temporal question, hotspot and forecasting models answer it. A bank's decision is "what is this asset worth" or "will this loan default," which is a per-case valuation/risk question, regression and classification models answer that. The same raw data (addresses, dates, locations) supports both, but the *useful* model shape differs because the downstream decision differs.

### Combining Datasets Is a Specific Feature, Not a Vague Boost
"Combine the datasets" cashes out concretely: housing vacancy/price-trend as an added feature in a policing model, or crime counts/trend as an added feature in a valuation/risk model. Naming the exact feature being added is what makes it possible to also name the exact ethical risk it introduces (see below), a vague "more data helps" framing hides that risk.

## Ethical Considerations (Q3, Full Answer)

The assignment suggests using "the FDS data strategy framework" as a guide. The closest real match found is the U.S. **Federal Data Strategy**'s **Data Ethics Framework** (GSA, 2020 Action Plan; strategy.data.gov / cdo.gov), which sets seven tenets for government data use: uphold applicable laws and ethical standards, respect the public and communities, respect privacy and confidentiality, act with honesty and integrity, hold yourself and others accountable, promote transparency, and stay informed of developments in data science. I can't confirm from outside the course whether this is specifically what the assignment means by "FDS", flagging that honestly rather than guessing, but the tenets (especially "respect privacy and confidentiality" and "honesty and integrity") line up with points 1, 3, and 6 below, so I'm using them as a loose structural anchor.

**1. The homicide data records policing as much as crime.** Recorded incidents depend on what gets reported, patrolled, and classified, not purely on where violence occurs. Areas with heavier historical police presence generate more recorded incidents partly *because* of that presence, independent of the true underlying rate. Treating this data as a clean measure of "crime" bakes that measurement bias into any downstream model.

**2. Combining crime data into a bank's valuation or lending model risks redlining by proxy.** A model that lowers a valuation or raises a risk score because of a neighborhood's recorded crime rate denies credit or inflates cost for every resident of that neighborhood, regardless of their individual creditworthiness, reconstructing a historical form of discrimination (redlining) through a seemingly neutral feature. Because recorded crime itself correlates with historically over-policed, majority-Black and Latino neighborhoods (see point 1), "crime rate" functions as a proxy for race even when race is never an input. This isn't hypothetical: The Markup's mortgage-lending investigation found Black, Latino, Asian, and Native American applicants denied at significantly higher rates than similarly-qualified white applicants even after holding 17 other variables constant, without race ever being a model input directly.

**3. Combining housing data into a police resource-allocation model risks a feedback loop.** Sending more patrols to neighborhoods flagged as declining (by price or vacancy trend) means more officers are present to record incidents there, which inflates that area's recorded crime, which justifies sending even more patrols next cycle, regardless of whether crime is actually higher there than anywhere else. The model's own output becomes an input to the next round of data it's trained on. This is my own proposed extension of the mechanism, not what Lum & Isaac (2016) actually modeled: their real predictive-policing algorithm (PredPol-style) takes no housing data at all, only past crime type, place, and time. Their simulation still found the feedback loop would direct patrols to Black neighborhoods at roughly twice the rate implied by actual drug-use estimates, using crime-reporting data alone. That is a stronger, more unsettling result than the housing-data version above: the loop doesn't need an extra proxy feature like housing trends to produce biased allocation, biased policing records feeding back into policing decisions is already enough on its own (Lum & Isaac, 2016, "To Predict and Serve?"; Ensign et al., 2017, "Runaway Feedback Loops in Predictive Policing" formalizes the same mechanism mathematically). Combining in housing data, as I describe above, would be an additional, separate risk on top of that, not the mechanism Lum & Isaac themselves demonstrated.

**4. Aggregation to the community-area level hides real heterogeneity (the modifiable areal unit problem).** A community area spans many blocks; a high area-level homicide count can be driven by a handful of blocks while most of the area is unaffected, and the visualization and any resource-allocation decision built on it treats the whole area uniformly regardless.

**5. The substitute datasets limit how specific this answer can be.** Because the exact datasets the assignment intended are unrecoverable, this discussion addresses the ethical issues inherent to *this class* of data (city crime records + city housing records, joined by location), which are well-documented and generalize, but it cannot speak to whatever specific collection choices, time range, or granularity the original intended dataset actually had.

**6. Individual/location privacy.** Even at aggregated levels, homicide location data can be sensitive (identifying specific blocks or addresses with low counts), and combining it with housing records (which are tied to specific parcels and owners) narrows anonymity further than either dataset alone.

None of this means the datasets shouldn't be used, it means any model built on this combination needs an explicit accounting of which features are proxies for what, and ideally an audit of outcomes by neighborhood demographic composition before deployment, not just an accuracy number.

## Files

```
PCW/Session 11 - Data Ethics/
├── README.md                  (this file, includes full Q3 ethics answer)
├── pcw_lesson_11.ipynb        (complete notebook, executed end to end, real data and metrics)
└── data/
    ├── chicago_homicide_by_community_area_2021_2025.csv   (cached API pull)
    └── AmesHousing.txt                                     (cached source file)
```

## How to Run

```bash
cd "PCW/Session 11 - Data Ethics"
jupyter notebook pcw_lesson_11.ipynb
```

Requires `pandas`, `requests`, `scikit-learn`, `matplotlib` (all present in the project's `cs156-venv`). If the `data/` cache is deleted, the notebook re-downloads both datasets live from their public sources on next run.

## Next Steps

This session's pattern, pick a real client decision, name the model that fits it, name the exact feature a second dataset would add, then name the exact ethical risk that specific combination creates, is the general move for any future "should we combine these datasets" question, not just this assignment.

---

**Last Updated**: Session 11, Fall 2026
**Status**: Ready for submission
