# Medical-Transcript-Classifier
Medical data classifier using LLMs for feature generation and XGBoost for classification

**Scenario**: Working in IT at a large hospital, you find a storage closet full of [medical transcript notes](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions?resource=download) (from Kaggle). Some are labeled with the relevant medical specialty, some are not. You wish to label the medical transcripts with the correct specialty, from a list of the known possible medical specialties.

We will try five different approaches based around using an LLM for feature extraction, and then performing some (or no) machine learning or statistical post-processing to output a classification

We make three LLM Queries to extract features from these transcripts:
1) Give title, content, and keywords, ask LLM to classify into categories while reporting confidence as in [here](https://arxiv.org/pdf/2412.14737) . This was a 16-shot, one shot per possible medical specialty
2) Give a LLM access to entire training dataset, ask it to generate one to three Likert-scale questions per medical specialty (rewarding uniqueness from other medical specialties)
3) For every transcript, ask LLM to complete the questionnaire generated from (2)

We are interested in determining the "best" way to classify different medical transcripts into medical specialties. We will compare the following classification approaches
1) 16-shotting the LLM for classification as in LLM query (1)
2) Using the LLM Likert scale responses to implement an XGBoost algorithm
3) Using the LLM Likert scale responses to implement a multiclass logistic regression with L2 regularization
4) Using both the LLM classification from (1) and the LLM Likert scale responses to implement XGBoost
5) Using both the LLM classification from (1) and the LLM Likert scale responses to implement a multiclass logistic regression with L2 regularization

We initially explored the first LLM query being three-shot, however we found it to give an terrible predictive performance (accuracy of 9%) so we elected to use the 16-shot version.

The original source data, spanning $n=4999$ rows, has $40$ unique medical specialties. To avoid paying for credits on HuggingFace, I chose to reduce this dataset to $n=178$ observations. This was done by randomly selecting $200$ observations to keep, and then removing all observations whose medical specialty occurred $2$ or fewer times in the dataset, as presumably there is not enough information available to properly classify these. This resulted in a dataset of size $n=178$. Even after this pruning, since there are many specialties with very few ($<10$) observations, we perform a train-test $75-25$ split stratified on the medical specialty, to ensure all available medical specialties are represented in the training dataset. The test-training split is only relevant for LLM prompt (2), for the other prompts they were run for every medical transcript individually.

For each approach, we will evaluate performance on both the training and testing data using the classical categorical evaluation metrics as produced by the `sklearn` package (Accuracy, Precision, Recall, and F1-score). 

**Results**: Based on the above metrics and considering interpretability, we found the multiclass logistic regression with L2 regularization on the Likert questionnaire resultsto perform very well and preserve feature interpretability. I found XGBoost with the Likert Questionnaire responses and 16-shot LLM classifications and confidence levels to be the most consistent based on accuracy on in and out of sample, but I felt the interpretability of multiclass logistic regression outweighted the increased accuracy in the training set. The 16-shot LLM to perform the best on predictive accuracy with an accuracy of 46.6%, but in-training had the relatively low accuracy of 71%, which is even worse when you consider how $16$ shots is over $10$% of our sample size. Logistic regression also performed reasonably well, but was outclassed in-sample by XGBoost. In the testing set, we outperformed a benchmark using ([multiclass logistic regression with vectorized features]([url](https://www.kaggle.com/code/bchnhtnguyn/ai-for-medical-transcriptions-dataset))) by over 60%.

Surprisingly, including the LLM classifications as a feature for logistic regression along with the Likert questionnaire degraded performance as compared to just the Likert questionnaire.

## Implementation Details 

All coding was performed in Python using the libraries: `Pandas`, `numpy`, `sklearn`, `xgboost` and `huggingface_hub`. The LLM we are using were selected from HuggingChat, specifically [DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2), the most advanced DeepSeek model available at the time of writing (to my knowledge). To generate the questionnaire, ChatGPT-5 was used, as DeepSeek experienced difficulties handling the csv file containing the training data.

During Exploratory Data Analysis (EDA), we found $33$ observations missing a transcription, and $1068$ observations missing keywords. Thus, during LLM prompting for the evaluation of the Likert scale items of (2), we were prepared to instruct the LLM to not respond if the question references the transcript or keywords section when the data does not have that section, and then impute the average response to that question during post-processing. In practice all the questions referred to the transcription in general, so this strategy did not need to be implemented.

For the approaches using Likert scale variables, we first subtract $3$ from all evaluations, so negative values correspond to affirmative, and positive to negative. Our Likert Questionnaire generated by the LLM has $33$ questions, so we elect not to employ any dimension reduction or aggregation techniques. We elect to treat these as numeric values (not ordinal) for statistical analysis, as this should be a reasonable approximation and further reduces the dimensionality of our data.

XGBoost's hyperparameters were determined by Cross-Validation on the training data accuracy score. We elected not to tune the hyperparameters of the penalized logistic regression as implemented by `sklearn`, as there were many medical specialty categories with few observations, so depending on the fold not all categories are fairly represented.

The metrics used for model comparison were all implemented in the `sklearn` Python package, and included the F1-score, precision, and recall.

All prompts are available in the `prompts` folder, stored as word files. The 16-shot prompt was generated programmatically with one shot for each of the sixteen possible classifications. The code used to produce the results shown below are in `LLM_and_models.ipynb`, the $3$-shot versions are in `LLM_and_models_depreciated_three_shot.ipynb`.

API calls implemented in Python using the `huggingface_hub` library, and if a LLM run failed to provide the desired information, we would rerun it manually.


### Prompting Strategies

Following best prompting practices as described by [Yang et al., 2024](https://arxiv.org/pdf/2412.14737), (1) uses sixteen shot prompting and requests a confidence score along with the LLM's best guess of classification, where the sixteen shots were pulled from the training dataset.

For the Likert Scale items generated in (2) and evaluated in (3), zero-shot prompting is used. Multishot prompting would be used if subject matter experts were available to create examples.

## Results

| Model                                | Accuracy -- Training | Accuracy -- Testing |
| ------------------------------------ | -------------------- | ------------------- |
| LLM 16-shot                          | $0.71$               | ==$0.47$==          |
| XGBoost Likert Scale                 | $0.98$               | $0.36$              |
| XGBoost Likert + 16-shot             | ==$0.99$==           | $0.44$              |
| Logistic Regression Likert           | $0.91$               | $0.44$              |
| Logistic Regression Likert + 16-shot | $0.94$               | $0.4$               |
| **Baseline**: Logistic Regression no LLM | $0.47$               | $0.27$              |




 Overall XGBoost Likert + 16-shot had the best training performance and a respectable testing performance. I suspect our small sample size to be the driving force behind why the accuracy is low.
## Future Directions
- Employ classical semantic analysis techniques such as word2vec for feature extraction, and compare the results
- Preprocessing the Likert scale questions, using dimension reduction techniques such as factor analysis or models from item response theory
- Expand the size of our dataset to be more robust to out-of-sample medical transcripts to improve predictive accuracy
- Explore feature importance of Likert Questions through SHAP values to determine how the LLM questionnaire could be tailored
