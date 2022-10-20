# Hate speech detection with context information

Build classifier to detect hate speech from Fox News website.

The dataset is from https://arxiv.org/pdf/1710.07395.pdf
which contains annotated disussion threads from 10 piece of news on Fox News website. Each comment is provided with surrounding comments.
This project attempts to usw previous comments as context information to build context-aware classifier to see whether the performance is improved and therefore investigate the influence of context 
on hate speech detection.

Example:
```
"So, how many sexual assaults would there be if Muslims were burned in ovens?"

"Probably a lot less. What's your point?"
```

The above two sentences are in the same discussion thread. They are clearly hate speech against a particular religion but it would be hard to understand the hateful intent of the second sentence without considering the previous sentence.
Effective hate speech detector should be able to flag both sentences.
