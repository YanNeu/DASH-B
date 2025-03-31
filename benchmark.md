
## DASH-B: Leaderboard

|Pos.| Model | ACC | TNR | TPR |
| --- | --- | --- | --- | --- |
|1|gpt-4o-mini-2024-07-18|86.3%|77.0%|95.7%|
|2|InternVL2_5-26B|77.5%|57.3%|97.8%|
|3|InternVL2_5-38B|76.2%|54.8%|97.6%|
|4|InternVL2_5-26B-MPO|76.1%|54.8%|97.4%|
|5|LLaVA-Onevision-05b|75.1%|60.2%|90.1%|
|6|InternVL2_5-78B|74.1%|50.3%|97.8%|
|7|InternVL2_5-8B|71.7%|47.2%|96.2%|
|8|Ovis2-8B|71.4%|44.8%|98.0%|
|9|PaliGemma2-10b|69.8%|48.0%|91.6%|
|10|InternVL2_5-8B-MPO|69.4%|42.3%|96.4%|
|11|PaliGemma2-3b|68.9%|40.9%|96.8%|
|12|LLaVa-1.6-Llama|65.2%|37.0%|93.4%|
|13|Ovis2-4B|64.8%|31.0%|98.6%|
|14|Ovis2-1B|64.6%|35.1%|94.0%|
|15|PaliGemma-3b|62.0%|26.4%|97.7%|
|16|LLaVa-1.6-Mistral|61.7%|30.1%|93.4%|
|17|Ovis2-2B|61.7%|27.3%|96.1%|
|18|LLaVa-1.6-Vicuna|53.7%|10.4%|96.9%|

## Metrics:

We denote images that contain the object as *positive* and images that do **not** contain the object as *negative*:

- **true negative rate (TNR)**: accuracy over all images *not* containing the object
- **true positive rate (TPR)**: accuracy over all images containing the object
- **accuracy (ACC)**: mean of TNR and TPR



