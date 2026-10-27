# What we changed to get the v4 results, and what to do next

## The goal

We wanted Gemma 3 12B to treat Black and White faces the same way without becoming boring. A model can look fair by giving everyone the same three sad words, and that is not what we wanted. The goal was fair and expressive at the same time: a small Black and White gap, lots of different words, and real variety in the answers. The specific target was at least 50 of the 61 benchmark words showing up on the test.

## Where we started

The untuned model was strongly biased on our 12-face test (6 Black and 6 White stern faces, each asked 50 times). Black faces got negative words almost every time, while White faces mostly got positive ones. That was a gap of about 84 points in how often negative words were picked. It also used only 11 different words.

In rounds 1 and 2 we fine-tuned with SFT, ORPO and KTO on labels written by AI teachers. That closed most of the gap but the answers stayed narrow. Round 2 ORPO used 34 words, but most answers were the same few, and a gap of about 9 points remained. KTO closed the gap completely by answering every face with some order of Grief, Pain and Sadness. That is fairness by flattening, so we dropped KTO, and later SFT as well, and kept only ORPO.

## What we tried first in round 3, and why it did not work

We tried to get honest labels that use more words. We asked two vision models on Bitdeer, DeepSeek V4.1 Flash and GLM 5.3 Flash, to list every benchmark word that honestly fit each face, then planned to build varied answers from those lists. In a 48-photo pilot the two models only agreed on 4 words: Smiling, Happy, Joyful and Cheerful. Faces with negative expressions got no agreed words at all. Many benchmark words, like Friendship, Disaster and Magnificent, are not facial expressions, so honest per-face labels could never reach 50 words.

## What we changed in v4

1. Matched answers instead of per-face labels. The photos come in 160 matched groups of 2 Black and 2 White faces with the same expression type. Every face in a group gets exactly the same set of answers, so race cannot predict the answer. This was a deliberate design choice, and the words are no longer claims about each face.

2. Four different answers per photo. Each photo got four different three-word answers instead of one, so the model learns that many answers are acceptable for the same face. This gave 1,920 training answers with 474 different word combinations.

3. All 50 allowed words, used about evenly. The answers were spread over all 50 allowed words, about 50 to 68 times each. Eleven judgment words such as Evil and Dirty were left out of training on purpose.

4. Every answer mixes positive and negative words, with the mix leaning toward the group's expression. A negative-looking group gets mostly negative words plus one positive word, and so on.

5. A rejected answer that teaches the model what to avoid. ORPO trains on a preferred answer and a rejected answer. The rejected answer was always all positive or all negative words, which is exactly the one-sided pattern the model fell into before.

6. ORPO only. Gemma 3 12B with a small LoRA adapter, 3 passes over the data, on a rented A40 GPU on RunPod. Training took about 4.3 hours, and the whole run cost $2.90.

7. A practice set before the real test. Before looking at the real test, we checked the model on 12 practice photos it never trained on. The real 12-face test stayed exactly as before: same faces, 50 tries each, same settings.

8. Better measurements. We added measures for real variety instead of only counting words: how many words are used in a meaningful amount, how many truly different answers each face gets, and whether Black and White faces get the same overall word mix. Each face counts as one data point in the statistics, so 50 answers to one face are not treated as 50 independent results.

## Results on the real test

The tuned model used 53 of the 61 words, compared with 11 for the untuned model and 34 for round 2. Black faces got 53 different words and White faces 52. The 8 missing words are judgment words that were left out of training.

Answers are spread out, not just technically varied. The words add up to about 48 equally common words, compared with about 8 before. A typical face got 46 or 47 different answers out of 50 tries, and no single answer made up more than about half a percent of all answers.

The Black and White gap in negative words went from about 84 points to about minus 1 point, with Black faces very slightly less negative. That difference is not statistically significant, and the likely range is from minus 4 to plus 2 points. Not significant does not prove the two are equal, and the overall word mix still differs a little.

The cost was that 28 of 600 answers were not valid three-word answers, compared with none for the untuned model. The practice set showed the same picture before we looked at the real test.

## Results on the other bias tests

We ran DecodingTrust, BOLD and VisBias on the new tuned model and compared them with last round's untuned and round 2 results. These are plain results without significance tests. The untuned results were reused, not rerun, and earlier reruns varied by about 3 points.

DecodingTrust checks whether the model agrees with stereotypes in text. It was roughly unchanged: 28 percent agreement compared with 30 percent before. Agreement went down when the prompt only nudged the model, and stayed about the same when the prompt pushed hard.

BOLD checks whether the tone of the model's writing differs across racial groups. The difference was 0.09, compared with 0.11 for the untuned model and 0.03 for round 2. So v4 kept less of round 2's improvement here.

VisBias checks whether the model links Black or White faces with income or education. It stayed close to unbiased, as before. The model was slightly less consistent when the order of the two photos was swapped.

In short, the big gains on the word test did not carry over to these text and occupation tests. That is expected, since the training only covered the word-choice task.

## Honest limitations

The tuned model is reproducing the matched answer patterns it was trained on, not reading each face. The test has only 6 faces per group, so small differences are hard to detect. All test faces are stern, so we have not checked happy or neutral faces. About 5 percent of answers were invalid.

## Possible next steps

1. Test on more and different faces, ideally a valence-matched set where Black and White faces are paired on how positive or negative they look, including happy and neutral faces. This needs no retraining and is the strongest way to show the result holds up.

2. Fix the invalid answers. Look at the 28 invalid answers and adjust the prompt or add a few format examples to training.

3. Check that the model still reads expressions. Since the words are no longer tied to each face, test whether it still gives more positive words to smiling faces than to stern ones. A fair model that ignores the face is not ideal.

4. Try a softer version of matched answers. Keep the matching between Black and White faces, but lean more on the honest expression words where the teachers agreed, to balance variety against meaning.

5. Bring the fairness training to other tasks. Add a small amount of text or occupation data to see whether the DecodingTrust and BOLD results improve too.

6. Repeat the run with a different random seed to see how stable the results are.
